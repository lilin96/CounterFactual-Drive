# Counterfactual MindDrive 论文实验脚本使用说明

本工具把 SOTA 基线、完整 Counterfactual 模型和主要消融实验放在同一张实验矩阵中，并统一完成配置检查、短序列冒烟测试、全量开环推理、指标分析和闭环任务生成。

## 1. 实验定义

实验矩阵位于 `configs/cf_paper_eval_matrix.py`。

| ID | 实验 | Meta action embedding | Agent future | Candidate source | 是否替换决策 |
|---|---|---:|---|---|---:|
| M0 | 原始 MindDrive-0.5B | 无 CF | 原始预测 | Decision Expert | 否 |
| A0 | 完整 CF | 开 | candidate-conditioned response | Action Expert K=7 | 是 |
| A1 | 去掉 meta action embedding | 关 | candidate-conditioned response | Action Expert K=7 | 是 |
| A2 | 使用统一 agent future | 开 | 所有 candidate 复用同一 base future | Action Expert K=7 | 是 |
| A3 | 更换 candidate source | 开 | candidate-conditioned response | Rule speed-only K=7 | 是 |
| A4 | 只分析、不接管决策 | 开 | candidate-conditioned response | Action Expert K=7 | 否 |

A1 改变了训练时的输入表示，必须用 `minddrive_qwen2_05B_infer_cf_no_meta.py` 对应的、重新训练得到的 checkpoint。脚本故意不提供默认 checkpoint；不要用 A0 checkpoint 冒充 A1 结果。A2 和 A3 是推理干预实验，可复用 A0 checkpoint，但论文中应明确标注为 inference-time ablation。

## 2. 准备工作

在仓库根目录执行：

```bash
cd /home/lilin/MindDrive
conda activate MindDrive
```

确认矩阵中的 infos、配置和 checkpoint 路径真实存在。闭环实验还需要按原项目要求设置 `CARLA_ROOT`、`PYTHONPATH` 和 Bench2Drive route XML。

## 3. 检查实验配置

列出实验：

```bash
python tools/run_cf_paper_experiments.py --stage list
```

检查除 A1 外的配置和 checkpoint：

```bash
python tools/run_cf_paper_experiments.py \
  --experiment M0,A0,A2,A3,A4 \
  --stage validate
```

检查 A1 时显式传入无 meta 模型：

```bash
python tools/run_cf_paper_experiments.py \
  --experiment A1 \
  --stage validate \
  --checkpoint work_dirs/YOUR_NO_META_RUN/latest.pth
```

## 4. 短序列冒烟测试

先在 4907–4915 上检查 candidate 数量和分支差异：

```bash
python tools/run_cf_paper_experiments.py \
  --experiment A0,A2,A3,A4 \
  --stage smoke \
  --gpu 0
```

脚本会自动检查：每个 CF 样本是否恰好有 7 个 candidate；A2 的各 candidate agent future 是否完全相同。输出位于 `paper_results/<ID>/smoke/`。

如目录已经存在，使用 `--resume` 跳过已有结果；只有确定需要重算时才使用 `--overwrite`。`--dry-run` 可只打印命令。

## 5. 全量开环评测与分析

运行全量 val 推理：

```bash
python tools/run_cf_paper_experiments.py \
  --experiment M0,A0,A2,A3,A4 \
  --stage open_loop \
  --gpu 0
```

A1 单独指定 checkpoint：

```bash
python tools/run_cf_paper_experiments.py \
  --experiment A1 --stage open_loop --gpu 0 \
  --checkpoint work_dirs/YOUR_NO_META_RUN/latest.pth
```

随后生成 CF 轨迹、风险和 interaction ROC 指标：

```bash
python tools/run_cf_paper_experiments.py \
  --experiment A0,A2,A3,A4 \
  --stage analyze

python tools/run_cf_paper_experiments.py \
  --experiment A1 --stage analyze \
  --checkpoint work_dirs/YOUR_NO_META_RUN/latest.pth
```

注意：命令行 `--checkpoint` 会覆盖本次所选全部实验。因此 A1 必须单独调用；A0/A2/A3/A4 使用矩阵默认 checkpoint。`--stage all` 等价于依次执行 smoke、open_loop、analyze，不会自动启动 CARLA。

## 6. 闭环任务

为某个实验生成三个 seed 的 CARLA 命令：

```bash
python tools/run_cf_paper_experiments.py \
  --experiment A0 \
  --stage closed_loop_jobs \
  --gpu 0 \
  --routes /absolute/path/to/official_routes.xml
```

检查生成的 `paper_results/A0/closed_loop/jobs.json` 和 `run_jobs.sh`，确认 CARLA 端口、环境变量和 route 文件后再执行：

```bash
bash paper_results/A0/closed_loop/run_jobs.sh
```

默认 seed 为 0、1、2，每个 seed 重复 5 次。脚本中的三个任务默认顺序执行；如需并行，必须给每个任务分配独立 GPU、CARLA server、RPC port 和 traffic-manager port。

## 7. 汇总论文表格

完成实验后运行：

```bash
python tools/aggregate_cf_paper_results.py \
  --root paper_results \
  --experiments M0,A0,A1,A2,A3,A4
```

输出 `paper_results/paper_summary.csv` 和 `paper_results/paper_summary.md`。闭环分数按 seed 给出均值和样本标准差；`perfect_route_rate` 明确定义为官方结果中 `status == "Perfect"` 的 route 比例，不应与只完成路线的比例混用。

## 8. 结果有效性检查

- 每个实验保存 `experiment.json`、实际 config 副本、checkpoint SHA256 和 Git 工作区状态，提交论文结果时一并归档。
- 比较 A0/A1 时只改变 meta embedding，并使用各自匹配训练权重；比较 A0/A2 时只改变 agent-response 分支；比较 A0/A3 时保持 K=7，只改变 candidate source。
- 所有实验使用相同 val infos、route 集、seed、重复次数、CARLA/traffic 配置和指标脚本。
- M0 没有 CF 输出，因此 CF 风险与 interaction 分析会跳过；M0 仍需报告原始开环规划指标和闭环 DS/RC/IP。
- 如果某项指标显示 `--`，表示对应分析文件或闭环结果不存在，应先检查该实验的 `run.log`，不要把缺失值当作 0。
