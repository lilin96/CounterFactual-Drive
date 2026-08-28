# Counterfactual MindDrive 论文主结果与消融实验测试 SOP

本文交给测试人员执行。目标是生成可用于论文的：

1. Bench2Drive 闭环 SOTA 对比表；
2. 开环规划与 CF 内部诊断表；
3. 三条核心消融：meta-action embedding、candidate-conditioned agent response、candidate source；
4. 可复现的日志、JSON、PKL、配置快照和统计结果。

框架细节先阅读 [COUNTERFACTUAL_HANDOFF_ZH.md](./COUNTERFACTUAL_HANDOFF_ZH.md)。

---

## 1. 最终要回答的论文问题

测试报告必须能够回答：

1. CF reranker 相比原 MindDrive 是否提高闭环 Driving Score 和 Success Rate？
2. 提升主要来自减少碰撞，还是以路线完成度、舒适性或其他违规为代价？
3. meta-action embedding 是否提供有效条件信息？
4. candidate-conditioned agent response 是否优于所有 candidate 共用一套 agent future？
5. 使用 MindDrive Action Expert 候选是否优于规则候选？
6. 提升是否在多随机种子/多次 CARLA 重复下稳定，而非单次偶然结果？

内部 rule risk 降低不能单独作为 SOTA 证据。SOTA 主结论必须来自官方 Bench2Drive 闭环指标；开环和内部 risk 只用于补充分析。

---

## 2. 固定实验条件

除被消融变量外，所有实验固定：

```text
MindDrive backbone/checkpoint initialization
CF hidden_dims=256
future_steps=6
agent_score_threshold=0.25
valid_agent_labels=(0,1,2,3,7,8)
relevance_threshold=0.5
训练数据及 train/val split
训练 iterations、batch size、optimizer、LR
CARLA 版本
Bench2Drive 版本与 route split
传感器配置
traffic manager 参数
评测 route 顺序
重复次数和随机种子
```

基准配置与权重：

```text
Base config:
  adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py

Full CF config:
  adzoo/minddrive/configs/minddrive_qwen2_05B_infer_counterfactual_meta_aug.py

Base checkpoint:
  ckpts/minddrive_rltrain.pth

Current full CF checkpoint:
  work_dirs/cf_valid_agents_action_expert_4gpu_bs1_10k/latest.pth
  实际链接到 iter_20000.pth

Validation infos:
  data/infos/b2d_infos_val.pkl
```

当前 CF checkpoint 的训练快照中：

```python
use_ego_meta_embedding=True
train_candidate_augmentation=False
```

因此论文中不能把当前权重描述成“使用 candidate augmentation 训练”。

---

## 3. 必须先完成的实现检查

### 3.1 No-meta 实验必须单独训练

`use_ego_meta_embedding` 改变 graph 层输入维度和参数结构。不能在 full checkpoint 上推理时简单关闭，否则会产生 checkpoint shape mismatch 或随机初始化层。

必须准备：

```text
CF-Meta checkpoint:     use_ego_meta_embedding=True
CF-NoMeta checkpoint:   use_ego_meta_embedding=False
```

两者从同一个 `minddrive_rltrain.pth` 初始化，使用完全相同的训练数据、seed、LR、batch size 和 iterations。

### 3.2 Unified-agent-future 需要明确推理开关

该消融定义为：

```text
所有 ego candidate 都使用同一组原 MindDrive base agent futures；
不执行 ResponsePredictor 的 speed/path 输出到 RuleBasedTrajectoryRealizer 的轨迹修改；
risk scorer 对 candidate ego 与统一 base agent futures 评分。
```

推荐新增显式配置项：

```python
use_candidate_conditioned_agent_response=True / False
```

关闭时应满足：

```python
realized_agents[k] == base_agents
```

对所有 candidate 和 valid agents 均严格成立。

这个消融可以复用 full CF checkpoint，只改变推理路径，因而可隔离 agent response realization 的贡献。不要通过把 `relevance_threshold` 改成极大值来冒充该消融；正式实现应保存开关和值到结果元数据。

### 3.3 Candidate-source 比较必须控制候选数量

Action Expert 当前产生 7 条候选；规则 fallback 默认产生：

```text
7 speed × 6 path = 42 条规则候选
```

直接比较 7 vs 42 不公平，候选越多本身就增加 reranker 搜索空间。

主消融必须报告固定 `K=7` 的公平版本：

```text
Action Expert: 原生 7 条 speed-conditioned candidates
Rule K=7: 从规则候选中预先固定 7 个动作，所有样本使用同一集合
```

建议 Rule K=7 使用相同 path command，仅改变 7 个 speed action，使语义与 Action Expert 7-way 输出对齐。可以把 Rule K=42 放在附录作为额外实验，但不能代替 K=7 主消融。

测试人员启动大规模实验前，必须让开发人员确认以上两个开关和 Rule K=7 已实现，并用 10 帧 smoke test 验证输出。

---

## 4. 实验矩阵

### 4.1 论文主表

| ID | 方法 | Meta emb. | Agent response | Candidate source | Replace decision |
|---|---|---:|---:|---|---:|
| M0 | 原始 MindDrive-0.5B | – | – | 原 Decision Expert | 0 |
| M1 | CF-MindDrive（Full） | ✓ | ✓ | Action Expert K=7 | 1 |

外部对比方法按官方同版本协议引用或复测：

```text
UniAD-Tiny
UniAD-Base
VAD
ORION-7B
MindDrive-0.5B
MindDrive-3B
CF-MindDrive-0.5B (ours)
```

仓库 README 中已有参考值，但正式论文必须核对这些值的 route split、CARLA/Bench2Drive 版本和指标定义。不同协议的数字不能放进同一主表而不加说明。

### 4.2 核心消融表

| ID | Meta emb. | Candidate-conditioned response | Candidate source | 是否需重训 |
|---|---:|---:|---|---:|
| A0 | ✓ | ✓ | Action Expert K=7 | Full checkpoint |
| A1 | ✗ | ✓ | Action Expert K=7 | **必须重训** |
| A2 | ✓ | ✗，统一 base agent future | Action Expert K=7 | 否，推理消融 |
| A3 | ✓ | ✓ | Rule K=7 | 否，推理消融 |

推荐补充一行用于说明 CF head 本身和“是否替换最终决策”的区别：

| ID | Meta emb. | Response | Candidate source | Replace |
|---|---:|---:|---|---:|
| A4 | ✓ | ✓ | Action Expert K=7 | 0（只分析，不控制） |

A4 的闭环控制行为原则上应与 M0 一致；如果不一致，说明 CF 接入产生了非预期副作用。

### 4.3 可选敏感性实验

篇幅允许时增加：

```text
relevance threshold: 0.3 / 0.5 / 0.7
agent score threshold: 0.2 / 0.25 / 0.4
candidate K: 3 / 5 / 7
risk without interaction term
risk without nominal/progress regularization
```

这些是敏感性分析，不应混入三项核心消融。

---

## 5. 每个实验的目录规范

统一使用：

```text
paper_results/
  <experiment_id>/
    config.py
    checkpoint_info.txt
    env.txt
    git_status.txt
    open_loop/
      results.pkl
      trajectory_interaction_metrics.json
      trajectory_interaction_metrics.md
      cf_risk_summary.json
      cf_risk_summary.md
    closed_loop/
      seed_0/
      seed_1/
      seed_2/
    aggregate/
      metrics.json
      metrics.csv
      report.md
```

实验开始前保存：

```bash
mkdir -p paper_results/<experiment_id>
cp <CONFIG> paper_results/<experiment_id>/config.py
readlink -f <CHECKPOINT> > paper_results/<experiment_id>/checkpoint_info.txt
git status --short > paper_results/<experiment_id>/git_status.txt
env | sort > paper_results/<experiment_id>/env.txt
```

`env.txt` 可能包含敏感 token，交付前应删除 W&B/API/代理等凭据；论文实验必需的两个 `MINDDRIVE_CF_*` 值需保留在单独的无敏感信息记录中。

---

## 6. Phase 0：10 帧 smoke test

每个实验先跑相同的连续 10 帧，不通过则不得启动完整评测。

检查项：

```text
模型和 checkpoint 无 CF shape mismatch
候选数量符合实验定义
selected_idx == argmin(total risk)
valid mask 与 boxes/scores/labels 长度一致
无 NaN/Inf
M0/A4 未替换原决策
A0/A1/A2/A3 在 replace=1 时确实替换
A2 对所有 candidate 满足 realized == base future
A3 确实为 Rule K=7，不是默认 42 candidates
```

建议使用连续序列工具而不是随机单帧，因为 MindDrive 有 temporal memory：

```bash
MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=1 \
MINDDRIVE_CF_REPLACE_DECISION=1 \
CUDA_VISIBLE_DEVICES=0 \
python tools/scan_live_cf_candidate_differences.py \
  --config <CONFIG> \
  --checkpoint <CHECKPOINT> \
  --infos data/infos/b2d_infos_val.pkl \
  --start 4907 \
  --end 4916 \
  --out paper_results/<ID>/smoke.json \
  --seed 0
```

对于 A2/A3，应使用对应新增 config 开关；不要只改输出文件名。

---

## 7. Phase 1：开环完整验证集

### 7.1 原始 MindDrive M0

```bash
cd /home/lilin/MindDrive

MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=0 \
MINDDRIVE_CF_REPLACE_DECISION=0 \
CUDA_VISIBLE_DEVICES=0 \
python adzoo/minddrive/test.py \
  adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py \
  ckpts/minddrive_rltrain.pth \
  --out paper_results/M0/open_loop/results.pkl \
  --seed 0 \
  --deterministic
```

### 7.2 Full CF A0/M1

```bash
MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=1 \
MINDDRIVE_CF_REPLACE_DECISION=1 \
CUDA_VISIBLE_DEVICES=0 \
python tools/test_cf_replace_decision.py \
  adzoo/minddrive/configs/minddrive_qwen2_05B_infer_counterfactual_meta_aug.py \
  work_dirs/cf_valid_agents_action_expert_4gpu_bs1_10k/latest.pth \
  --out paper_results/A0/open_loop/results.pkl \
  --seed 0 \
  --deterministic
```

### 7.3 其他消融

保持同一命令框架，仅替换：

```text
A1: no-meta config + no-meta checkpoint
A2: full config/checkpoint + use_candidate_conditioned_agent_response=False
A3: full config/checkpoint + Rule K=7 source
A4: full config/checkpoint + REPLACE_DECISION=0
```

注意：`adzoo/minddrive/test.py` 的 dataloader 使用连续 non-shuffle sampler。不要对验证集 shuffle，也不要将互不连续的 frame 随意分配给普通 sampler。

### 7.4 开环指标生成

对每个 CF 实验：

```bash
python tools/analyze_cf_predictions.py \
  --results paper_results/<ID>/open_loop/results.pkl \
  --infos data/infos/b2d_infos_val.pkl \
  --out-dir paper_results/<ID>/open_loop

python tools/eval_counterfactual_risk.py \
  --cf paper_results/<ID>/open_loop/results.pkl \
  --out-dir paper_results/<ID>/open_loop/risk

python tools/plot_interaction_roc.py \
  --results paper_results/<ID>/open_loop/results.pkl \
  --infos data/infos/b2d_infos_val.pkl \
  --out-dir paper_results/<ID>/open_loop/interaction_roc
```

开环至少汇报：

```text
Planning ADE / FDE 或项目统一定义的 2s L2
agent ADE / FDE
factual relevance ROC-AUC / PR-AUC
response speed/path accuracy
base vs selected total risk（补充指标）
candidate change rate
affected-agent rate
```

重要：`tools/analyze_cf_predictions.py` 中的 interaction/response 指标是 factual pseudo-label validation，不是真实 counterfactual ground truth。论文表头和正文必须写清楚。

---

## 8. Phase 2：Bench2Drive 闭环主评测

### 8.1 使用官方评测 split

必须使用与论文 SOTA 表一致的官方 Bench2Drive evaluation routes。仓库中的：

```text
data/routes/rollout_routes.xml
```

是 rollout/collection 路线，不能未经确认直接作为论文 SOTA 闭环路线。

开始前在实验记录中写明：

```text
Bench2Drive commit/version
CARLA version
route XML 路径与 SHA256
总 route 数
weather/scenario 覆盖
repetitions
```

### 8.2 重复次数

最低要求：

```text
每条 route repetitions=5
实验 seeds=0,1,2
```

如果官方协议只定义固定 seed，则遵循官方协议，并额外在附录报告 3-seed 稳定性。所有方法必须采用相同 repetitions 和 seed 集合。

### 8.3 单 worker 命令模板

仓库现有 `run_collection.sh` 已设置 `REPETITIONS=5`，但它面向数据收集。用于论文评测前，应复制为独立 eval 脚本并固定输出目录，避免 collection queue 和 decode 步骤污染结果。

底层调用模板：

```bash
MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=<0_or_1> \
MINDDRIVE_CF_REPLACE_DECISION=<0_or_1> \
CUDA_VISIBLE_DEVICES=<GPU> \
python adzoo/minddrive/rollout.py \
  <CONFIG> \
  --routes=<OFFICIAL_EVAL_ROUTES_XML> \
  --checkpoint=paper_results/<ID>/closed_loop/seed_<SEED>/result.json \
  --port=<CARLA_PORT> \
  --traffic_manager_port=<TM_PORT> \
  --repetitions=5 \
  --resume \
  --use_carla
```

这里 `--checkpoint` 是 leaderboard 统计 JSON，不是模型权重。模型权重路径由 agent/config 加载，测试前必须核对 `team_code/minddrive_b2d_agent.py` 实际读取的 `ckpt_path`。

每个实验使用独立：

```text
CARLA port
traffic manager port
SAVE_PATH
checkpoint JSON
日志文件
```

不得让两个实验复用同一个 `--checkpoint` JSON 或 `--resume` 目录。

### 8.4 闭环主指标

从 leaderboard `global_record` 汇总：

```text
Driving Score ↑                 score_composed
Route Completion ↑              score_route
Infraction Penalty ↑            score_penalty
Success Rate ↑                  Perfect/Completed 按官方定义计算
Collision pedestrian ↓          / km
Collision vehicle ↓             / km
Collision layout/static ↓       / km
Red-light infraction ↓           / km
Stop-sign infraction ↓           / km
Off-road / route deviation ↓
Route timeout ↓
Agent blocked ↓
```

论文主表至少放：

```text
2s L2 ↓
Driving Score ↑
Success Rate ↑
```

消融表优先放：

```text
Driving Score ↑
Route Completion ↑
Success Rate ↑
Vehicle/Pedestrian/Static collision ↓
```

### 8.5 统计显著性

不能只对 global mean 做比较。保留每条 route、每个 repetition 的结果，按配对 route 计算：

```text
mean ± std
95% bootstrap confidence interval
paired bootstrap 的 A0 - Ax 差值区间
```

如果 95% CI 跨 0，应写“趋势改善但未达到统计显著”，不能写“显著优于”。

CARLA crash、timeout 和失败 route 不得静默删除。必须报告：

```text
planned runs
completed runs
crashed runs
retried runs
最终纳入统计的规则
```

---

## 9. 三项核心消融的解释规则

### 9.1 Meta-action embedding：A0 vs A1

唯一允许变化：

```text
use_ego_meta_embedding=True → False
```

需要比较：

```text
闭环 DS/SR/collision
factual relevance AUC
speed/path response accuracy
candidate 间 probability span
最终 agent-future change rate
```

如果 no-meta 参数量略少，应同时报告 trainable parameter count；不要把参数量差异隐藏。

### 9.2 Agent response：A0 vs A2

A2 中所有 candidate 共用 base agent future。保持：

```text
同一 ego candidates
同一 valid agents
同一 risk weights
同一 checkpoint
同一 relevance predictor
```

只禁止 speed/path response 对 agent trajectory 的 realization。重点观察：

```text
collision / TTC 是否恶化
selected candidate 分布是否改变
Driving Score / Success Rate 是否下降
```

必须用单元断言证明 A2：

```python
np.max(abs(realized_agent_futures[k] - base_agent_futures)) == 0
```

### 9.3 Candidate source：A0 vs A3

保持 K=7、同一 CF checkpoint 和 risk scorer，仅改变 ego candidate 几何来源：

```text
Action Expert learned candidates
vs.
Rule K=7 candidates
```

额外汇报候选集合质量：

```text
oracle min-L2（7 候选中离 GT 最近者）
candidate diversity（pairwise trajectory distance）
selected L2
selected/oracle gap
候选越界率、低进度率
```

否则最终差异无法判断来自 reranker 还是候选集合本身。

---

## 10. 论文表格模板

### 10.1 SOTA 主表

| Method | Params | 2s L2 ↓ | Driving Score ↑ | Success Rate ↑ |
|---|---:|---:|---:|---:|
| UniAD-Tiny | | | | |
| UniAD-Base | | | | |
| VAD | | | | |
| ORION-7B | | | | |
| MindDrive-0.5B | | | | |
| MindDrive-3B | | | | |
| CF-MindDrive-0.5B (ours) | | | | |

### 10.2 核心消融表

| ID | Meta | Response | Source | L2 ↓ | DS ↑ | RC ↑ | SR ↑ | Collision ↓ |
|---|:---:|:---:|---|---:|---:|---:|---:|---:|
| A1 | | ✓ | AE-7 | | | | | |
| A2 | ✓ | | AE-7 | | | | | |
| A3 | ✓ | ✓ | Rule-7 | | | | | |
| A0 | ✓ | ✓ | AE-7 | | | | | |

建议将 A0 放最后并加粗。

### 10.3 CF 机制分析表

| ID | Rel. AUC ↑ | Speed Acc ↑ | Path Acc ↑ | Changed frames ↑ | Mean affected agents | Mean risk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| A1 | | | | | | |
| A2 | | | | 0 | | |
| A3 | | | | | | |
| A0 | | | | | | |

---

## 11. 最终验收清单

测试负责人提交结果前逐项勾选：

```text
[ ] 每个实验保存了 config 和解析后的关键参数
[ ] 每个 checkpoint 路径、iteration、哈希已记录
[ ] MINDDRIVE_CF_* 环境变量已记录
[ ] A1 使用独立训练的 no-meta checkpoint
[ ] A2 有 unified-future 数值断言
[ ] A3 使用公平 Rule K=7，而非默认 K=42
[ ] M0/A4 最终决策未被 CF 替换
[ ] open-loop 使用完全相同 infos 和 sampler
[ ] closed-loop 使用完全相同官方 routes/repetitions/seeds
[ ] crash/timeout 没有被静默删除
[ ] 逐 route 原始 JSON 已保留
[ ] 汇报 mean/std/95% CI
[ ] SOTA 数字协议和来源逐项核对
[ ] 内部 risk 没有被包装成真实安全 ground truth
[ ] factual pseudo-label 指标没有被描述为 counterfactual GT
```

---

## 12. 推荐执行顺序

```text
1. 开发确认 A2 和 Rule K=7 开关
2. 训练 A1 no-meta checkpoint
3. 四组 10-frame smoke test
4. 四组完整 open-loop
5. 汇总 L2、interaction、response 和 risk
6. 先跑 M0/A0 小规模闭环子集检查控制链
7. 跑 M0/A0 官方完整闭环，确认主结论
8. 跑 A1/A2/A3 完整闭环消融
9. 统计 paired CI、crash 和 per-scenario 结果
10. 填主表、消融表和附录敏感性表
```

优先级上，闭环 M0 vs A0 是论文主结论，A0/A1/A2/A3 是核心消融。可视化和内部 risk 图应在这些结果稳定后再制作。
