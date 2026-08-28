# MindDrive Counterfactual 版本测试验证交接说明

本文是说明现有版本与原始 MindDrive 的框架差异、训练/推理计算流程、正确复现方法、输出字段、已知限制与验收项目。

## 1. 当前交付基线

建议测试时固定使用以下组合，避免训练与推理结构不一致：

```text
推理配置：adzoo/minddrive/configs/minddrive_qwen2_05B_infer_counterfactual_meta_aug.py
CF 权重：work_dirs/cf_valid_agents_action_expert_4gpu_bs1_10k/latest.pth
权重实际文件：iter_20000.pth
验证 infos：data/infos/b2d_infos_val.pkl

典型场景：
代表场景：v1/PedestrianCrossing_Town13_Route747_Weather19
代表 index：5404（frame_idx=497）
```

两个运行开关：

```bash
环境变量

MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=1
MINDDRIVE_CF_REPLACE_DECISION=1
```

含义：

- `MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=1`：使用原 MindDrive Action Expert 产生的 7 条 ego 候选，而不是 CF head 内置的规则候选。
- `MINDDRIVE_CF_REPLACE_DECISION=1`：用 CF risk scorer 选出的候选替换最终 `ego_fut_preds`。
- 第二个开关只有在第一个开关有效且 Action Expert 候选生成成功时才会替换决策。
- 只想观察 CF 输出但不改变原决策时，应将 `MINDDRIVE_CF_REPLACE_DECISION=0`。

注意：配置/目录名中包含 `meta_aug`，但当前训练配置和本次权重保存的配置均为：

```python
train_candidate_augmentation = False
```

代码支持 candidate augmentation，但这次交付权重不能宣称使用了该训练增强。

## 2. 原始 MindDrive 计算流程

原始 MindDrive 的主要推理数据流为：

```text
多相机图像 + 历史/车辆状态
        |
        v
图像 backbone / position embedding
        |
        +--------------------+
        |                    |
        v                    v
object head             map head
检测、类别、运动预测      车道/地图预测
        |                    |
        +---------+----------+
                  |
                  v
object/map visual queries
                  |
                  v
LM / meta-action / Action Expert
                  |
                  v
原始 ego_fut_preds（最终规划轨迹）
```

原版只对当前激活的 speed/path meta action 取对应 ego trajectory。周围 agent 的运动预测由原检测/运动 head 给出，不会针对未执行的 ego 候选重新响应。最终决策默认直接使用 Decision/Action Expert 结果。

## 3. CF 版本增加了什么

CF 版本保留原始 MindDrive 主干，在原模型输出之后增加一个候选条件化 reranking head：

```text
原 MindDrive object queries Q_obj
原 MindDrive map queries Q_map
                  |
                  v
scene_tokens = concat(Q_obj, Q_map)       (B, M, 896)
                  |
                  +-------------------------------+
                  |                               |
                  v                               v
7 条 ego candidates                 原 MindDrive agent futures
                  |                               |
                  +---------------+---------------+
                                  |
                                  v
                 MetaActionInteractionGraph
              ego-agent + agent-agent message passing
                                  |
                                  v
                         ResponsePredictor
                   relevance / speed / path logits
                                  |
                                  v
                    RuleBasedTrajectoryRealizer
                     candidate-conditioned agents
                                  |
                                  v
                        RuleBasedRiskScorer
                                  |
                                  v
                  argmin(total risk) 选择候选
                                  |
                     replace=1 时替换 ego_fut_preds
```

CF head 不是第二套感知模型。它直接复用原 MindDrive 的 scene tokens、检测框和基础 agent futures，只预测“如果 ego 采用候选 k，哪些 agent 相关、它们可能采取什么离散响应”。

## 4. Agent valid mask

### 4.1 训练阶段

训练使用 GT agents。agent 必须同时满足：

```text
未来有效步数 >= 2
类别属于 (0, 1, 2, 3, 7, 8)
```

对应类别为：

```text
0 car
1 van
2 truck
3 bicycle
7 pedestrian
8 others
```

traffic sign、traffic cone、traffic light 不进入 CF graph。筛选后只把 compact valid agents 送入 graph，而不是先处理所有 agent 再 mask loss。

### 4.2 推理阶段

推理使用原 MindDrive decoded agents。agent 必须满足：

```text
detection score >= 0.25
类别属于 (0, 1, 2, 3, 7, 8)
```

有效 agent 被 compact 后送入 graph。预测完成后再 scatter 回原来的 300 个 decoded agent 顺序，以保证结果字段与 `boxes_3d/scores_3d/labels_3d/trajs_3d` 对齐。

关键输出：

```text
cf_agent_valid_mask
cf_num_valid_agents
```

测试时不要把“300 个 decoded queries”误认为“300 个参与 CF 推理的 agent”。例如 5404 有 300 个 decoded agents，但只有 7 个 valid agents。

## 5. CF 训练计算流程

训练配置设置：

```python
counterfactual_train_only = True
```

因此：

- 原 MindDrive 主干仍然执行 forward，以生成 object/map queries。
- 原 map、detection、motion、LM/planning loss 不计算。
- 只有 `counterfactual_head.*` 参数允许更新。
- 当前约 1,627,534 个参数、42 个 parameter tensors 可训练；约 868,983,736 个原模型参数冻结。

每个样本的 CF 训练流程：

1. 从 `gt_attr_labels` 取前 12 维，reshape 为 6 步二维未来 offset。
2. 从后续 6 维取 future mask。
3. 根据未来有效步数和动态类别生成 valid-agent mask。
4. compact GT agents。
5. 将 agent offset `cumsum` 并加 GT box center，得到绝对 BEV future。
6. 将 GT ego future `cumsum`，得到绝对 BEV future。
7. 根据 ego/agent 几何距离、路径重叠和 TTC 产生 relevance 伪标签。
8. 根据 agent 自身 future 产生 speed/path meta-action 伪标签。
9. graph 编码场景、ego future、agent futures 与 ego meta embedding。
10. predictor 输出 relevance、7 类 speed、6 类 path logits。
11. 计算 BCE relevance loss 和 speed/path CE loss。

当前训练损失字段：

```text
loss_cf_rel
loss_cf_speed
loss_cf_path
loss_cf_real       # 当前权重 0.05
```

如果未来打开 `train_candidate_augmentation=True`，还会出现：

```text
loss_cf_aug_rel
loss_cf_aug_speed
loss_cf_aug_path
```

augmentation 的重要限制：替代 ego candidate 会重新计算 relevance label，但 speed/path target 仍由同一个 logged agent future 扩展而来；它不提供真实 counterfactual agent future ground truth。

某个 DDP rank 可能碰到零 valid-agent 样本。代码会返回连接到 CF trainable parameters 的可反传零 loss，避免：

```text
RuntimeError: element 0 of tensors does not require grad
```

多卡训练应保留：

```python
find_unused_parameters = True
```

## 6. CF 推理计算流程

### 6.1 Ego candidates

推荐设置 `MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=1`。Action Expert 产生 7 条 speed-conditioned ego trajectories，并附带 speed/path meta information。

如果该开关关闭或候选未生成，CF head 使用内部规则候选；两种 candidate source 不能混为同一实验。输出字段：

```text
cf_ego_candidate_source
cf_candidate_meta_actions
```

### 6.2 Candidate-conditioned graph

对每个 candidate `k`：

```text
输入：scene tokens、ego_future[k]、同一组 valid base agent futures、boxes、ego meta
输出：每个 valid agent 的 relevance/speed/path logits
```

graph 显式包含：

- ego → agent 边；
- agent 的 KNN agent → agent 边；
- 候选 ego future 的起终点、相对速度、距离；
- 可选的 ego speed/path meta embedding。

### 6.3 Agent response realization

predictor 的概率先离散化：

```python
speed_label = argmax(speed_logits)
path_label = argmax(path_logits)
```

RuleBasedTrajectoryRealizer 根据离散 speed/path 调整基础 agent future，但只在以下条件成立时采用调整结果：

```python
relevance >= 0.5
```

否则直接返回原始 base agent future。该硬门控会产生一个重要现象：不同 candidate 的 relevance/speed/path 概率可能明显不同，但只要 relevance 没跨过 0.5 或 argmax 未改变，最终 `agent_futures` 仍可能完全相同。这不是画图索引错误。

### 6.4 Risk 和候选选择

每个 candidate 的总风险为规则项加权和：

```text
total =
    8.0 * collision
  + 1.0 * ttc
  + 0.2 * interaction
  + 0.2 * comfort
  + 2.0 * progress
  + 1.0 * nominal deviation
  + map_rule
```

选择：

```python
selected_candidate = argmin(total)
```

该模块定位为 reranker，因此对偏离原 MindDrive 轨迹和损失进度进行了惩罚。即使所有 candidate 的 agent futures 相同，只要 ego futures 不同，risk 仍然可能不同。

## 7. 原版和 CF 版的行为差异总结

| 项目 | 原始 MindDrive | Counterfactual 版本 |
|---|---|---|
| 感知与地图主干 | 原始实现 | 完全复用原始实现 |
| Ego 规划 | 使用激活 meta action 的轨迹 | 同时评估 7 条 Action Expert 候选 |
| Agent future | 每帧一组原始预测 | 每个 ego candidate 一组 realized future |
| Ego-agent 交互 | 不显式进行候选条件化响应预测 | graph 显式预测 relevance/response |
| Agent 过滤 | 原检测结果 | score + dynamic class valid mask 后 compact |
| 候选评分 | 原 Decision/Action Expert 逻辑 | 规则 risk scorer 对 7 个场景 rerank |
| 最终轨迹 | 原 `ego_fut_preds` | replace=1 时替换为最低风险候选 |
| 训练参数 | 依原训练阶段而定 | 当前只训练 CF graph + predictor |
| 监督来源 | 原检测/规划/语言监督 | logged GT 几何伪标签，没有真实 CF future GT |

## 8. 主要输出字段

输出同时写入 `lane_results[0]` 和 `bbox_results[0]`：

```text
cf_selected_ego_future
cf_selected_meta_action
cf_candidate_meta_actions
cf_risk_scores
cf_ego_candidate_source
cf_counterfactual_scenes
cf_interaction_relevance
cf_response_meta_actions
cf_agent_valid_mask
cf_num_valid_agents
cf_replaced_decision_expert
decision_expert_ego_fut_preds
```

其中：

```text
cf_counterfactual_scenes[k].ego_future
cf_counterfactual_scenes[k].agent_futures
```

分别是 candidate `k` 实际送入 CF head 的 ego future 和最终 realized agent futures。

## 9. 正确测试连续驾驶序列

MindDrive 使用 temporal memory。测试某个中间 index 时，不能直接单帧启动并认为结果等价于真实驾驶序列。必须从该 scene 的 frame 0 开始，复用同一个 model instance 顺序推理。

代表场景完整范围：

```text
4907 = frame 0
5404 = frame 497
5418 = frame 511
```

连续推理并在 5404 保存图和数据：

```bash
cd /home/lilin/MindDrive

MINDDRIVE_CF_REPLACE_DECISION=1 \
MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=1 \
CUDA_VISIBLE_DEVICES=0 \
python tools/scan_live_cf_candidate_differences.py \
  --config adzoo/minddrive/configs/minddrive_qwen2_05B_infer_counterfactual_meta_aug.py \
  --checkpoint work_dirs/cf_valid_agents_action_expert_4gpu_bs1_10k/latest.pth \
  --infos data/infos/b2d_infos_val.pkl \
  --start 4907 \
  --end 5404 \
  --out work_dirs/analysis_cf_base_20k/live_case_study/continuous_seed0_4907_5404.json \
  --visualize-indices 5404 \
  --visualize-dir work_dirs/analysis_cf_base_20k/live_case_study/continuous_5404_data \
  --show-text \
  --seed 0 \
  --device-id 0
```

`--seed` 用于控制 Action Expert `do_sample=True` 引入的随机性。对比实验必须固定 seed、起始 frame、config、checkpoint 和环境变量。

单帧工具 `tools/live_cf_candidate_scenes_only.py` 适合快速检查，但从空 memory 启动，不应用于声明真实连续序列结果。

## 10. 5404 验证基准

使用连续 frame 0→497、seed 0 的当前结果：

```text
index：5404
frame_idx：497
valid agents：7 / 300 decoded
selected candidate：5
ego candidate 最大差异：约 4.67 m
agent future 最大差异：约 2.87 m
relevance 最大差异：约 0.30
```

各 candidate 的 affected agent IDs：

```text
candidate 0: [0, 4, 6]
candidate 1: [0, 4, 6]
candidate 2: [0, 3, 4, 6]
candidate 3: [0, 4, 6]
candidate 4: [0, 3, 4, 6]
candidate 5: [0, 4, 6]
candidate 6: [0, 4, 6]
```

保存的数据：

```text
work_dirs/analysis_cf_base_20k/live_case_study/continuous_5404_data/
  index_5404.png
  index_5404_plot_data.npz
  index_5404_plot_data.json
```

NPZ 包含：

```text
ego_candidates             (7, 6, 2)
base_agent_futures          (300, 6, 2)
realized_agent_futures      (7, 300, 6, 2)
boxes                       (300, 9)
scores / labels             (300,)
agent_valid_mask            (300,)
relevance                   (7, 300)
speed_labels/path_labels    (7, 300)
speed_probs                 (7, 300, 7)
path_probs                  (7, 300, 6)
trajectory_change           (7, 300)
affected_mask               (7, 300)
map_points                  (50, 11, 3)
risk_*                      (7,)
```

## 11. 建议验收项目

### A. 回归兼容性

1. CF head 关闭时，原 MindDrive 推理路径和关键指标不应改变。
2. `REPLACE_DECISION=0` 时，原 `ego_fut_preds` 不应被 CF 改写。
3. `REPLACE_DECISION=1` 时，保存的 `decision_expert_ego_fut_preds` 应等于替换前轨迹，最终 `ego_fut_preds` 应等于 `cf_selected_ego_future`。

### B. Shape 和索引对齐

1. `len(cf_agent_valid_mask) == len(boxes_3d) == len(scores_3d)`。
2. 每个 candidate 的 relevance/response/realized futures 与 decoded agent 顺序一致。
3. invalid agent scatter 后 relevance 为 0、response label 为 -1、realized future 保持 base future。
4. `cf_num_valid_agents == cf_agent_valid_mask.sum()`。

### C. Candidate 条件化有效性

1. 检查 7 条 ego candidates 不完全相同。
2. 比较 relevance/speed/path probability，而不能只比较最终 realized trajectory。
3. 对 5404 应观察到 candidate 2/4 比其他候选多出 affected agent 3。
4. 验证 `selected_candidate == argmin(cf_risk_scores['total'])`。

### D. 连续序列一致性

1. 从同一 scene frame 0 顺序推理。
2. 固定 seed 后重复运行，比较目标帧保存的 NPZ。
3. 不要把单帧冷启动结果与带 temporal memory 的连续结果混合汇报。

### E. 训练稳定性

1. 单卡和多卡均检查 zero-valid-agent batch。
2. loss 必须始终 `requires_grad=True`。
3. 仅 `counterfactual_head.*` 参数发生变化。
4. 检查 optimizer 实际 LR 为 `1e-4`，没有继承旧 `head_decay_rate=4.0`。
5. 保留 `find_unused_parameters=True`。

## 12. 已知限制和解释边界

1. 没有真实 counterfactual agent future 标注；response 是从 logged factual trajectory 和几何伪标签学习的。
2. Realizer 是规则模块，不是学习到的连续轨迹生成器。
3. relevance 使用 0.5 硬阈值；概率变化不一定转化成轨迹变化。
4. speed/path 使用 argmax；概率改变但类别不变时 realized future 可能不变。
5. 推理 compact packing 当前明确支持 `batch_size=1`；批量 B>1 尚未实现每样本变长 packing。
6. risk scorer 是规则 reranker，其权重不是端到端学习所得。
7. 当前 Action Expert 使用 sampling；必须固定 seed 才能做严格复现。
8. 配置名中的 `meta_aug` 不代表当前权重实际开启 candidate augmentation。
9. 约 89% 的已扫描帧中，7 组最终 agent futures 完全相同；需要同时报告中间概率差异和最终轨迹差异。

## 13. 关键代码位置

```text
mmcv/models/detectors/minddrive.py
  原模型与 CF head 的训练/推理接入、环境变量和决策替换

mmcv/models/counterfactual/counterfactual_head.py
  valid-agent compact、训练 loss、逐 candidate 推理和 scatter

mmcv/models/counterfactual/interaction_graph.py
  candidate-conditioned ego-agent/agent-agent 图计算

mmcv/models/counterfactual/response_predictor.py
  relevance/speed/path heads

mmcv/models/counterfactual/trajectory_realizer.py
  离散响应到 realized future 的规则实现

mmcv/models/counterfactual/risk_scorer.py
  candidate risk 分项和加权

tools/live_cf_candidate_scenes_only.py
  单帧快速推理与候选场景图

tools/visualize_cf_candidate_scenes_only.py
  affected/context 筛选和绘图

tools/scan_live_cf_candidate_differences.py
  连续序列扫描、目标帧画图与 NPZ/JSON 保存
```

## 14. 测试报告至少应记录的信息

每次实验至少记录：

```text
git commit 或代码快照
config 路径
checkpoint 路径及 iteration
两个 MINDDRIVE_CF_* 环境变量
CUDA/GPU
seed
scene folder
连续推理起始 index/frame
目标 index/frame
candidate source
decoded/valid agent 数量
7 条 ego candidates
每个 candidate 的 affected IDs
agent trajectory 最大差异
relevance/speed/path probability 最大差异
各 risk 分项和 selected candidate
是否替换最终决策
输出 JSON/NPZ/图片路径
```

只有同时记录这些信息，原 MindDrive 与 CF 版本、不同 checkpoint、单帧与连续序列之间的结果才具有可比性。
