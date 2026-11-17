# Motivation Review - AdaRES-Y50
实际整理时间：2026-10-16T14:08:33.053814+08:00。科学判据先见本轮13:39起的Y50预注册；此文是测试后完整整理，不伪造事前时间。

## 1. Goal
```json
{
  "metric": {
    "name": "mIoU",
    "target": 0.515,
    "dataset": "ScanRefer"
  },
  "baseline_experiment_id": "AdaRES-B1",
  "description": "AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.",
  "constraints": [
    "Use server-2 GPUs 0 and 1 only; never use server-6.",
    "All experiments launched after AdaRES-B3 must use exactly one GPU; run at most one independent experiment on GPU 0 and one on GPU 1.",
    "Do not design or queue a successor while AdaRES-B3 is running; after B3 reaches a terminal state, admit at most two structurally independent single-GPU candidates.",
    "Use a fixed 30-epoch protocol for the B2 and B4 extension. Resume both from their epoch-20 last checkpoints without resetting the optimizer or LR scheduler; do not tune the epoch count on validation.",
    "Train from the released SPFormer sparse-UNet backbone; the decoder is authored in this project.",
    "Report mIoU, Acc@0.25, Acc@0.5, active queries, latency, and peak memory.",
    "Do not tune thresholds, temperatures, loss weights, feature subsets, or the fixed 30-epoch protocol on ScanRefer validation."
  ],
  "banned": [
    "Do not use server-6.",
    "Do not report the released b005/CFNet checkpoints as AdaRES-B0 results.",
    "Do not scan the ranking margin or the occupancy loss weight on validation."
  ],
  "pipeline_policy": {
    "research_on_capacity_gap": true,
    "steady_ticks_before_design": 1,
    "fill_to_capacity": true,
    "design_buffer": 0,
    "max_candidates_per_design_tick": 2
  }
}
```
历史CSV champion Y36 .5110，目标.515未达标；本轮没有新的验证集指标。

## 2. Story Status
| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| 两两候选重叠缺失 | 已否定Y47实现 | W2本来有完整两两重叠 | 不重开 |
| 全局单mask连通读数 | 当前Y49实现拒绝 | 高成本、扩样0见证 | 一般机制未证伪 |
| 高阶共同覆盖可显式补充指示器 | 有界信息路径成立 | 等Gram父输出0差/新统计非零；native控制 | 训练收益未知、提交授权待定 |

## 3. Experiment Assessment
| Experiment | mIoU | 与动机的关系 | 对故事线贡献 |
|---|---|---|---|
| W2 | 当前冻结FIRST .5099093531 | 科学参照 | 保留，不被未训练路径取代 |
| Y48 | FIRST .5034034117 | 目标一致性 | 当前实现小目标组伤害，已拒绝 |
| Y49 | 未训练 | 单mask全局信息 | 当前实现资源不合格，非普遍证伪 |
| Y50 | 未训练/未知 | 多候选高阶共同覆盖信息 | 验收完成不等于性能提升 |

## 4. Gap Analysis
仍缺同协议已训练Y50与W2/B1配对证据；native特征相关不证明因果收益。官方unique/multiple元数据仍未知。

## 5. Design Rationale
固定面积与所有pairwise Gram仍可改变三重及以上共同覆盖，故本候选不是Y47换名、Y29排序损失、Y40词语对齐或Y49拓扑。
FG/BG条件二阶中心化竞争覆盖从mask联合分布进入最终置信度，保持原mask、loss和single selector。

## 6. Next Batch
见exp_design_Y50_1cb1.md；提交授权前不入队，不用旧HEAD伪绑定。
