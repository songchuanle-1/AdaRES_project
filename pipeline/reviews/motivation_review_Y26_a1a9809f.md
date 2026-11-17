# Motivation Review - AdaRES_project

2026-10-04，北京时间；用户请求“继续设计”，event租约a1a9809f。

## 1. Goal

当前目标文件完整内容：
```json
{
  "metric": {"name": "mIoU", "target": 0.515, "dataset": "ScanRefer"},
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
    "research_on_capacity_gap": true, "steady_ticks_before_design": 1,
    "fill_to_capacity": true, "design_buffer": 0, "max_candidates_per_design_tick": 2
  }
}
```
W2历史champion.5100，目标.5150，差.0050。最新用户限制优先：
GPU0保留给项目外，仅server-2 GPU1授权；不改goal/资源白名单/调度。

## 2. Story Status

| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| W2竞争指标器 | 当前champion，未达标 | 已完成30轮及冻结全验证 | 可靠提升与轻量Pareto未证明 |
| Y25训练软掩码反馈 | 本实现拒绝 | 全验证相对W2负向CI | 不能将TRAIN/EVAL差异宣称唯一因果 |
| Y23有效语言支持 | 实验已完成、收益不确定 | 全验证CI跨0 | 不是重开或调参的理由 |
| Y26被丢弃的qla postnorm | 实际路径缺陷确认 | TRAIN64及完整反传12张量 | 接回并不等于优化/精度改善 |

## 3. Experiment Assessment

| 实验 | 冻结全验证mIoU | 动机关系 | 本轮处理 |
|---|---|---|---|
| W2 | .510108 | 点质量竞争父模型 | 保持原文件和检查点 |
| B1 | .502122 | 标准指标器基线 | 后续冻结比较，不重跑训练 |
| Y18/Y17 | .507606/.504157 | 损失/描述一致性 | CI跨0，保留原结论 |
| Y23 | .507472 | 语言padding支持 | CI跨0，不改token/阈值 |
| Y25 | .494117 | TRAIN连续掩码证据反馈 | 相对W2全体/小/中目标明显有害，保持拒绝 |

## 4. Gap Analysis

源码中qla的LayerNorm结果未赋回，trained checkpoint六层weight全1、
bias全0；原native损失无其梯度。它位于语言条件化query进入self-attention/
竞争指标器的路径，任务相关但不证明归一化设计最优。
此前padding、指标器监督、mask反馈实验没有测试这条qla affine路径。

## 5. Design Rationale

Y26只接回已有qla postnorm，TRAIN/EVAL一致，0新增参数；
不是Y25软掩码反馈、Y23padding或G1监督的换名。
预测盲TRAIN64/256表达式：native/repeat/identity完全相同，
connected明显非恒等；点IoU .741665/.743230，RMS控制.746795。
因此不主张中心化特有任务优势，不按控制读数选择另一个模型。
完整loss初始上升+.07624trained8/+.03564released-init4；
后续缓存输入控制已将其主要定位到尺度变化，而非中心化，
保留不利结果并冻结同一Y26结构。真实训练才决定有效性。

## 6. Next Batch

仅Y26，见`pipeline/designs/exp_design_Y26_a1a9809f.md`。
本轮budget0，Y25正常deathwatch2/3，不伪造容量或重复preflight。
完成工程/机制验证后，下一次合法队列释放才审查准入；不填充RMS/center
变体。文献与新颖性未审查，不声称新颖。已有终态与路线调整决定保持。
