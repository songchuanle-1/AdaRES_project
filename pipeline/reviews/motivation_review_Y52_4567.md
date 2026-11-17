# Motivation Review - AdaRES-Y52
实际整理时间：2026-10-16T16:10:31.428737+08:00。前置预注册与测试各有独立时间；本整理不伪装成事前文件。
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
历史CSV Y36 .5110/B1 .5020；目标.515未达标。Y50首次原生验证约.312，仅是早期值。
## 2. Story Status
| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| 最终indicator直接回归质量 | 不成立 | 实际loss_indi为匹配CE；score扰动不改变固定推理 | 不能宣称直接修复选择器 |
| 辅助质量目标与点级IoU口径不同 | CPU支持 | 8TRAIN场景/32表达/每表达全部128mask；47/4096筛选差异 | 原生匹配后频率未知 |
| 修正口径改善共享query并提升选择 | 未验证 | 独立CPU原型及梯度对照 | 全模型原生测试、资源与训练结果 |
## 3. Experiment Assessment
| Experiment | mIoU | 与动机的关系 | 对故事线贡献 |
|---|---|---|---|
| W2 | 冻结FIRST .5099093531 | 科学参照 | 不被未训练原型取代 |
| Y36 | 历史CSV .5110 | 目标检查数值champion | 不等于显著优于W2 |
| Y50 | 首次约.312 | 高阶共同覆盖训练中 | 非最终效果或项目最佳 |
| Y52 | 未训练 | 已有辅助目标的测度一致性 | 新候选仅CPU组件准备 |
## 4. Gap Analysis
等权超点IoU、点质量加权硬标签IoU、真实点目标IoU已分别计算。平均绝对质量目标差.00865423，最大.48834160；这些不是mIoU收益。超点内部混合贡献均值.000017484，不能与占用加权混为一谈。
## 5. Design Rationale
保持最终CE、选择器、mask、权重和固定协议；只将既有辅助MSE的标签测度改为精确点交并计数。不是Y19/Y32全query软indicator目标、Y29排序损失或Y48所有权目标重开；不宣称文献新颖性。
## 6. Next Batch
同一Y52继续：原型冻结清单见pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json。全模型匹配/梯度和资源未运行，不准入，不自动提交。
