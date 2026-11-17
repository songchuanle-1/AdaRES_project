# Motivation Review - AdaRES Y54
时间：2026-10-17T13:14:22.508061+08:00；原生诊断支持，尚无训练收益。
## 1. Goal
当前目标.515，历史CSV冠军Y36 .5110、差.0040。完整目标如下：
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
## 2. Story Status
| 假设 | 状态 | 缺口 |
|---|---|---|
| 语言padding进入前向置信度Y53 | 源码+8TRAIN/32表达反证，拒绝 | 不实现、不重开Y23 |
| 辅助点测度Y52 | 有限原生证据，待限定提交许可 | 未训练、收益未知 |
| 细粒度3D支持有效性Y54 | 原生双场景有真实虚构支持及匹配干预分离 | 独立实现/反向/恢复/资源/训练效果 |
## 3. Experiment Assessment
Y50当前固定30实现因相对W2中目标伤害规则拒绝；不由总体均值的微小基线差异宣称收益。Y53的新反证不是历史Y23损失支持问题的重新评估。
## 4. Gap Analysis
可信置信度需要证据来自真实3D支持，不能依赖另一场景造成的padding。当前全有效表达负控制变化0，但缺少独立原型的原生反向和可执行门禁。
## 5. Design Rationale
只把现有batch_mask沿native fine gather传递到空间邻域和fine attention，排除不存在的支持；不增加头/权重、不修改语言/匹配/损失或512/128超参数，不与Y52叠加。
## 6. Next Batch
仅同一Y54独立七文件准备，见对应实验设计；通过全部门禁且获限定提交许可之前不入队。原生诊断：pipeline/analyses/y54_fine_support_native_fde0_fde01739a9864bae9bb342a577235e66.json
