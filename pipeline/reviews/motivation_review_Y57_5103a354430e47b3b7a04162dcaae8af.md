# Motivation Review - AdaRES_project / Y57
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
历史规范冠军Y36 .5110，目标.5150，差.0040；Y36新配对评估正在执行，不预写结果。
## 2. Story Status
同一Y57均值混叠假设，续做上轮完整256通道方案，不创建新ID或重开历史拒绝路线。
## 3. Experiment Assessment
上轮Y52 .509708428366/Y54 .506342952986/W2 .510107995189/B1 .502221764356为全量FIRST，
不是本轮新结果；Y52/Y54对W2总体区间跨0。原8场景标量全输入条件信号为负，完整保留。
## 4. Gap Analysis
完整256通道对全输入gain .02372823且CI跨0，对同容量shift gain .02439002；两个下降场景
保留。八TRAIN场景与自适应补测不是因果/总体/跨seed收益或新颖性证明。
## 5. Design Rationale
按`pipeline/designs/exp_design_Y56_Y57_6debf88fed634a13b772f27bdc244bbd.md`保留769原输入，增加全部256peer对角方差，宽度固定1025，
head hidden256、全部7层、原W2监督/seed1999/30轮/Q128不改。预测新增66048参数是形状计算，
不是资源PASS。先匹配控制、原生后向及配对资源，不用验证集调参。
## 6. Next Batch
只实现独立六文件包及唯一res YAML。CPU/原生/成本测试逐项执行；尚无Git提交、准入或训练。
