# Motivation Review - AdaRES-Y51
整理时间：2026-10-16T14:37:17.679315+08:00。这是已完成诊断的整理，不伪造事前审查时间；原始判据见本轮预注册和数值程序冻结。

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
目标.515未达标；台账Y36 .5110、B1 .5020为原有记录，不是本轮复测。

## 2. Story Status
| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| Y50高阶共同覆盖 | 既有提交授权阻塞，未到期 | 上轮独立包/原生/恢复/资源证据保留 | 不重复测试或重设重试时间 |
| Y51公共logit偏移造成监督与部署错配 | 本诊断前提已否定 | native RES已softmax前景概率；65组完整选择掩码不变 | 不由此创建数值校准变体 |

## 3. Experiment Assessment
| Experiment | mIoU | 与动机的关系 | 对故事线贡献 |
|---|---|---|---|
| 原有Y36/W2/B1 | 本轮未评测 | 保留既有科学参照 | 没有新指标提升证据 |
| Y51 | 未训练/不创建模型 | 审计训练置信度与部署取值的一致性 | 排除本实际RES路径的公共模态错配前提 |

## 4. Gap Analysis
实际父模型不是原始前景logit排序，而是两类softmax前景概率singleargmax。原训练CE与该选择路径没有被本判据发现的公共平移错配。
该结论不证明全部数值鲁棒性、全模型泛化或任务收益。32条原生TRAIN输入的float64重建不是完整GPU推理等价声明。

## 5. Design Rationale
固定32随机fixture、精确半整数阳性对照及8场景32条既有TRAIN输入；原始前景logit规则有29组排名变化，而真实native RES始终选择相同掩码，说明控制并非无效扰动。
float64仅显式适配原生固定float32 one-hot缓冲区的dtype叶子，标签/公式/权重不变；字面原生float32损失及梯度另在精确阳性fixture上逐位相等。未放宽判据。

## 6. Next Batch
本诊断不进入模型设计/准入，不扫描排名、温度、权重或阈值。Y50继续原有授权阻塞；没有新增人工请求或训练启动。
