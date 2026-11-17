# Motivation Review - AdaRES_project / AdaRES-Y57
快照：2026-10-19 18:48:24 上海；CPU观测：19:02:40-19:02:46；当前训练双采样：19:05:44-19:05:50。科学状态：表示容量证据，任务相关性仍未知。

## 1. Goal
本轮读取的完整 goal.json：
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
  "pipeline_policy": {"research_on_capacity_gap": true, "steady_ticks_before_design": 1, "fill_to_capacity": true, "design_buffer": 0, "max_candidates_per_design_tick": 2}
}
```
历史规范CSV冠军Y36为0.5110，B1为0.5020，目标0.5150，差0.0040；不同冻结重评估的精确值不与本表混用。本轮准入预算0、储备准备预算1；只新增一个稳定诊断ID，不新增队列或训练。

## 2. Story Status
| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| Y52点级质量监督 | 原fixed30训练推进中 | 本轮19:05双采样，第4轮912到921/4686 | 完成训练及冻结评估，不以早期指标定性 |
| Y54空间细支持 | 原fixed30训练推进中 | 本轮19:05双采样，第5轮2223到2232/4686 | 同上 |
| Y56精确复制数量造成上下文扰动 | 原候选保留，未执行新GPU检验 | 上轮32缓存表达未发现精确query/softmask复制；七层batch2程序已冻结 | 两卡均被原训练占用；不重新跑旧CPU对照 |
| Y57竞争query特征分布被均值聚合混叠 | 本轮表示容量对照成立 | 严格训练W2头、32表达、float32/64共64组 | 并未建立任务相关性、原生可达性或性能收益 |

## 3. Experiment Assessment
| 实验 | mIoU | 与动机的关系 | 本轮可作出的判断 |
|---|---|---|---|
| B1 | 历史0.5020 | 标准置信度基线 | 本轮未重评估 |
| Y36 | 历史0.5110 | 当前规范冠军 | 尚未达到0.5150 |
| Y52 | 18:48监控最佳0.3683 | 点级质量目标 | 训练早期，不代表终态效果 |
| Y54 | 18:48监控最佳0.4462 | 细支持信息 | 同上 |
| Y57 | 未知，未训练 | 竞争特征信息容量 | 不是精度或科学准入PASS |

## 4. Gap Analysis
- W2原生头接收焦点query、重叠加权peer均值、两者差及最大重叠；没有独立的peer特征散布输入。源码按原字节及LF规范化分别绑定；不是凭模块名推断。
- 本轮在固定8个TRAIN场景、32表达、每表达128query中固定焦点index0，不依预测选择样本。对非焦点query施加坐标0决定、且与常数向量及加权标准化均值正交的Householder反射。mask、点数、竞争权重、数量和焦点query固定；每query均值/方差在浮点界内保持。
- float32/64各32组均满足匹配条件，变换前后每组均128个唯一query；加权逐通道二阶矩差最小分别0.00061038509/0.00061042838，最大约0.20108378。它们不是mIoU变化。焦点logit最大误差分别9.536743e-7/1.776357e-15，重复误差0、排列误差最大1.907349e-6，均在预注册dtype算术界内。两组同peer特征阴性均无散布见证。
- 反射是合法的头输入代数干预，不是已证明可由真实decoder/数据产生的反事实。更重要的是，mask和GT不变时真实IoU也不变，所以置信度不变本身并非预测错误，不能由此声称发现了任务缺陷。
- 该固定旧缓存的采集源码指定W2权重路径，却未当时记录checkpoint blob SHA；本次严格回放绑定e340728权重，不能追认旧采集权重字节。8场景/32表达在两精度下重复不是64个独立样本。

## 5. Design Rationale
- Y57是固定mask/数量下peer特征取值分布的容量诊断；Y56改变精确复制数量，Y50研究多mask空间联合覆盖，Y39/Y42研究超点内部点特征分布。未改名重开这些路线，也没有恢复任何拒绝ID。
- 先检验信息是否可区分，再检验它是否与真实质量/选择错误相关。当前只完成前者，不实现、提交、准入或训练新模型。
- 下一项任务证据必须从固定TRAIN样本原生捕获中绑定源码、权重、实际point-GT及预测；先检验质量关联和decoder可达的干预。预注册失败/无关联时保留负结果，不靠换焦点、调方向、扫描阈值或验证集择优获得阳性。

## 6. Next Batch
见同token的 `pipeline/designs/exp_design_Y57_ac152ab56dde4aa09b02724bda42c733.md`。本轮完成一个有执行证据的有界储备准备；不宣称所有路线耗尽、不设科学无信息增益冷却、不将Y57伪装成GPU阻塞。Y56原始资源条件保留。
