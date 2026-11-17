# AdaRES-Y49 Motivation Review

## 1. Goal
当前目标文件内容如下；本审查不更改目标、授权或禁令。

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
    "research_on_capacity_gap": true,
    "steady_ticks_before_design": 1,
    "fill_to_capacity": true,
    "design_buffer": 0,
    "max_candidates_per_design_tick": 2
  }
}
```

台账四舍五入冠军Y36为0.5110、目标差0.0040，并非科学晋升。W2仍是科学比较对象；目标未达到，未获Phase2确认。

## 2. Story Status
| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| W2点质量竞争提供可用基线 | 科学比较对象保留 | 既有冻结全量比较 | 不能据此断言全局几何都不可见 |
| Y46全组件质量/均值提供显式信息 | 原型与生产CPU支持，等待原生生产测试 | 同一Y46的修订1和独立包控制 | 原生生产/优化器恢复/成本未通过 |
| Y48相同非空输出掩码的标签闭包 | 正在固定30轮训练 | 已有真实准入及步数证据 | 最终效果未知 |
| Y49在组件均值之外显式读取全局图结构 | 部分诊断/CPU原型支持 | 本轮预注册及三组执行结果 | 原生任务缺口、可学性、成本、独立生产均未证实 |

## 3. Experiment Assessment
| Experiment | mIoU | 与动机的关系 | 对故事线贡献 |
|---|---|---|---|
| W2 | 冻结FIRST 0.5101051604 | 比较对象 | 继续保留，不被新CPU探针取代 |
| B1 | 冻结FIRST 0.5022143852 | 基线 | 后续训练结果必须比较 |
| Y45 | 冻结FIRST 0.5076567776 | 指标分解训练结果不确定 | 不晋升、不调参、不外推否决整个机制 |
| Y8 | 既有否决 | 查询重叠图上的分数扩散 | 保持关闭；Y49不扩散或平均查询置信度 |
| Y31/Y36/Y41 | 仅复用实际源函数作匹配控制 | 形状协方差/局部归一化/边界概率 | 不重开这些候选或重测既有终态 |

以上冻结FIRST值来自既有Y45全量审查，不是本轮新评估，也不作为Y49预训练门槛。

## 4. Gap Analysis
预注册后才审查路线并执行测试。12点整数截角四面体和立方八面体均连通且各自正则，规范化协方差相同。在固定常量特征/掩码/点质量上，实际W2/Y31/Y36/Y41/Y46函数输出最大差2.22e-16，Y46组件行、权重、组精确相同，而固定全局读数不同。

该物理见证的度数为3与4，存在度数混杂；独立8节点三正则立方图/Mobius梯图控制仍有不同全局读数，但不声称该抽象对照具有原生欧氏几何实现。它只证明该显式统计不总能化为度或组件质量，不能证明骨干/注意力普遍盲目。

原生TRAIN8全部32表达式/128查询的CPU原型错配控制32/32发生变化；但相同组件质量分区而全局值不同的原生见证为0。此局限保留，不能宣布真实任务上的父表示缺失已经证实。

## 5. Design Rationale
稳定ID为AdaRES-Y49，假设ID为global-mask-resolvent-coherence；不是Y46缓存优化、Y8分数扩散或已关闭路线的改名。使用原继承最近距离等距边的弱无向图；在每个FG/BG诱导图上求固定I+D-A的逆对角线，再按原点质量平均。两个读数与原生查询共同进入宽度256的小头。

全局求解是新的显式交互路径，但单位尺度不是理论最优或经验证最优；不进行时间、图邻域、温度、阈值或权重扫描。原生任务意义与必要性未知，文献新颖性未证实。四张量CPU梯度和构造见证都不等于准入。

## 6. Next Batch
仅准备同一Y49储备，见`pipeline/designs/exp_design_Y49_c3e5.md`。12:59:55完成CPU钩子32例；下一项是有真实独占空卡后执行已冻结的TRAIN8原生全损失、反传、匹配读数和成本探针，不是启动训练。

13:00:31-36本地观测两张授权卡均有进程：GPU0外部19723、GPU1原Y48 8877；无可分配卡。Y46尚在原定有界复查窗口，本轮不重复其CPU/终态审查。未新增准入或队列任务。
