# Experiment Design - AdaRES-Y50

## 1. Motivation
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
当前缺口是高阶共同覆盖能否改善置信度学习，而不是再寻找重叠阈值或温度。

## 2. Current Best Results
| Experiment ID | Config | State | Best mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|---|
| Y36 | 历史条目，本轮未重查 | done_complete | CSV .5110 | 未知 | 未知 | 未重查 |
| W2 | adares_w2_pointmass_competition_res | done_complete | 当前冻结FIRST .5099093531 | 未重查 | 未重查 | fixed30 |
| Y50 | adares_y50_joint_incidence_res | blocked | 未训练/未知 | 未知 | 未知 | 未启动 |

## 3. Analysis And Direction
不是缺少pairwise，而是在等全Gram下仍存在高阶共同覆盖差；实际父指示器匹配控制已通过，不能外推整个上游网络盲区。

## 4. Experiment Plan
独立六文件包，固定FG/BG条件二阶中心竞争覆盖，原维度、loss、masks和selector不变；最终一次指示器前景logit增加可学习信息路径。
新增66561参数/4张量，配置逐字节native W2。生产CPU、原生TRAIN8、released两步、全状态恢复、成对资源均实测通过。
训练后继承已冻结比较：released SPFormer，seed1999，fixed30，单GPU，Y50/W2/B1各9508/141全量重复两次，FIRST为主，scene-bootstrap10000/seed1999。
沿用原TRAIN size cuts1487/3325；预定子组delta CI upper<0相较任一W2/B1触发伤害审查，跨0不确定。报告mIoU、Acc@.25/.5、128活跃查询、单argmax、延迟/峰值显存及重复噪声。
禁止validation阈值/温度/权重/特征子集/epoch择优，未知官方unique/multiple不冒称已分层。

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 0 | 外来PID19723，保留 | 非本项目 | 只读核验，不处置 |
| server-2 | 1 | 诊断已结束，训练未启动 | Y50仅待授权 | 白名单不等于已准入 |

## 6. Implementation Checklist
独立代码/配置、父AST/字节/初始化/RNG、native梯度/错配、两步训练/恢复、推理/压力均完成。
未自动提交，未生成admission receipt，未修改queue/marker，未启动正式训练；待明确授权后限定提交已测候选字节，再走canonical dispatch并验native步数增长。

## 7. Risk Assessment
高阶统计可能冗余、场景关联可能反转；保留全部负向个例。原生活动与小成本不是训练收益或Pareto优势。
诊断权重与临时checkpoint不作为正式训练起点。归因仅限被实际控制的指示器边界。

## 8. Smell Check
新信息来自多候选点位联合覆盖中的三重交互，不是改变原pairwise数值或扫描系数。结构由条件二阶矩定义，无数值搜索。
Status: IMPLEMENTED / NOT ADMITTED; submitted-commit authorization pending.
