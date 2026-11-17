# AdaRES-Y49 Experiment Design

Status: DRAFT, CPU prototype only; not admitted, queued or launched.

## 1. Motivation
项目动机原文："AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."

上游审查：`pipeline/reviews/motivation_review_Y49_c3e5.md`。本轮12:48:41+08设计快照活跃1/目标2，准入预算1、储备准备预算1；Y46因外部GPU占用未到复查时间。先注册Y49，再执行可证伪测试，不因逻辑队列空位推断物理GPU空闲。

## 2. Current Best Results
| Experiment ID | Config | State | 冻结FIRST mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | 保留科学比较对象 | 0.5101051604 | 本文未复核 | 本文未复核 | 固定30 |
| B1 | 既有B1 | 保留基线 | 0.5022143852 | 本文未复核 | 本文未复核 | 既有协议 |
| Y45 | adares_y45_metric_decomposition_res | complete_inconclusive | 0.5076567776 | 本文未复核 | 本文未复核 | 30 |
| Y48 | adares_y48_mask_equivalence_res | 原训练进行中 | 未知 | 未知 | 未知 | 固定30 |

既有FIRST数值复用Y45冻结审查，不是本轮新评估或晋升。Y36台账0.5110不替代科学比较对象；目标0.515不变。

## 3. Analysis & Direction
32组真实源函数控制支持构造输入上的显式信息区别；额外同度数抽象控制表明固定全局读数不总是度数函数。两者都不证明真实任务上的必要性或新颖性。

TRAIN8捕获32表达式/128查询的CPU全局描述符，错配控制32/32分离；同组件质量分区/不同全局值的原生见证为0。原型四张量CPU代理CE最小梯度0.0101877356；钩子32例最小梯度0.0003606402。所有数值是执行/信息路径证据，不是训练效果。

不重开Y8/Y31/Y36/Y41，不改Y46实现或当前Y48。原生可达性、资源和独立生产测试仍是待办，不将它们写成已通过。

## 4. Experiment Plan
| ID | Config Path | Purpose / Hypothesis | Expected Result | Priority | Depends On |
|---|---|---|---|---|---|
| AdaRES-Y49 | 尚未创建独立配置 | 测试组件质量之外的全局图读数是否能成为可学的置信度输入 | 原生路径是否可达、实际代价；效果未知 | 储备 | 原生/资源/独立性/完整准入 |

冻结原型：每个硬FG/BG诱导图使用原等距最近边的弱无向支持，L=D-A；按点质量加权平均diag((I+L)^-1)，空侧0。不得删除组件或改变mask>0。SciPy稀疏LU和单位矩阵分块求精确对角线，块大小64是执行组织，不是模型选择；固定前四表达式已对照块1及联合反序。

查询做既有宽度的LayerNorm，与两个读数拼接，Linear(258,256)/GELU/Linear(256,1)，仅加到最终前景indicator。四张量66561参数；六辅助输出和禁用父路径在32例CPU钩子上精确不变。下一步原生探针程序已绑定源哈希，CPU和GPU分支使用AST完全相同的描述符/小头。

若将来全部门槛通过：released SPFormer、seed1999、固定30轮、原128查询/单一选择、原目标与优化器调度不变。终态评估Y49/W2/B1全9508表达式/141场景各两遍，FIRST为主；场景聚类bootstrap10000/seed1999；原冻结TRAIN点数切分1487/3325；子组区间上界<0触发伤害审查。跨决策边界为不确定，不声称等效或无伤害。保留重复噪声与训练后成对时间/显存/Pareto审查；Acc@0.25、Acc@0.5、活跃查询必须报告，缺失标未知。

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 0 | 外部PID19723，保护 | 非本项目，不读取其内容 | 13:00:36本地观测94%、23586MiB |
| server-2 | 1 | 原Y48 PID8877，保护 | adares_y48_mask_equivalence_res | 第2/30轮4070到4079/4686，78%、22714MiB |
| server-2 | 首张真实空闲授权卡 | 将来的独占原生诊断，不是新训练 | Y49诊断程序 | 须现场空闲、无计算PID及独占租约；旧快照/计时器不授权 |

不使用server6，不改其他项目或调度；本设计不授予额外并发。

## 6. Implementation Checklist
- [x] 先注册稳定诊断ID与哈希预注册。
- [x] 实际源控制32例、同度数抽象控制及物理几何局限记录。
- [x] 固定TRAIN8/all32/all128 CPU描述符、错配、代理梯度及固定数值控制。
- [x] 同源AST钩子32例：最终前景、六辅助、禁用/重复、错配控制。
- [ ] 独占空卡上的原生全损失/反传、缓存控制及成本测量。
- [ ] 物理独立生产包/配置、released-init真实tools.train、完整状态恢复和资源压力。
- [ ] 六项门槛实际审查、精确源提交和哈希准入。
- [ ] 仅准入后走既有受控调度路径，验证原生步数增长。

## 7. Risk Assessment
原生捕获CPU描述符总测试22.62秒，单表达式最慢5.08秒，未经成对原生测量；66561参数不能证明轻量化。求逆对角计算量、图细分敏感性、硬图脱离梯度、空掩码含义和单位尺度均是风险。原生同组件分区见证为0，不抹掉该结果或制造诊断PASS。

CPU钩子是声明过的替身，不等于原生模型。独立生产/优化器/资源未执行，不允许准入；模型权重仅诊断临时使用。失败按真实范围保留，局部实现修订必须同ID预注册，不选验证表现、不改门槛、不盲重启。

气味审查：新增参数虽小，仍有点图全局求解的显式信息交互，并非仅缩放/LayerNorm或查询分数扩散；固定单位算子没有经过验证扫描。但原生任务价值、必要性和文献新颖性全部未知，不将结构不同当作成功。
