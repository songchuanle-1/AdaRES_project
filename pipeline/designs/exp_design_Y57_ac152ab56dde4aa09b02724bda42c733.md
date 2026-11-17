# Experiment Design - AdaRES_project / AdaRES-Y57
创建：2026-10-19，本轮token ac152ab56dde4aa09b02724bda42c733。Status: DRAFT。仅诊断储备；未批准正式模型/训练。

## 1. Motivation
目标原文：AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
上游为同token的六段motivation_review。训练满载、准入预算0，储备预算1；先登记Y57 diagnosis再执行首项对照。角色是验证peer均值的信息容量，而非新增训练或保证mIoU收益。

## 2. Current Best Results
| Experiment ID | Config | State | Best mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|---|
| AdaRES-B1 | 既有基线 | 历史终态 | 0.5020 | 未知 | 未知 | 未知 |
| AdaRES-Y36 | 既有冠军 | 历史done_complete | 0.5110 | 未知 | 未知 | 未知 |
| AdaRES-Y52 | adares_y52_point_target_res | 确认原训练推进 | 18:48监控0.3683 | 未知 | 未知 | 19:05第4/30轮 |
| AdaRES-Y54 | adares_y54_fine_support_res | 确认原训练推进 | 18:48监控0.4462 | 未知 | 未知 | 19:05第5/30轮 |
| AdaRES-Y57 | 无正式配置 | CPU诊断后design | 未知 | 未知 | 未知 | 未训练 |

## 3. Analysis & Direction
### 3.1 现有结果
冠军距目标0.0040，运行实验未终态。新peer散布输入是否对质量有价值完全未定。
### 3.2 不沿用的方向
Y50被拒绝实现及Y53/Y55反证保持；不修改Y52监督目标、Y54连接或Y56复制假设。
### 3.3 尚缺的证据
真实point-IoU/选择错误相关性；反事实的decoder/数据可达性；完整原生前后向/资源/实现独立性；训练后泛化、子组伤害和Pareto比较。文献新颖性未知。
### 3.4 本轮决定
表示容量对照有结果，但恒定mask的IoU标签也恒定，不能解释成任务缺陷。保留同一Y57继续设计，不做正式变体或准入。

## 4. Experiment Plan
| # | ID | Config Path | Purpose / Hypothesis | Expected Result | Priority | Depends On |
|---|---|---|---|---|---|---|
| 1 | AdaRES-Y57 | 无；已冻结CPU程序 | 固定mask/权重/数量/焦点，均值不变而peer散布变化 | 已执行；64/64表示见证、阴性2/2，无任务结论 | 本轮完成 | 固定缓存/严格W2头 |
| 2 | AdaRES-Y57 | 未实现 | 先在源码/CPU核实从同8场景提取point-GT与缓存几何/点数绑定的可行性；不得使用选择后的样本 | 明确是否能得到可核对的实际质量标签，失败保留 | 下一项 | 本轮review及缓存来源限制 |
| 3 | AdaRES-Y57 | 未实现 | 另行冻结新鲜原生TRAIN batch2捕获：索引0,44,60,101,115,126,137,171、全部32表达/Q128，不依结果选焦点；绑定当时权重SHA和真实point-GT，检验peer散布与质量/错误的关系，并设计decoder可达干预 | 可为有关系/无关系/证据不足；均不自动准入 | 待前项完成 | 真空闲且获独占租约的授权GPU |
| 4 | AdaRES-Y57 | 无 | 只有任务相关缺陷和匹配因果测试成立后才考虑同ID具体实现；六门槛另测 | 当前全部未执行，不称PASS | 条件项 | 上述真实证据 |

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 0 | 原Y54，保持 | adares_y54_fine_support_res | 19:05原生步数增长，22718MiB/原PID |
| server-2 | 1 | 原Y52，保持 | adares_y52_point_target_res | 19:05原生步数增长，22716MiB/原PID |
| server-2 | 未分配 | Y57无GPU测试/无训练 | 无 | 本轮CPU一线程；未来捕获仍需实时空闲及独占检查 |
没有新增排队；不停止、移动或重启训练腾卡。Y56继续原七层batch2计划，不与Y57混为同一检验。

## 6. Implementation Checklist
- 已登记唯一新稳定ID，冻结源码/缓存/权重、dtype界、匹配条件和阴性；冻结后执行，未搜索方向。
- 已严格训练头CPU回放、32表达两精度64组和两个同peer阴性；source/cache/checkpoint前后哈希不变。
- 已Python3.8语法/compile；错误token拒绝且台账前后哈希相同。未跑共享流水线全回归，本轮不修改共享代码。
- 未实现模型/config，无跨变体依赖包；Phase3a独立性、原生反向及资源门槛尚不适用/未执行，不作PASS。
- 未提交、推送、修改队列、准入或启动训练；现有授权边界及固定30轮/seed1999/Q128/禁止验证集搜索均不变。

## 7. Risk Assessment
- 代数可区分不等于任务有用；恒mask干预的IoU不变尤其不能当成头预测错误。
- 原缓存权重未在采集时绑定；重复精度不是增加场景样本量。下一项必须核对真实GT/几何，再决定是否值得新鲜原生采集。
- 全模型稀疏前后向、BN行为、延迟/显存、训练收益未知；本轮CPU不产生这些PASS。
- 失败不抹除，不修改门槛或训练目标；按implementation/inconclusive/hypothesis区分范围。旧拒绝路线不自动重开。

## 8. Smell Check
当前无新增可训练结构，不能把加二阶矩/归一化当成已证明的贡献。未来实现若考虑先逐peer非线性再聚合，必须有任务相关缺陷、明确因果对照及独立原生验证，不能从本轮容量构造直接跳入队。
