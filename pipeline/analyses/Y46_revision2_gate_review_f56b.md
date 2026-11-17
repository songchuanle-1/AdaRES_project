# Y46 Revision2 Executability And Resource Review

2026-10-16，前台token f56bf800538f452ebf4b11f81ca8fd8c。
审查对象是当前两次修订后的具体实现，不是整个全局组件置信度假设。
实际decoder SHA363506506bf0e289d67dfe72c1a0041e66a67fd1962247a726c34d47b838c225。

| Gate | Result | Actual Evidence | Limits |
|---|---|---|---|
| diagnosis | 保留已有信息区分支持 | 原生TRAIN8/32捕获与真实源控制 | 不证明训练收益、必要性或整体网络盲性 |
| controls | 当前修订通过所测执行控制 | CPU32实际生产钩子；32原生缓存精确重复全部通过；错配分离 | 不是完整loader位级复现 |
| native_backward | 通过所测生产路径 | released-init真实tools.train两步、8张量非零梯度和更新、模型/AdamW/PolyLR状态精确恢复 | 临时诊断，不是完整训练 |
| resources | 当前实现不予准入 | 固定TRAIN8/最大对串行推理和原生压力的实际成对成本 | 显存可运行不等于轻量Pareto通过 |
| independence | 独立包实际检查通过 | 父AST剥离、五个父文件/配置字节、DEC初始状态和RNG保持 | 未提交、未签发准入收据 |
| constraints | 所测协议保持 | released SPFormer/seed1999/固定30/128查询/单一选择/server2GPU1 | 无调参、无正式训练、无Phase2批准 |

## Resource Decision
固定场景对728/6780，2次warmup/3次读数，候选native训练耗时
17.642870/19.155023/15.316430秒，禁用父路径0.958390/0.958899/0.958847秒。
候选峰值reserved17.867188GiB，虽小于24GiB，但训练步中位耗时约为父路径18.4倍。
串行预测scene0497_00三次16.598758/16.599820/16.581493秒，父路径约0.22秒。
其他TRAIN8场景也明显变慢；首次父路径冷启动10.84秒不作加速参照。
新增328961参数约为W2的0.20%，不能用参数小/显存足以忽略真实时间开销。

当前实现与项目轻量化Pareto方向不相容，不签发资源准入；拒绝此具体实现，
failure_scope=implementation，而不是声称核心机制被普遍证伪。
不拟合新的数值阈值，不以小样本mIoU/显著性/预期收益作为训练前否决标准。
拒绝依据是该实现已实际暴露的严重计算成本，而非未运行的训练效果。
两次局部修订预算已使用，本轮不追加第三次修订、不改名绕过历史状态。

## Preserved Failures
原子index_add聚合的缓存重复差异2.98e-8至5.96e-8，原精确规则判失败；
取证器继承的cached_controls.repeat_exact=true常量不是实际通过，实际forensics全部false。
修订2最初stable关键字API不兼容，失败日志/源码保留；唯一整数键只实现相同稳定排序，
不算第三次模型修订，也不放宽重复门槛。旧新CPU64最大差1.665e-16，组件输入不变。
原始运行成本、revision1缓存成本、全部失败证据和源归档都保留。

## Evidence
- pipeline/analyses/y46_revision2_compat_cpu_f56b_f56bf800538f452ebf4b11f81ca8fd8c.json
- pipeline/analyses/y46_revision2_compat_native_f56b_f56bf800538f452ebf4b11f81ca8fd8c.json
- pipeline/analyses/y46_revision2_compat_controls_f56b_f56bf800538f452ebf4b11f81ca8fd8c.json
- pipeline/analyses/y46_revision2_stress_f56b_f56bf800538f452ebf4b11f81ca8fd8c.json
- pipeline/analyses/y46_revision2_prereg_f56b_f56bf800538f452ebf4b11f81ca8fd8c.json
- pipeline/analyses/y46_revision2_api_compat_prereg_f56b.md
