# Y46 Revision2 Stable-Ordering API Compatibility

2026-10-16 12:51+08:00，尚未重测。修订2 CPU探针在首次钩子运行中失败：
目标PyTorch的argsort不接受stable关键字。原失败日志和bf57bebe版本完整保留。
这不是机制、目标、模型结构或数值准则变化，也不追加第三次模型修订。
原预注册要求仍是按group稳定排序后逐组求和。
仅用唯一整数键group*N+row_index实现同一稳定顺序，替代不被支持的API关键字。
键为int64，不引入阈值或模型数值参数；分组、权重、组件和所有测试不改。
旧新聚合CPU64等价、原精确缓存重复、原生梯度、更新/恢复和资源全部仍待重测。
原日志：pipeline/analyses/y46_revision2_cpu_f56b_f56bf800538f452ebf4b11f81ca8fd8c.log。
