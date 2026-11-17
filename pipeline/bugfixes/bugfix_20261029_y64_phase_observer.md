# Bugfix: Y64验证阶段被训练步数观察器误报

- **时间**: 2026-10-29 04:46 上海
- **实验**: AdaRES-Y64
- **错误类型**: 只读监测误报；原训练未失败
- **错误日志摘要**: 旧观察器远端第20行 AssertionError，仅比较Train计数器五秒内是否增长。原失败日志和只读诊断完整保留。
- **根因**: 正常Val[24/30]期间Train计数保持不变；真实诊断记录Val计数仍在增长，原PID/cwd/GPU进程未变。
- **修复**: 新增observe_y64_server1_phase.py，按原生日志出现顺序解析Train/Val，以(epoch, Train0/Val1, step)及日志字节严格增长确认进度。最多三次五秒观察；静止只记unconfirmed，不授权重启。倒退/截断/错误总轮数或分母仍拒绝，六源码/YAML/marker/准入/GPU0独占核验保留。旧hash-bound观察器与失败证据不修改。
- **重试**: 第1次新观察器只读复核已恢复：上海04:45:08至04:45:13，Train25/30从23至31/4686；同PID2752219、marker9706、六源码和独占通过。
- **验证**: 8项CPU保护测试通过，覆盖验证期、阶段切换、下一轮、静止、错分母/轮数、倒退与日志截断。另直接解析本次真实Val诊断尾部得到epoch24。
- **证据**: pipeline/checks/y64_server1_progress_d35e78a8839e4730b6b5623198771c8e.log；y64_phase_diagnostic同token.json/.log；y64_server1_phase_progress同token.json/.log；y64_phase_guard_tests同token.log。
- **边界**: 无远端训练修改、重启、checkpoint改写、准入重放、部署或硬件/调度变化。训练效果仍待固定30和原冻结全量后评。
