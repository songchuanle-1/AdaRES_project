# Bugfix: 队列有开口时正确核验外来 GPU 占用
- **时间**: 2026-10-25，上海时间；本轮快照22:17:04。
- **实验**: AdaRES_project审计；Y62持续训练，Y63排队，Y57暂停。
- **错误类型**: 前台postflight资源状态误判，以及历史可变目标路径导致的证据哈希失效。
- **错误日志摘要**: `Blocked GPU candidate has a queue opening`；随后实际验收发现Y22的旧goal证据路径哈希不匹配。
- **根因**: `exclusive_gpu_slot_available`只反映本项目队列预约；外来compute/图形负载不在队列内。新增server1只更改资源目标后，Y22仍引用可变goal.json的旧内容哈希。
- **修复**:
  1. 保留队列开口判断；增加本轮token/snapshot、服务器身份/白名单、配置哈希、候选ID及原始查询日志绑定的只读GPU占用核验。
  2. 从原始GPU清单、利用率、显存、compute PID及真实canonical预约重算空槽，不能用结果摘要自称“忙”。所有授权卡都不可用才能保持blocked。
  3. 证据须本轮产生且不超过5分钟；查询失败、缺失设备、任一卡空闲、旧token或篡改证据仍拒绝。提供canonical gpu_occupation.py采集入口及文档，旧观察不覆盖。
  4. Y22旧goal的原字节按两项已记录资源替换反向恢复，SHA256精确等于历史53da...c966；存为独立快照，仅更换该条证据路径，拒绝状态/哈希/科研结论均不变。Git当前goal版本不同，未拿它冒充旧证据。
- **重试**: 旧问题先用测试实际复现；19项针对性及147项回归均通过。实际全部研究记录验证已PASS；正式report/finish将随后执行，结果以该token收据为准。
- **证据**:
  - checks/closure_gpu_fix_tests_412be16119c9454f9be9c3e8f5cf69b7.json
  - checks/closure_goal_snapshot_repair_412be16119c9454f9be9c3e8f5cf69b7.json
  - checks/closure_gpu_fix_real_validation_412be16119c9454f9be9c3e8f5cf69b7.json
  - checks/server1_training_progress_412be16119c9454f9be9c3e8f5cf69b7.json
- **边界**: 未改模型、原生YAML、训练参数、准入门槛、资源白名单、现有进程或调度。旧失败证据保留；一次资源核验不证明训练效益。
