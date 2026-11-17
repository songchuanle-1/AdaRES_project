# Y51 Gauge Audit Disposition

审查时间：2026-10-16T14:37:17.679315+08:00；stage=rejected；failure_scope=hypothesis，仅指本公共模态错配前提。
精确native IPDN.predict_by_feat的RES分支已经F.softmax(pred_indis,dim=-1)[:,1]后singleargmax。
32fixture+1半整数阳性对照+32条原TRAIN输入，65/65完整选择掩码不变；原始前景logit排序29组变化。
最大原生选择器分值差2.2204460492503131e-16；float64对照CE差1.7763568394002505e-15，梯度差5.5511151231257827e-17，均在事前8eps*nodes*Q包络内。
字面原生float32损失/梯度阳性控制精确相等。float64损失one-hot缓冲dtype适配显式记录，不能称未改动的完整原生训练。
第一次探针因get_iou误从ipdn.utils导入而在任何数值测试前失败；原始脚本/日志保留，v2只提取实际loss.py的原生get_iou AST，判据/模型不变。
不重开已拒绝ranking/readout家族，不创建缺乏此诊断前提的校准模型，不把CPU审计当mIoU收益。无GPU使用、模型修改、正式准入或训练。
Y50授权阻塞未到期，不重复它的测试、不增加证据或改变retry；其原始提交授权条件仍为唯一已登记未完成事项。
