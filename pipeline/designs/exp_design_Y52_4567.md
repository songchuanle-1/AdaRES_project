# Experiment Design - AdaRES-Y52
整理时间：2026-10-16T16:10:31.428737+08:00；状态：CPU_PREPARED / NOT_ADMITTED。
## 1. Motivation
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
本轮容量2、运行1、准入预算1；遵循先登记候选再测试，非为填卡而准入。
## 2. Current Best Results
| ID | State | Best mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|
| Y36 | 历史完成 | CSV .5110 | 未知 | 未知 | 固定30 |
| Y50 | 原始进程训练中 | 早期约.312 | 约.335 | 约.284 | 当前2/30 |
| Y52 | CPU准备 | 未知 | 未知 | 未知 | 未训练 |
## 3. Analysis & Direction
已确认的是辅助目标差异，不是训练收益。直接置信度质量回归的前提被否定；较窄的共享query监督假设仍需测试。
## 4. Experiment Plan
| ID | Config Path | Purpose | Next Test | Depends On |
|---|---|---|---|---|
| AdaRES-Y52 | configs_song/model_paper5/adares_y52_point_target_res.yaml | 仅修正辅助point-IoU目标 | 原生TRAIN匹配/反向、资源/恢复 | 真实空闲独占授权卡；六门禁及后续提交许可 |
最终及所有既有辅助层均用detached精确点计数；原生>.5筛选不变。单目标RES，显式拒绝不支持的多目标训练。推理、采样、decoder和参数命名保持不变。
## 5. GPU Allocation Plan
| Server | GPU | Assignment | Justification |
|---|---|---|---|
| server-2 | 0 | 外来PID19723，保留 | 非AdaRES占用 |
| server-2 | 1 | Y50 PID31539，保留 | 不移动、不重启 |
| server-2 | 首个真实空闲授权卡 | 仅下一项诊断，尚无预留 | 重新实时检查并取得独占租约 |
## 6. Implementation Checklist
- [x] 独立六文件model包及唯一basename YAML；无跨变体import，原型七文件hash冻结。
- [x] 精确计数32随机控制、空目标/空mask控制、跨2scene/5expression映射；独立模块导入。
- [x] 最终+辅助Criterion组件损失和score梯度与展开点参考精确一致；其余loss不变。
- [ ] 实际稀疏backbone/FPS及原生匹配后的全模型梯度、资源、真实两步/完整恢复。
- [ ] 提交许可、真实commit绑定、完整六门禁receipt、规范dispatch；本轮均未执行。
## 7. Risk Assessment
新增参数0；推理代码未变不等于已测训练资源PASS。CPU v1逐位梯度断言失败保留；v2最大2.273736754e-13，与原样重复误差相同，score梯度精确；非降低科学阈值。最终选择不直接消费score，辅助任务影响可能无效或有害，须训练后判定。
## 8. Smell Check
这是既有监督标签从等超点测度到真实点测度的因果纠正，不是阈值/温度/权重/epoch扫描，也不包装为新架构。文献新颖性未知；原生匹配影响缺证，不作为已准入贡献。
## 9. Prospective Evaluation
若以后六门禁通过，released SPFormer/seed1999/固定30/singleGPU/128queries/singleargmax不变。主指标ScanRefer mIoU，W2/B1冻结比较，全9508表达/141scene各两次且FIRST为主；scene-bootstrap10000/1999；TRAIN尺寸切点1487/3325，子组相对任一参照delta区间上界<0触发伤害审查。报告Acc@.25/.5、查询数、延迟、峰值显存；无验证选择。当前此计划不是准入凭证。
