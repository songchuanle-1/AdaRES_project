# Experiment Design - AdaRES_project / Y56, Y57
创建：2026-10-20T21:24:56.414176+08:00。Status: DRAFT。上游：`pipeline/reviews/motivation_review_Y56_Y57_6debf88fed634a13b772f27bdc244bbd.md`。保持原稳定ID，不新增占位候选。
## 1. Motivation
项目动机原文：AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
队列0/容量2，准入预算2；续做两个已有候选，不以填满GPU替代门槛。
## 2. Current Best Results
| 实验 | mIoU | Acc25 | Acc50 |
|---|---|---|---|
| Y52 | 0.50970843 | 0.610013 | 0.562053 |
| Y54 | 0.50634295 | 0.603176 | 0.556584 |
| W2 | 0.51010800 | 0.610749 | 0.563000 |
| B1 | 0.50222176 | 0.610433 | 0.552903 |
以上四方法9508表达/141场景各双次；FIRST为主。Y36新配对评估因真实剩余时间不足未启动，
完整五方法比较尚未闭环。重复、当前子组和成本见`pipeline/analyses/design_and_terminal_review_6debf88fed634a13b772f27bdc244bbd.json`。
## 3. Analysis & Direction
### 3.1 当前结果
Y52/Y54原fixed30结束；不因epoch30日志的分段best较低覆盖原best，也不新选epoch。
### 3.2 负证据
Y57标量在全部输入条件下无稳定增益，完整向量有两个下降场景，均保留。
Y56最终层未见复制，不能把初始层局部现象泛化。
### 3.3 尚缺证据
两项正式独立实现、控制分离、原生后向、新张量梯度、配对资源、训练收益均未完成。
### 3.4 决定
Y56优先做原生任务损失可达性，不直接实现生产去重；
Y57下一项是全通道二阶矩诊断原型及匹配对照，不从观察关联生成准入。
## 4. Experiment Plan
| ID | 配置 | 下一项 | 预期/失败判据 | 依赖 |
|---|---|---|---|---|
| Y56 | 尚未创建 | 当前真实复制case与无复制输入，冻结逆复制上下文干预并逐层捕获原生损失/梯度 | 可为无活动/反向/有活动，均保留；不是固定收益门槛 | 本轮case证据 |
| Y57 | 尚未创建 | 独立包拟名adares_y57_peer_second_moment，保留769维并拼接256维逐通道方差；全部7层共享原挂接 | 对齐/错位同容量、排列/同peer、原生loss与每个新张量有限非零梯度 | 本轮完整向量证据和全部负结果 |
固定W2监督/30轮/seed1999/Q128/单argmax/SPFormer初始化，不使用诊断训练W2权重正式初始化。
Y57仅改peer上下文信息流；不重开Y42超点内部特征分布或Y50联合mask覆盖。
设计预测参数增量66048，仅是形状计算，不是资源PASS；不裁通道或选层。
## 5. GPU Allocation Plan
仅server-2 GPU0/1，每卡单任务。下项每次重查空闲/进程并获得独占诊断租约；
本轮冻结评估在GPU1串行执行，不从旧空闲快照宣称当前卡空闲。不用server-6。
## 6. Implementation Checklist
- 已完成：source/checkpoint/GT绑定、原生batch2观察、同容量/错位和实际复制匹配诊断、6项分析回归。
- 待完成：正式物理独立6文件包+唯一YAML、代码级控制、原生全后向/新参数梯度、配对时延显存。
- 待门槛全部通过后，按实际已有授权核对精确测试文件的提交范围；当前无测试完生产包，不制造授权阻塞。
- 未来训练前继续冻结全9508/141双FIRST，与W2/B1/Y36各双次；scene-bootstrap10000/1999，
  TRAIN cuts1487/3325，任何有意义子组对任一参考CI上界<0触发伤害审查。
  报告mIoU/Acc25/Acc50/Q/时延/显存/参数；噪声来自独立重复，不造常数。
## 7. Risk Assessment
八场景与自适应补测不能证明总体收益；局部复制反事实不是预测伤害。
未建立新颖性、跨seed稳定性或全局Pareto。正式实现前所有受影响门槛仍须实测。
## 8. Smell Check
Y57增加均值聚合不可恢复的peer二阶矩信息通路，而非调整温度、loss系数或验证阈值；
全部256通道来自原d_model，未择优。Y56是复制计数的不变性诊断，并非改名重开历史拒绝方向。
