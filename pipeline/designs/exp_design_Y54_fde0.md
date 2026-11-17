# Experiment Design - AdaRES Y54
时间：2026-10-17T13:14:22.508061+08:00；状态：实现准备，不是准入。
## 1. Motivation
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
角色：检验去除批内虚构3D支持是否能使共享query学习及置信度证据更一致；不承诺提升。
## 2. Current Best Results
| ID | 状态 | mIoU | Acc50 | spmIoU |
|---|---|---|---|---|
| Y36 | 历史CSV完成 | .5110 | 未知 | 未知 |
| W2 | 上轮全量冻结FIRST | .5101051604 | .5629995793 | .4737185291 |
| B1 | 上轮全量冻结FIRST | .5021216787 | .5527976441 | .4666861819 |
| Y54 | 未训练 | 未知 | 未知 | 未知 |
## 3. Analysis & Direction
实际原四组TRAIN配对有3600个跨层虚构fine支持槽、34172个可见query-key边，非独立样本数。虚构支持注意力质量为正；原样repeat0，padding干预作用12.5120，正确mask后0；全有效表达置信度负控制0。不能外推训练收益或普遍数值界。
## 4. Experiment Plan
| ID | Config | 假设 | 下一门禁 |
|---|---|---|---|
| AdaRES-Y54 | configs_song/model_paper5/adares_y54_fine_support_res.yaml | native fine支持与邻域继承有效mask | 独立原型、实际TRAIN反向/恢复和资源 |
## 5. GPU Allocation Plan
仅server-2 GPU0/1；GPU0外来工作保留。诊断需要实时空闲和独占租约，不预留卡。提交/准入前不派发训练。
## 6. Implementation Checklist
- [x] 原生实际支持审计及matched干预/正负/重复控制。
- [ ] 独立六文件包+唯一basename YAML；仅DEC.forward_iter_pred三项mask连接，逆向AST相等审计。
- [ ] 实际原生反向、非空优化器/调度/RNG完整恢复、资源和同权重推理对照。
- [ ] 限定人工提交许可、真实commit与六门禁回执、canonical派发及递增训练步数。
## 7. Risk Assessment
新增参数预期0仍须实测；padding纠正改变训练上下文，但标准单场景推理无padding时应保持行为。少量输入不证明最坏情况或Pareto；保留潜在空邻域/小支持、数值及原生loss约束。原型不改损失/匹配、不部署诊断W2权重。
## 8. Smell Check
缺陷证据与因果干预来自native不存在支持；不是扫描mask阈值/温度/权重/轮数，也不把修复包装为新架构或已证新颖性。只恢复原生有效性信息流。
## 9. Prospective Evaluation
若以后正式准入：released SPFormer/seed1999/fixed30/单GPU/128query/singleargmax和原loss不变。冻结ScanRefer val全9508表达/141scene，Y54/W2/B1/当前冠军Y36各两次且FIRST主结果；scene-bootstrap10000/1999，TRAIN cuts1487/3325，任一参照子组delta区间上界<0触发伤害审查。报告mIoU/Acc25/Acc50/active queries/warm latency/peak memory/parameters。Y36 exact best检查点须在准入前读取唯一既有训练归档并绑定哈希，不按新验证结果择优。无验证调参，单seed区间不覆盖训练seed不确定性。
