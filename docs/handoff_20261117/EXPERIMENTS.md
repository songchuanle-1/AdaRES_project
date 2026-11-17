# AdaRES 实验统计

统计时间（UTC）：2026-11-17T11:07:08.427881+00:00。原始源提交：`0c6c844719c8466ddd8888d054fe3b7ee658e643`。

## 统计口径

按稳定实验 ID 去重，合并混合格式 CSV、逐实验注册表和候选台账。
候选不等于实际训练；延长训练的 B2/B4-E30 等保留原 ID，不能据此计算独立从头训练次数。
注册表终态不自动代表 checkpoint 完整性；训练 best 不等于冻结 FIRST，也不等于已证实改进。

- 有训练记录的唯一实验 ID：**85**。
- 训练状态：`{"closed_low_metric": 1, "done_complete": 81, "done_earlystop": 1, "paused": 1, "running": 1}`。
- 候选记录：**155**，阶段：`{"blocked": 41, "closed": 12, "complete": 19, "rejected": 82, "training": 1}`。
- 合并后涉及的唯一 ID：**159**。
- 原始 CSV：87 行，83 个 ID；注册表 85 份。
- 新式 `complete_scientific_review.json`：6 份；旧式完整后评需查原始分析，不能把此数当作全部后评数。
- 准入收据文件：79，可能包含修订，不是训练次数。

## 有训练记录的实验

| 实验 | 训练状态 | 训练 best mIoU | 新式正式 FIRST | 正式结论/当前候选阶段 |
|---|---|---:|---:|---|
| AdaRES-B0 | closed_low_metric | 0.333000000 |  |  |
| AdaRES-B1 | done_complete | 0.502000000 |  |  |
| AdaRES-B2 | done_complete | 0.505000000 |  |  |
| AdaRES-B2-E30 | done_complete | 0.505800000 |  | closed |
| AdaRES-B3 | done_complete | 0.501400000 |  |  |
| AdaRES-B4 | done_complete | 0.507000000 |  | closed |
| AdaRES-B4-E30 | done_complete | 0.509400000 |  | closed |
| AdaRES-B5 | done_complete | 0.503000000 |  | closed |
| AdaRES-C1 | done_complete | 0.495000000 |  | closed |
| AdaRES-C1-E30 | done_complete | 0.500800000 |  | closed |
| AdaRES-C2 | done_complete | 0.498000000 |  | closed |
| AdaRES-C2-E30 | done_complete | 0.501400000 |  | closed |
| AdaRES-D1 | done_complete | 0.508000000 |  | closed |
| AdaRES-F4 | done_complete | 0.506000000 |  | closed |
| AdaRES-G1 | done_complete | 0.498000000 |  | closed |
| AdaRES-H5 | done_earlystop | 0.506100000 |  | closed |
| AdaRES-W2 | done_complete | 0.510000000 |  | complete |
| AdaRES-Y15 | done_complete | 0.230000000 |  | rejected |
| AdaRES-Y17 | done_complete | 0.504000000 |  | complete |
| AdaRES-Y18 | done_complete | 0.508000000 |  | complete |
| AdaRES-Y23 | done_complete | 0.507600000 |  | complete |
| AdaRES-Y25 | done_complete | 0.494000000 |  | rejected |
| AdaRES-Y27 | done_complete | 0.510000000 |  | complete |
| AdaRES-Y28 | done_complete | 0.506000000 |  | rejected |
| AdaRES-Y30 | done_complete | 0.504000000 |  | rejected |
| AdaRES-Y31 | done_complete | 0.506000000 |  | complete |
| AdaRES-Y33 | done_complete | 0.505000000 |  | rejected |
| AdaRES-Y35 | done_complete | 0.505000000 |  | complete |
| AdaRES-Y36 | done_complete | 0.511000000 |  | complete |
| AdaRES-Y37 | done_complete | 0.505000000 |  | complete |
| AdaRES-Y39 | done_complete | 0.508000000 |  | complete |
| AdaRES-Y41 | done_complete | 0.509000000 |  | complete |
| AdaRES-Y42 | done_complete | 0.505000000 |  | complete |
| AdaRES-Y43 | done_complete | 0.508500000 |  | rejected |
| AdaRES-Y44 | done_complete | 0.510253727 |  | complete |
| AdaRES-Y45 | done_complete | 0.507658958 |  | complete |
| AdaRES-Y48 | done_complete | 0.503401041 |  | rejected |
| AdaRES-Y50 | done_complete | 0.503000000 |  | rejected |
| AdaRES-Y52 | done_complete | 0.509000000 |  | complete |
| AdaRES-Y54 | done_complete | 0.501500000 |  | complete |
| AdaRES-Y56 | done_complete | 0.510989130 |  | blocked |
| AdaRES-Y57 | paused |  |  | blocked |
| AdaRES-Y58 | done_complete | 0.503437638 |  | blocked |
| AdaRES-Y59 | done_complete | 0.495655596 |  | blocked |
| AdaRES-Y60 | done_complete | 0.507524908 |  | blocked |
| AdaRES-Y61 | done_complete | 0.503639519 |  | blocked |
| AdaRES-Y62 | done_complete | 0.508224785 |  | blocked |
| AdaRES-Y63 | done_complete | 0.507433712 |  | blocked |
| AdaRES-Y64 | done_complete | 0.502873182 |  | blocked |
| AdaRES-Y65 | done_complete | 0.509360254 |  | blocked |
| AdaRES-Y66 | done_complete | 0.508760810 |  | blocked |
| AdaRES-Y67 | done_complete | 0.506428182 |  | blocked |
| AdaRES-Y68 | done_complete | 0.503954947 |  | blocked |
| AdaRES-Y69 | done_complete | 0.501476943 |  | blocked |
| AdaRES-Y70 | done_complete | 0.502948999 |  | blocked |
| AdaRES-Y71 | done_complete | 0.498878628 |  | blocked |
| AdaRES-Y72 | done_complete | 0.500363588 |  | blocked |
| AdaRES-Y73 | done_complete | 0.497408599 |  | blocked |
| AdaRES-Y74 | done_complete | 0.505369604 |  | blocked |
| AdaRES-Y75 | done_complete | 0.504463553 |  | blocked |
| AdaRES-Y76 | done_complete | 0.505005240 |  | blocked |
| AdaRES-Y77 | done_complete | 0.510812283 |  | blocked |
| AdaRES-Y78 | done_complete | 0.503971934 |  | blocked |
| AdaRES-Y79 | done_complete | 0.500257671 |  | blocked |
| AdaRES-Y80 | done_complete | 0.509652555 |  | blocked |
| AdaRES-Y81 | done_complete | 0.507803380 |  | blocked |
| AdaRES-Y82 | done_complete | 0.507649481 |  | blocked |
| AdaRES-Y83 | done_complete | 0.500829399 |  | blocked |
| AdaRES-Y84 | done_complete | 0.502024174 |  | blocked |
| AdaRES-Y85 | done_complete | 0.505927265 |  | blocked |
| AdaRES-Y87 | done_complete | 0.504583001 |  | blocked |
| AdaRES-Y88 | done_complete | 0.503773630 |  | blocked |
| AdaRES-Y89 | done_complete | 0.503692687 |  | blocked |
| AdaRES-Y90 | done_complete | 0.503818095 | 0.503813720 | rejected_material_subgroup_harm |
| AdaRES-Y91 | done_complete | 0.511372507 |  | blocked |
| AdaRES-Y92 | done_complete | 0.509251475 | 0.509342623 | rejected_material_subgroup_harm |
| AdaRES-Y93 | done_complete | 0.505044580 |  | blocked |
| AdaRES-Y94 | done_complete | 0.507565260 | 0.507561106 | rejected_material_subgroup_harm |
| AdaRES-Y95 | done_complete | 0.508540154 |  | blocked |
| AdaRES-Y96 | done_complete | 0.508160174 | 0.508260988 | complete_inconclusive |
| AdaRES-Y98 | done_complete | 0.499404341 |  | blocked |
| AdaRES-Y99 | done_complete | 0.510926247 | 0.510926248 | complete_inconclusive |
| AdaRES-Y100 | done_complete | 0.516448557 |  | blocked |
| AdaRES-Y101 | done_complete | 0.506299555 | 0.506298970 | complete_inconclusive |
| AdaRES-Y102 | running |  |  | training |

## 仅候选记录

下列记录没有在注册表或原始 CSV 中找到训练记录，不能记为完成训练。完整假设、证据路径与下一步见 JSON/CSV。

| 实验 | 候选阶段 |
|---|---|
| AdaRES-E1 | rejected |
| AdaRES-F1 | rejected |
| AdaRES-F2 | rejected |
| AdaRES-F3 | rejected |
| AdaRES-H1 | rejected |
| AdaRES-H2 | rejected |
| AdaRES-H3 | rejected |
| AdaRES-H4 | rejected |
| AdaRES-J1 | rejected |
| AdaRES-J2 | rejected |
| AdaRES-K1 | rejected |
| AdaRES-K2 | rejected |
| AdaRES-L1 | rejected |
| AdaRES-L2 | rejected |
| AdaRES-M1 | rejected |
| AdaRES-M2 | rejected |
| AdaRES-N1 | rejected |
| AdaRES-N2 | rejected |
| AdaRES-N3 | rejected |
| AdaRES-O1 | rejected |
| AdaRES-O2 | rejected |
| AdaRES-P1 | rejected |
| AdaRES-P2 | rejected |
| AdaRES-Q1 | rejected |
| AdaRES-Q2 | rejected |
| AdaRES-R1 | rejected |
| AdaRES-R2 | rejected |
| AdaRES-S1 | rejected |
| AdaRES-S2 | rejected |
| AdaRES-T1 | rejected |
| AdaRES-T2 | rejected |
| AdaRES-U1 | rejected |
| AdaRES-U2 | rejected |
| AdaRES-V1 | rejected |
| AdaRES-V2 | rejected |
| AdaRES-W1 | rejected |
| AdaRES-W3 | rejected |
| AdaRES-X1 | rejected |
| AdaRES-X2 | rejected |
| AdaRES-Y1 | rejected |
| AdaRES-Y2 | rejected |
| AdaRES-Y3 | rejected |
| AdaRES-Y4 | rejected |
| AdaRES-Y5 | rejected |
| AdaRES-Y6 | rejected |
| AdaRES-Y7 | rejected |
| AdaRES-Y8 | rejected |
| AdaRES-Y9 | rejected |
| AdaRES-Y10 | rejected |
| AdaRES-Y11 | rejected |
| AdaRES-Y12 | rejected |
| AdaRES-Y13 | rejected |
| AdaRES-Y14 | rejected |
| AdaRES-Y16 | rejected |
| AdaRES-Y19 | rejected |
| AdaRES-Y20 | rejected |
| AdaRES-Y21 | rejected |
| AdaRES-Y22 | rejected |
| AdaRES-Y24 | rejected |
| AdaRES-Y26 | rejected |
| AdaRES-Y29 | rejected |
| AdaRES-Y32 | rejected |
| AdaRES-Y34 | rejected |
| AdaRES-Y38 | rejected |
| AdaRES-Y40 | rejected |
| AdaRES-Y46 | rejected |
| AdaRES-Y47 | rejected |
| AdaRES-Y49 | rejected |
| AdaRES-Y51 | rejected |
| AdaRES-Y53 | rejected |
| AdaRES-Y55 | rejected |
| AdaRES-Y86 | blocked |
| AdaRES-Y97 | blocked |
| AdaRES-Y103 | blocked |

## 重要解读

- Y36 原始完整后评的 FIRST：Y36 0.510716624292、W2 0.510009849923、B1 0.502112417447；来源为 AdaRES-Y36_frozen_review_1fb792ef.md。
- 后续冻结后评各自包含同设备 FIRST 对照，数值可能变化；不能把不同设备或重复次数的基线拼成一套配对结果。
- Y100 训练 best 0.5164485573768616 超过目标 0.515，但冻结正式后评尚未完成，不能宣布科学目标达成。
- Y90/Y92/Y94 完整后评为实质子组伤害，拒绝范围是该实现；Y96/Y99/Y101 为 complete_inconclusive，不晋升。
- Y88 八次原生预测已保存；正式完整比较/成本/后审尚未完成，成本从未启动，server-1 暂停约束仍有效。
- Y102 原进程存在但原生日志停滞；Y57 用户单独暂停；Y103 仅代码准备，未做 native/准入。
- 本表不计算科学成功率；候选 complete、训练 done_complete 与科学结论是三个维度。
