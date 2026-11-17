# Experiment Result Analysis - AdaRES-Y23

## 1. Basic Info
- 配置：configs_song/model_paper5/adares_y23_valid_token_support_res.yaml；server-2 GPU1；固定30轮、seed1999，无验证集调参。
- 用户要求的同一实验暂停/迁移已执行：保留原epoch4模型、AdamW和PolyLR，恢复到epoch5，总轮数不变。未保存的epoch5步数重放，不声称整个数据加载器/RNG或逐位轨迹一致。
- 本轮19:50:05至19:50:08本地观察：原恢复PID129584前后均不存在，无匹配训练进程；最终Val30/30 363/363，last epoch30，best epoch16，读取期间checkpoint不变。精确训练结束时间/耗时未知，远端时钟不作本地时间替代。
- 原生训练完成；规范监控仍为running/deathwatch2/3。未手工修改队列生命周期、重复preflight或重启训练。

## 2. Best Metrics
三种方法各自最佳checkpoint，均完成两遍全量9508表达式/141场景；以下为第一遍，非选择较好的一遍。

| Metric | Y23 | W2 | B1 |
|---|---|---|---|
| mIoU | 0.507471535 | 0.510104850 | 0.502221764 |
| Acc25 | 0.610012621 | 0.610748843 | 0.610433319 |
| Acc50 | 0.558161548 | 0.562999579 | 0.552902819 |
| spmIoU | 0.470801008 | 0.473716041 | 0.466782141 |
| 可用/选中query | 128/1 | 128/1 | 128/1 |
| 参数量 | 164629637 | 164629637 | 164430473 |
| 全量验证peak allocated GiB | 4.042351 | 4.042351 | 4.040677 |
| 全量验证peak reserved GiB | 6.486328 | 6.486328 | 6.484375 |

## 3. Convergence Summary
- 原best metadata mIoU0.507580519/epoch16；末轮原日志四舍五入mIoU0.504、Acc50 0.555、Acc25 0.601、spmIoU0.468。不是第一遍最佳checkpoint复评结果。
- 两遍mIoU：Y23 0.507471535/0.507463649，W2 0.510104850/0.510107995，B1 0.502221764/0.502119335。
- 单表达式最大重复IoU差分别0.044250607/0.029905617/0.973895609；平均小波动不代表所有表达式稳定。两遍只估计已观察噪声，不是通用数值上界。
- 上轮部分评估complete=false/completed_passes1及哈希807af4e6保留；其第一遍约100秒，不能据此归因580秒限时不足。本轮三次完整执行分别约198/198/189秒，精确上轮中断原因仍未知。

## 4. Comparison
- 目标0.515未达到，不晋升冠军。
- vs W2配对mIoU差-0.002633315，整场景10000次bootstrap/seed1999，95%CI[-0.010861580,0.005471885]。
- vs B1差+0.005249770，95%CI[-0.003244001,0.013465872]。
- 全部small/medium/large组区间均跨零；冻结原训练尺寸切点1487/3325，不按结果选择子组或重新划界。收益和材料伤害均不确定，不证明等价或安全。
- 按冻结方案使用max(candidate/reference mean(abs(repeat difference)))作为每组经验噪声，保留旧abs(mean difference)参考字段。Y23/W2全量噪声9.865979e-6，Y23/B1 0.000102429。四项合成控制通过：抵消不消失、零差、伤害需跨过边界、表达式集合不匹配拒绝。历史Y18/Y17证据未改写。

冻结训练成本：相同前8个独立训练场景及原最大场景对indices728/6780；两个场景按原生predict batch=1串行执行，各2次预热/3次测量，Y23/W2各自训练完成最佳权重，不同时驻留两个模型。

| 测量汇总 | Y23 | W2 |
|---|---|---|
| train8 median秒 | 0.078542245 | 0.078383661 |
| train8描述性p95秒 | 0.120190539 | 0.118815105 |
| train8 peak allocated/reserved GiB | 0.960723/1.234375 | 0.960723/1.234375 |
| 最大场景对串行读数median秒 | 0.188517095 | 0.190241622 |
| 最大场景对串行读数描述性p95秒 | 0.218374875 | 0.226114751 |
| 最大场景对peak allocated/reserved GiB | 1.243689/1.757813 | 1.243689/1.757813 |

- train8 median/p95相对回退+0.202%/+1.158%；最大场景对-0.906%/-3.423%；如实报告，不认定统计显著性能差异。
- 汇总读数混合场景；每场景只有3次采样，p95只具描述性，不能可靠估计尾延迟。固定candidate-then-parent串行顺序，瞬时硬件因素未隔离。没有质量优势，不宣称Pareto优势。

## 5. Conclusion
- 冻结验证执行完成，科学效果inconclusive，目标未达；同一稳定候选记录为complete，不是rejected，也不是冠军/因果机制收益证明。
- 不重复训练、改轮数、调权重、挑子组或换名字重新准入。规范训练终态等待已有监控正常确认；本轮准入预算0，不制造新准入。
- official unique/multiple未知；单训练seed、最佳checkpoint比较不量化训练seed不确定性，也未提供独立因果训练对照。保留历史原生推理不稳定与恢复数值差异。

## 6. Failure And Evidence
- 成本测量器首次把最大场景对拼为batch2，原生predict输出长度1055503与单场景GT537406不匹配。仅修正测量器为同一两场景串行batch1，未改模型或科学清单；原失败日志SHA256 17b66b2d569c7a34bfcfd3e5775561193e6af26fd4f05cd93d493f8e4229c5a3保留。这不是原生训练崩溃，也不放宽准入门槛。
- 本轮终态JSON：y23_terminal_cc2267b1187942dd8269b3cf600bc5d2.json，SHA256 291882710c45ed4ccc42555feb1f9a51a02ce77a7f4f6fded3457d216c60cb71。
- Y23/W2/B1全量JSON哈希：de27e49be74b54c80a8f3b6fe6f773d2448488ef8309a8045c4cc34c7896812e / 83e8cbef2ba78b0afb13117bfef23b1908f849dafb0ebd1bd4f98de0d0ff28b5 / 6afbc51608f5f4adbd6b63dcfcaa68ce8c700dc3296b727388b3576fcdbaa1ca；原始stdout/stderr日志保存。
- 整场景比较哈希00f3f97fd4ae7079379992e50854145d1b669ca9d4d30544e5992152fcb0fcb9；串行成本原始JSON 5f99b9adb3245f86aebedfb40b1e55509564d228eb6b1c8a09c1c7d2ebaa4fbf；成本汇总0eecdf5f8978d72a573314f92d8b3d308fc413fe2b21bfc20cd6ab0cc3bc138d。
- 全量复评窗口19:50:15至20:01:15，串行成本20:03:54至20:04:57，本地时间。20:06:04 GPU1 1MiB/0%，无compute PID；GPU0未查询或使用。占用只代表观察瞬间。
- 本轮postflight与租约释放仍待实际finish确认；此报告不预先声称执行COMPLETE。
