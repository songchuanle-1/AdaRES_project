# Research Pipeline Report - AdaRES_project
快照时间：2026-10-04T10:29:13.634927+00:00；生成报告时快照年龄：1579 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-04 18:29:13
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-W2 — mIoU=0.5100 (gap to target: +0.0050)
       Epochs: 30/30, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0080

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5100, target=0.5150, gap=+0.0050
================================================================================
```

## 1. GPU Overview
- 仅列授权卡；未采集利用率、显存或进程时标为未知，不推断空闲。
| Server | GPU | Util | Memory | Process |
|---|---|---|---|---|
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-04 18:29 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y25_mask_feedback_res] 进程未见 (deathwatch 2/3) — 暂不处置

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y25 | configs_song/model_paper5/adares_y25_mask_feedback_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 用户手动要求继续设计。begin仅执行一次；18:29:13北京时间event快照active1/running1/capacity1/waiting0/budget0/due0，Y25deathwatch2/3无真实训练进程。未重复preflight，未修改queue生命周期；GPU0保留未查询/使用。
- 新建稳定候选AdaRES-Y26，预注册后实际TRAIN64/256表达式审计qla被丢弃的LayerNorm路径。原实现/repeat/identity完全相同；原六层affine全1/0且完整loss无梯度。Connected输出真实改变，TRAIN64点IoU .743230vs原.741665/RMS.746795；25正26负13相同，不能宣称可靠收益。
- 实际trained8/init4完整native反传：接回12张量均finite/nonzero，最小梯度.004308。总loss初始增加+.07624/+.03564，不利结果保留；跟进预注册缓存decoder+criterion中心化/RMS/affine-detach对照，证明主要是尺度变化带来初始loss上升。Cached connected/detached/repeat前向0；共享梯度roundoff2.38e-7/repeat5.96e-8保留，不宣称bitwise或中心化特有优势。
- 建立独立Y26六文件源码包和字节相同parent配置，唯一区别是qla postnorm在TRAIN/EVAL均接回；七文件hash/AST范围/无跨variant import通过。原W2/Y25/train.py未改。Native tools.train2 optimizer steps实际更新全部12affine张量，412个decoder梯度张量finite。模型/AdamW/PolyLR恢复前完全相同；下一步loss0/model1.526e-5/optimizer4.618e-9误差保留。
- 实际预测盲1055503points/7618SP大场景压力：W2/Y26 params164629637/0新增，训练median.903605/.902047s、reserved13.400391/13.417969GiB。串行TRAIN8+固定最大2场景推理median.092301/.091173s、reserved1.757813GiB双方，128queries/1selected；三读p95仅描述，未训练Pareto未知。
- 18:49:20至18:49:21本地独立只读核实GPU1 1MiB/0%/无computePID、原Y25PID不存在。规范源包、五项GPU诊断输出、因果审查和后训练冻结方案均hash绑定；13个源/脚本syntax通过，memory119行。
- 科学状态：Y26机制/工程六门通过并进入admission审查阶段，但未admitted/queued/正式training/scoped commit，效力与新颖性未知。执行状态：此notes写入时postflight尚未执行，以finish/receipt及独立锁核实为准。
- 本轮收据尚未产生，用量和窗口暂记未知；finish后在任务展示实际begin至receipt局部窗口与产出。cache属于input、reasoning属于output，不重复相加，不作账单。当前实际新候选1/正式训练启动0；最终计数以收据为准。

## 5. Decisions Made
- 保留Y26同一稳定ID和同一预注册结构，不因RMS/center控制读数改模型、不调参或凑变体。已有Y25拒绝结论保留，不重开、换名或提升champion。
- 本轮预算0禁止额外准入，不将物理空GPU伪称资源阻塞，也不伪造queue或重复monitor。下次正常tick解除Y25canonical旧记录且有合法budget后，重核hash/约束，scoped commit测试源、六门receipt、canonicaldispatch并核实native递增steps；本轮无后台或新任务操作。
- 冻结Y26后训练30轮/seed1999/releasedSPFormer、ScanRefer全9508/141场景、Y26/W2/B1两次重复、scene-cluster10000bootstrap、TRAIN尺寸界1487/3325和material-harm规则；收益/Pareto最终由真实训练决定。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
