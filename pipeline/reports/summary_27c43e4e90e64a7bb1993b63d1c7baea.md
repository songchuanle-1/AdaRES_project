# Research Pipeline Report - AdaRES_project
快照时间：2026-10-10T04:49:56.216015+00:00；生成报告时快照年龄：69 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-10 12:49:55
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
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-10 12:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y37_pair_distribution_res] 1/30 尚无验证 - gpu=0
  [adares_y36_local_adjacency_res] Best=0.5099 24/30 no_improve=2/30 gpu=1 ? 序列: 0.5018, 0.5082, 0.5099, 0.5084, 0.5082

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y36 | configs_song/model_paper5/adares_y36_local_adjacency_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y37 | configs_song/model_paper5/adares_y37_pair_distribution_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 本轮快照为2026-10-10 12:49:56+08，模式event；归属已核实，closure启用、Windows调度禁用。begin只执行一次，未重复preflight；第0-3段直接复用该快照事实表。
- 此次event承接上一轮已完成的队列、研究产物和候选状态变化；新增终态0、待续候选0。科学状态：Y36/Y37仍training，Y35此前complete_inconclusive，均不是本轮新结论。
- 本轮原始监控：Y36/GPU1为24/30轮、舍入best0.5099、未改善2/30；Y37/GPU0为1/30轮、尚无验证。表格未采集字段保留未知，不把训练期best作为冻结评估，不根据未知利用率声称GPU空闲。
- 活跃2/目标2、等待0、准入预算0；保留两项原固定30轮训练。没有新诊断、设计审查、候选、准入或派发，没有修改科学证据、checkpoint、调度或其他项目。
- 仅更新七段研究记忆及完整六段报告，保留上一轮真实收尾和证据引用；收尾postflight尚待执行，不能提前称COMPLETED。
- 本轮收据用量窗口和实际产出将在finish后读取；生成当前报告时其数值未知。统计只覆盖begin到收据的本地部分窗口，不是账单；缓存包含在输入、推理包含在输出，不重复相加。

## 5. Decisions Made
- 无新研究决策；保留W2、目标0.515和冻结评估规则，等待原Y36/Y37终态或故障证据后续做。
- 两项原训练持续中，没有需用户处理的实质变化；不为填满截止时间重复设计或测试，不盲目重启或降低准入门槛。
- 本轮完成以实际finish PASS/COMPLETE及独立租约锁释放为准；自动化不变，下次精确调度时间未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
