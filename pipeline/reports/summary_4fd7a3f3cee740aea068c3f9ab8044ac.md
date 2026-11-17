# Research Pipeline Report - AdaRES_project
快照时间：2026-11-01T10:45:55.275483+00:00；生成报告时快照年龄：316 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-01 18:45:54
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-Y36 — mIoU=0.5110 (gap to target: +0.0040)
       Epochs: ?, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0090

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5110, target=0.5150, gap=+0.0040
================================================================================
```

## 1. GPU Overview
- 仅列授权卡；未采集利用率、显存或进程时标为未知，不推断空闲。
| Server | GPU | Util | Memory | Process |
|---|---|---|---|---|
| server-1 | 0 | 未知 | 未知 | 未知 |
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-11-01 18:45 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y72_visual_mask_refinement_res] Best=0.5004 30/30 no_improve=15/30 gpu=0 ? 序列: 0.4938, 0.4902, 0.4878, 0.4920, 0.4871

--- server-2 (192.168.1.163) ---
  [adares_y74_language_update_gate_res] Best=0.5026 16/30 no_improve=1/30 gpu=0 ? 序列: 0.4985, 0.4898, 0.4929, 0.5026, 0.4993
  [adares_y75_language_query_init_res] Best=0.4415 5/30 no_improve=0/30 gpu=1 ? 序列: 0.3000, 0.3771, 0.4384, 0.4415

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y74 | configs_song/model_paper5/adares_y74_language_update_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y75 | configs_song/model_paper5/adares_y75_language_query_init_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y72 | configs_song/model_paper5/adares_y72_visual_mask_refinement_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 129：complete=16 / rejected=79 / closed=12 / blocked=19 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 53（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 先查三原运行并尝试已有合格队列派发，waiting0；实际native计数递增，Y72 Train30仍在训练、Y74 Train16、Y75 Val5，原源/配置/功耗/收据和独占保持，无新终态。
- 核Y76/Y77已测试版本、四份server2 native计划、Y76 server1备用计划/CPU预检证据及Y72完整终态准备依赖；不重复已经完成的准备，也未把CPU回调视作GPU检查。
- 实时canonical occupation确认三授权卡都有原compute进程与预约；本轮实际resource_check受阻，无GPU检查/准入/启动。两个observer初并发未获project lock，随后串行成功，后续保持串行。
- 保存当前handoff与七段记忆，正式FIRST/成本无更新，Y57继续暂停。

## 5. Decisions Made
- 先推进现有两项独立准备Y76/Y77，等待原slot owner自然fixed30归档及fresh idle/no compute/no reservation；不提前停止原训练、不重复CPU或扩充同机制副本。
- Y72未完成不执行终态helper或激活Y76。Y72完成并完整核验归档后优先Y76 server1原生两步，通过就同轮原完整协议freeze/准入/安全门控派发；保留专用恢复要求。
- 科学goal/原数据/seed1999/fixed30/禁用项和350W/280W/390W保持；不触碰server6或其他项目，不改频率。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
