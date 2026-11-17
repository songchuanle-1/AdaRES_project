# Research Pipeline Report - AdaRES_project
快照时间：2026-10-08T16:49:38.933354+00:00；生成报告时快照年龄：89 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-09 00:49:38
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
================================================== 2026-10-09 00:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y33_token_attention_res] Best=0.4931 10/30 no_improve=0/30 gpu=1 ? 序列: 0.4651, 0.4730, 0.4735, 0.4750, 0.4931

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y33 | configs_song/model_paper5/adares_y33_token_attention_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 本轮仅AdaRES_project。核实实际任务归属、Windows项目调度Disabled、closure_policy启用及前轮锁已消失，保留GPU0禁用与server-2仅GPU1授权；未操作其他项目、调度或后台。
- begin仅一次，首先读取紧凑视图；快照北京时间2026-10-09 00:49:38.933354，event，真实截止2026-10-09 01:49:38.933354，未延长。monitor/goal均0，容量1、活跃1、运行1、认领0、等待0、预算0、到期0、新终态0；原因research_artifact_changed/research_work_changed。
- 同一Y33监控摘要10/30、Best mIoU=0.4931、no_improve=0/30，最近五个验证值0.4651、0.4730、0.4735、0.4750、0.4931。本快照已返回正常舍入指标，未再出现前轮日志对齐警告；仅说明当前摘要可用，不宣称根因查明或永久修复，没有手动改监控器或绕过保护。早期数值不是冻结全量评估、最终收益或champion晋升依据。
- 规定0-3事实表仍基于本轮preflight，不可得的指标、GPU利用率/显存/进程保持未知。前轮2026-10-08 23:49的原生步数及22800MiB/75%均为历史证据，不冒充本轮实时数据；GPU0未查询或使用。本轮未重跑等价进度探测、preflight、诊断或历史审查。
- 仅读取短记忆与本轮报告一次，更新七段操作记忆并保存中文动作/决策，复用六段事实表。没有候选、台账、证据、设计、准入、模型或训练设置变更，没有新启动/重启或Y23恢复。写入时本轮仍待finish，科学状态training，效果、必要性、创新性与训练后成本优势未知；前轮d97ad104已COMPLETE、FINALIZED/PASS并核实锁消失。
- 本轮收据在写入时尚未生成，窗口/用量/实际变更计数未知；finish后核实收据与释放，不作释放后写入。统计仅本地begin至收据的局部窗口，排除早读、晚到记录和最终答复；缓存包含在输入、推理包含在输出，不重复累加或当账单。实际产出为操作记忆更新与六段报告，无新增科学证据。

## 5. Decisions Made
- 无新科学决策。继续同一Y33固定30epoch/seed1999/原生128queries/单次标准argmax；0.4931为早期日志数值，不改目标0.515、W2历史champion0.5100、门槛、训练或冻结评估协议。
- 活跃目标已满足、预算0、到期0，不增加候选或无信息增益复测。继续按规范监控观察真实故障/终态及指标对齐；终态后执行既定冻结全量、整场景bootstrap、训练来源分组和配对成本方案，不将本快照正常摘要当作永久对齐保证。
- 本轮为例行在训状态，无需用户操作或主动提醒。真实postflight通过并独立核实释放后安静收尾，tick完成不代表科学成功。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
