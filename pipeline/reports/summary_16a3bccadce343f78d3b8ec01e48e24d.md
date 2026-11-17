# Research Pipeline Report - AdaRES_project
快照时间：2026-10-19T02:52:32.072334+00:00；生成报告时快照年龄：3 秒；模式：steady

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 10:52:31
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
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-19 10:52 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [不可达]

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2:
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| 无 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 快照内无运行任务 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 2；准入预算 2。
- 研究储备 2 / 2；储备准备预算 0，不增加训练并发或启动授权。
- 候选累计记录 107：complete=14 / rejected=79 / closed=12 / blocked=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 30（含修订或重试，不代表成功率）。
- 研究等待：储备已满且仅有未到复查时间的外部阻塞；复查条件时间 2026-10-19T13:21:55.539122+08:00，不代表已解锁或获得启动授权。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y52 | Human explicitly authorizes only the seven exact tested Y52 files, or supplies their exact commit; then revalidate hashes/current goal and six gates, issue bound receipt and use canonical dispatcher with fresh idle checks. Do not deploy diagnostic W2 weights. | 2026-10-19T13:21:55.539122+08:00 |
| AdaRES-Y54 | Explicit authorization ONLY for the seven exact Y54 files in tested_bundle, or supply their exact commit. Then revalidate current goal/source/gates/commit and use existing dispatcher with live idle checks and normal Y50 terminal reconciliation. Do not extend Y50 authorization, commit other files, or deploy diagnostic weights. | 2026-10-19T13:22:24.348599+08:00 |

## 4. This Tick's Actions
- begin 已执行监控和目标检查；此表基于该快照，不代表 tick 结束时的实时状态。
- 本报告尚未执行 postflight，结果以 finish 命令及收据为准。
- 无新增可处理事件；直接生成报告，不重复设计或改写未变记忆。

## 5. Decisions Made
- 本轮无新决策。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
