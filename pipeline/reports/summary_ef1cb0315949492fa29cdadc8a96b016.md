# Research Pipeline Report - AdaRES_project
快照时间：2026-09-28T04:17:39.973490+00:00；生成报告时快照年龄：270 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-09-28 12:17:39
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
================================================== 2026-09-28 12:17 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程

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

## 4. This Tick's Actions
- 本轮前台 begin 在 2026-09-28 12:17:39 +08:00 完成监控和目标检查，返回码均为 0；候选 due_count 从误报的 12 降为 0，确认 closed 终态分类修复生效。
- 核对先前中止的异常标记及冷却签名：运行前已有 .tick_unhealthy，故障分析文件改变研究签名，先前冷却不再有效；本轮未进行新的科学诊断、候选准入或训练。
- 本轮仅作故障恢复检查；postflight 及租约最终状态以随后的 finish/abort 命令和收据为准。

## 5. Decisions Made
- 不把历史 X1/X2 结果冒充本轮候选进展，也不改写 goal、准入门槛、冷却签名或 Windows 调度。
- 验证 postflight 对本轮设计进展的真实要求；若缺少新增候选证据，记录 INCOMPLETE 并释放租约，留给之后的真实研究 tick。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
