# Research Pipeline Report - AdaRES_project
快照时间：2026-10-08T23:48:44.138048+00:00；生成报告时快照年龄：49 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-09 07:48:43
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
================================================== 2026-10-09 07:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y33_token_attention_res] Best=0.5026 20/30 no_improve=5/30 gpu=1 ? 序列: 0.5011, 0.4975, 0.5013, 0.4999, 0.5026

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
- 仅执行 AdaRES_project 前台 tick，实际任务 ID 与 owner 一致，最新 closure_policy 启用，Windows 项目调度 Disabled；server-2 仅 GPU1、GPU0 未授权、server-6 跳过，未操作其他项目、调度或后台。
- begin 一次后先读紧凑视图，再读短记忆和脚本报告。monitor_rc 与 goal_rc 均 0，event 原因为 research_artifact_changed；快照北京时间 2026-10-09 07:48:44.138048，实际截止 08:48:44.138048，未重复 preflight 或修改截止。
- 复用第 0 至 3 段事实表：容量 1、活跃 1、运行 1、认领及等待 0、准入预算 0、到期候选及新终态 0。Y33 监控摘要 20/30、四舍五入 best mIoU 0.5026、no_improve 5/30；尚在训练，非冻结全量结果，未填补事实表未知指标或推断实时 GPU 空闲。
- 只更新七段短记忆及中文报告动作、决策；无故障、终态或到期诊断，未重复健康进度探针、执行 GPU 测试、修改候选或历史证据、改动模型、准入或重启。科学状态 Y33 training，效益、必要性、新颖性及训练后 Pareto 未知，Phase 1 未达成。
- 写报告时 finish、收据窗口、用量及实际产出均待验证。finish 后读取真实收据并独立核实锁消失，释放后不改写项目；缓存包含于输入、推理包含于输出，不重复相加。本地窗口不包括更早读取、较晚记录和最终回复，不是账单。

## 5. Decisions Made
- 同一稳定候选 Y33 继续既定训练，本轮无到期诊断、终态或准入余量，不生成新设计、占位变体或等价测试，不以训练读数声称冻结科学进展。
- 保持 fixed30、seed1999、原生 128 queries、标准 argmax、冻结全量双次配对及子群损害规则。20/30 和 no_improve 5/30 均不授权改协议、调参或提前停止；四舍五入后相同的数值不用于推断精确相等或监控故障。
- 下一现有前台 tick 检查实际进度、指标对齐和终态，实质变化才触发针对性研究。具体调度时间本轮未读取，记未知，不更改调度或恢复后台。
- 仅 finish 明确 PASS/COMPLETE 且所属锁消失后称本轮执行完成；失败则保留证据、按原 token abort 并核实释放，不修改门槛、closure_policy 或历史冻结。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
