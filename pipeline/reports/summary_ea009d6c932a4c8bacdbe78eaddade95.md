# Research Pipeline Report - AdaRES_project
快照时间：2026-10-08T19:49:08.117773+00:00；生成报告时快照年龄：57 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-09 03:49:07
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
================================================== 2026-10-09 03:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y33_token_attention_res] Best=0.4992 14/30 no_improve=0/30 gpu=1 ? 序列: 0.4931, 0.4958, 0.4949, 0.4964, 0.4992

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
- 仅对 AdaRES_project 执行本轮前台 tick，实际 CODEX_THREAD_ID 与配置 owner 一致；读取唯一入口的前台规则及最新配置、项目授权，项目 Windows 调度禁用，未操作其他项目或调度。
- begin 一次后先读紧凑模型视图，再读取短记忆及脚本报告。快照为北京时间 2026-10-09 03:49:08.117773，实际截止 04:49:08.117773，未重复 preflight 或修改截止。monitor_rc 与 goal_rc 均 0，模式 event 的唯一原因是 research_artifact_changed。
- 复用脚本事实表：容量与活跃均 1、运行 1、认领及等待 0、准入预算 0、到期候选及新终态 0。Y33 监控摘要为 14/30、四舍五入 best mIoU 0.4992、no_improve 0/30；这些是早期训练读数，非冻结全量结果，未替代事实表未知值，也未推断 GPU 空闲。
- 科学状态仍为 Y33 training，效益、必要性、新颖性及训练后 Pareto 未知，目标仍未达成。仅更新七段短记忆和本轮中文报告动作、决策；未重复健康进度探针、进行 GPU 诊断、修改候选或历史证据、改动模型、准入或重启。
- 报告写入时执行收尾与本轮收据窗口、用量、产出均待验证，不能提前称完成。finish 后读取真实收据并独立核实锁消失，释放后不改写项目；缓存已包含于输入，推理已包含于输出，窗口排除更早读取、较晚日志和最终回复，不能当作账单。

## 5. Decisions Made
- 同一稳定候选 Y33 有训练进度但没有终态、故障或到期诊断，无需追加测试或新设计。准入目标已由既有设计周期满足，本轮不虚构研究耗尽、资源阻塞或新科学进展。
- 保持 fixed30、seed1999、原生 128 queries、标准 argmax、冻结配对全量评估和既定子群损害规则。GPU0 仍未授权，server-2 仅 GPU1，server-6 跳过；不重演历史 Y23 恢复操作或放宽门槛。
- 下一次现有前台 tick 继续监控真实进度、指标对齐及终态；发生实质故障或冻结结果时再开展针对性处理。具体调度时间本轮未读取，记未知。
- 仅 finish 返回 PASS/COMPLETE 且所属锁已核实释放后称本轮执行完成；失败则保留证据、按原 token abort 并核实释放，不修改 closure_policy 或历史冻结。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
