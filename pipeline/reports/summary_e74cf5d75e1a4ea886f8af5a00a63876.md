# Research Pipeline Report - AdaRES_project
快照时间：2026-10-08T18:50:53.822088+00:00；生成报告时快照年龄：75 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-09 02:50:53
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
================================================== 2026-10-09 02:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y33_token_attention_res] Best=0.4964 13/30 no_improve=0/30 gpu=1 ? 序列: 0.4750, 0.4931, 0.4958, 0.4949, 0.4964

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
- 本轮仅处理 AdaRES_project。实际任务 ID 与前台 owner 一致；按最新配置、项目指令和用户授权核对范围，Windows 项目调度仍禁用，未操作其他项目、调度或后台。
- begin 仅执行一次，首先读取紧凑模型视图；监控和目标检查均返回 0。快照为 2026-10-09 02:50:53.822088+08，实际截止为 03:50:53.822088+08，未修改截止或重复 preflight。
- 复用第 0 至 3 段脚本事实表：容量 1、活跃 1、运行 1、认领 0、等待 0、准入预算 0、到期候选 0、新终态 0。Y33 原生监控摘要为 13/30、四舍五入 best mIoU 0.4964、no_improve 0/30；这是早期训练监控，不是冻结全量结果，也不填补事实表中的未知指标或 GPU 占用。
- 仅更新七段短记忆和本轮中文动作、决策报告；未重复进度探针、执行 GPU 诊断、改动候选或历史证据、修改模型、准入、重启或追加训练。此前指标流对齐警告在本快照未再出现，原因及永久修复仍未知。
- 科学状态仍为 Y33 training，效益、必要性、新颖性及训练后 Pareto 未知，Phase 1 未达标、Phase 2 未授权。本轮报告生成时 finish 与执行收尾仍待验证；不能提前称完成。
- 本轮收据的实际用量窗口、输入/缓存、输出/推理、调用数及产出计数在报告写入时未知，需 finish 后读取真实收据并核实锁释放，不在释放后改写项目。缓存属于输入、推理属于输出，不重复相加；本地窗口不包括更早读取、较晚日志及最终回复，不是账单。

## 5. Decisions Made
- 只继续观察同一稳定候选 AdaRES-Y33：本轮没有终态、故障、到期诊断或准入余量，不产生新设计或等价重复测试，不以早期指标小幅变化声称科学进展。
- 保持既定 fixed30、seed1999、标准 argmax、原生 128 queries、冻结全量配对评估及子群损害准则；GPU0 禁用、server-2 仅 GPU1、server-6 跳过，历史冻结和所有反证保持不变。
- 下一次按现有自动化检查真实训练进度、指标对齐和终态；明确的新故障或终态才触发有信息增益的处理。具体调度时间本轮未读取，记未知。
- 仅 finish 明确 PASS/COMPLETE 且核实当前所属锁已消失后，称本轮执行完成；失败则保留证据、按原 token abort 并核实释放，不改准入门槛或 closure_policy。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
