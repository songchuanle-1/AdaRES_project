# Research Pipeline Report - AdaRES_project
快照时间：2026-10-10T13:49:05.340115+00:00；生成报告时快照年龄：96 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-10 21:49:05
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
================================================== 2026-10-10 21:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y39_point_distribution_res] Best=0.3200 2/30 no_improve=0/30 gpu=1 ?
  [adares_y37_pair_distribution_res] Best=0.5018 12/30 no_improve=0/30 gpu=0 ? 序列: 0.4697, 0.4747, 0.4926, 0.4920, 0.5018

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y37 | configs_song/model_paper5/adares_y37_pair_distribution_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y39 | configs_song/model_paper5/adares_y39_point_distribution_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 仅处理 AdaRES_project；当前环境任务 ID 与 execution_mode 的前台归属一致，closure_policy 已启用，Windows Pipeline 保持 Disabled。核对目标、GPU0/1 授权和项目指令一次，begin 仅一次；先读紧凑视图，再读短记忆及本轮报告，不重复 preflight。
- 本轮 event 的实际原因仅为 research_artifact_changed。快照 2026-10-10 21:49:05.340115+08，真实截止 22:49:05.340115+08；运行 2、认领 0、等待 0、目标容量 2、准入预算 0，新终态 0、到期候选 0，没有需要恢复的到期诊断。
- 原始监控显示 Y37 在 GPU0 为 12/30，舍入最佳 mIoU 0.5018，无改善 0/30；Y39 在 GPU1 为 2/30，舍入最佳 mIoU 0.3200，无改善 0/30。这是训练中监控，不是冻结的训练后全量比较；结构化表的未采集指标、当前 PID、GPU 利用率及显存均保留未知，历史启动证据不当作实时占用。
- 保持两项固定 30 轮训练及所有冻结协议。本轮没有新增候选、准入、训练启动或重复实验，没有改写历史科学证据；仅更新七段短记忆并补报告第 4/5 段，复用脚本第 0 至 3 段事实及规定表格。
- 执行状态：本轮已获得有效租约，finish 尚未执行，不能预称完成；只有真实 CLOSURE_POSTFLIGHT_PASS、COMPLETE 和独立锁释放核实后才称 COMPLETED。科学状态：Y37/Y39 为 training，训练后有效性、必要性、无伤害及 Pareto 结论未知。
- 收据用量窗口为本轮 begin 至收据记录，排除更早读取、迟到用量和最终回复；当前收据未生成，调用数、输入、缓存、输出、推理及实际产出统计未知。finish 后读取真实窗口和产出，不重复累计前轮；缓存属于输入，推理属于输出，不重复相加，不把用量当账单，释放后不改写报告。

## 5. Decisions Made
- 本轮无新增科学决策。目标容量已达到 2/2，且无终态或到期候选，按 event 合同收尾；不将训练中数值提升当作晋级，不为填满时间凑变体或重复测试，不虚构耗尽、阻塞或冷却。
- ScanRefer 目标仍为 0.515，自动数值冠军 Y36 舍入为 0.5110，尚未达标。保留 W2 科学比较基准、Y36 complete_inconclusive 和各候选已冻结的训练后评估协议，不进入未授权的 Phase 2。
- 后续仅处理真实终态、崩溃、到期候选或已冻结比较；不操作其他项目、调度或后台，不新建任务、不嵌套 codex exec。下一调度时间未知。本轮如 finish 失败，保留证据、按真实原因 abort 并核实释放。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
