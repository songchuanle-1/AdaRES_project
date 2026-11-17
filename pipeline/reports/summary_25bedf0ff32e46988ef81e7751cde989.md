# Research Pipeline Report - AdaRES_project
快照时间：2026-10-10T23:49:26.580972+00:00；生成报告时快照年龄：64 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-11 07:49:26
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
================================================== 2026-10-11 07:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y39_point_distribution_res] Best=0.5030 15/30 no_improve=0/30 gpu=1 ? 序列: 0.4919, 0.4947, 0.4954, 0.4940, 0.5030
  [adares_y37_pair_distribution_res] Best=0.5052 24/30 no_improve=4/30 gpu=0 ? 序列: 0.5052, 0.5003, 0.5002, 0.5010, 0.5047

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
- 本轮 token=25bedf0ff32e46988ef81e7751cde989。已核实实际任务 ID 与 foreground owner 均为 01a07e32-78bf-7670-925b-370911ea48db；最新模式 foreground、闭环策略启用、Windows Pipeline-AdaRES_project Disabled。按目标、项目指令及 server-2 GPU0/1 授权仅操作 AdaRES_project。
- begin --auto-steady 仅执行一次并首先读取 .tick_model_view.json。快照 2026-10-11 07:49:26.580972+08，实际截止 08:49:26.580972+08，模式 event、原因 research_artifact_changed，monitor_rc=0、goal_rc=0；未重复 preflight。
- 复用脚本事实表：活跃/运行 2，认领/等待 0，目标容量 2，准入预算 0，到期候选 0，新增终态 0。没有列出的待处理候选，不加载完整快照、历史或全部引用。
- 原始监控日志为 Y37 24/30、舍入最佳 mIoU=0.5052、no_improve=4/30；Y39 15/30、舍入最佳 mIoU=0.5030、no_improve=0/30。结构化实验表指标、实时 PID、GPU 利用率和显存仍未知，保留脚本表值，不据历史占用推断空闲。
- 科学状态仍为两项 TRAINING/training_not_efficacy，无新增终态或故障；Y39 本轮无对齐告警，历史 02:48:12 不可用证据及实际原因未知保留。现有固定 30 epoch 训练未重启、提前停止或调参，未重复旧评审、解析器检查、GPU 探测或派发；租约内仅更新七段短记忆与六段报告。
- 本轮未新增实验或测试；新增候选、候选阶段变更、新引用、推进训练及科学处置均为 0。收据部分用量窗口起止、调用数及输入/缓存/输出/推理 token 在写入时未知，finish 后核实。缓存含于输入、推理含于输出，不重复相加、不累计历史窗口、不代表账单。
- 执行状态在写入时为持租约待收尾，仅在 finish 返回校验通过与 COMPLETE 且独立确认锁释放后称本轮完成；失败保留证据并 abort 核实释放，不在释放后写项目。

## 5. Decisions Made
- 目标 0.515 尚未达到，自动 CSV 舍入 champion 仍为 Y36=0.5110、差距 0.004，不作科学晋升；冻结比较器 W2 保留，Phase 2 不启动。
- 授权两卡各一个实验、容量目标已满、预算 0，无到期或终态研究动作，本轮正常事件收尾，没有新科学决策。不凑变体、重复无信息增益测试、等待耗尽截止或放宽准入，不操作其他项目、目标、调度或后台。
- Y39 日志最佳由上轮 0.4954 升至 0.5030，仅是中途训练状态，不证明相对基线显著获益、效用、必要性、泛化、无害或 Pareto；Y37 同样按原固定 30 epoch 继续。终态仍使用 9508 表达/141 场景，各项与 W2/B1 各两次、FIRST 为主、场景 bootstrap 10000/seed1999、TRAIN 边界 1487/3325和原伤害/重复噪声/配对成本门槛。
- 无需用户操作，后续仅据新 canonical 快照实际终态、故障或到期动作处理；真实下次调度时间未知，不新建任务、不嵌套 codex exec、不恢复后台。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
