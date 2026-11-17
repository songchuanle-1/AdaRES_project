# Research Pipeline Report - AdaRES_project
快照时间：2026-10-10T20:49:12.102098+00:00；生成报告时快照年龄：74 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-11 04:49:11
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
================================================== 2026-10-11 04:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y39_point_distribution_res] Best=0.4933 11/30 no_improve=1/30 gpu=1 ? 序列: 0.4728, 0.4655, 0.4797, 0.4933, 0.4919
  [adares_y37_pair_distribution_res] Best=0.5052 20/30 no_improve=0/30 gpu=0 ? 序列: 0.5025, 0.5012, 0.5012, 0.5008, 0.5052

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
- 本轮 token=bfb0c5012da74f3fbe8c9a589ce13f1f。已核实实际任务 ID 与 foreground owner 同为 01a07e32-78bf-7670-925b-370911ea48db，执行模式为 foreground，闭环策略已启用，Windows Pipeline-AdaRES_project 为 Disabled。按最新目标、项目指令和 server-2 GPU0/1 授权，仅操作 AdaRES_project。
- begin --auto-steady 仅执行一次，首先读取紧凑 .tick_model_view.json。快照时间为 2026-10-11 04:49:12.102098+08，实际截止为 05:49:12.102098+08，模式 event，原因 research_artifact_changed；monitor_rc=0、goal_rc=0，不重复 preflight。
- 复用脚本事实表：活跃/运行 2，认领/等待 0，目标容量 2，准入预算 0，到期候选 0，新增终态 0。无本轮需处理的候选 ID，不倾倒完整快照、历史、引用或模板。
- 本轮原始监控日志：Y37 为 20/30、日志舍入最佳 mIoU=0.5052、no_improve=0/30；Y39 为 11/30、日志舍入最佳 mIoU=0.4933、no_improve=1/30。结构化实验表指标、实时 PID、GPU 利用率和显存仍未知，保持脚本表格原值，不用历史占用推断空闲。
- 两项仍处科学 TRAINING/training_not_efficacy 阶段，未列出终态或故障。Y39 本轮无指标对齐告警，保留 02:48:12 的历史不可用记录及原因未知；不重复解析器检查或旧候选评审，不重启、提前停止、调参或额外探测 GPU。
- 持租约更新七段短记忆和六段报告，仅补充本轮事实、动作及停止条件。实际新增候选、候选阶段变更、新引用、推进训练和科学处置均为 0；本轮未新增实验或测试，没有列出的到期或终态研究动作，现有训练正常保留。
- 本轮收据部分用量窗口起止、调用数及输入/缓存/输出/推理 token 在写入阶段未知，finish 后读取实际收据核实。缓存包含于输入、推理包含于输出，不重复相加、不累计此前窗口、不作为账单；执行状态此时为持租约待收尾，只有 finish 校验通过并独立确认锁释放后才称完成，失败则保留证据并 abort。

## 5. Decisions Made
- 目标 0.515 尚未达到，自动 CSV 舍入 champion 为 Y36=0.5110、差距 0.004，不作科学晋升；冻结科学比较器继续保留 W2，Phase 2 未启动。
- 两张授权卡各有一个固定 30 epoch 实验，容量目标已满足，预算为 0，且没有到期或终态研究工作。本轮正常事件收尾，不凑候选、重复无信息增益测试或等待耗尽截止时间，不改调度、目标或准入门槛。
- Y37 日志最佳由上轮 0.5025 升至 0.5052 仅代表中途训练状态，不证明训练完成、效用、必要性、泛化、无害或资源 Pareto。Y39 同样保留原固定协议，不按中途验证曲线调整阈值、温度、损失权重、特征子集或 epoch。
- 后续仅按新 canonical 快照的实际终态、故障或到期动作处理；终态比较仍为完整 9508 表达/141 场景，各项与 W2/B1 各两次、FIRST 为主、场景 bootstrap 10000/seed1999、TRAIN 分组边界 1487/3325及原伤害/重复噪声/配对成本门槛。当前无需用户操作，真实下次调度时间未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
