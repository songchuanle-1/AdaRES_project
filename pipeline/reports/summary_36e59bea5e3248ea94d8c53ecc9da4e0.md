# Research Pipeline Report - AdaRES_project
快照时间：2026-10-11T03:48:30.388058+00:00；生成报告时快照年龄：80 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-11 11:48:29
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
================================================== 2026-10-11 11:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y39_point_distribution_res] Best=0.5069 21/30 no_improve=0/30 gpu=1 ? 序列: 0.5052, 0.5057, 0.5049, 0.5063, 0.5069
  [adares_y37_pair_distribution_res] Best=0.5052 29/30 no_improve=9/30 gpu=0 ? 序列: 0.5024, 0.4999, 0.5044, 0.4994, 0.5023

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
- 已核实实际环境任务ID与execution_mode的绑定一致；仅对AdaRES_project执行一次canonical begin --auto-steady，首先读取compact model view。快照为2026-10-11 11:48:30.388058+08，deadline为12:48:30.388058+08；monitor_rc=0、goal_rc=0，未重复preflight。
- 复用本轮脚本事实表。begin原始监控显示Y37在GPU0训练29/30，日志四舍五入最佳mIoU 0.5052、no_improve=9/30；Y39在GPU1训练21/30，日志最佳0.5069、no_improve=0/30。两者科学状态仍为TRAINING/training_not_efficacy；结构化表内指标、当前PID、GPU利用率和显存仍未知，不以原始日志或历史进程值填充。
- compact view唯一事件原因为research_artifact_changed，活跃2、运行2、目标2、准入预算0、到期候选0、新终态0。没有可恢复的到期诊断或新崩溃/终态事件，不重复已关闭实验评审、不执行等价探针、不新增训练或候选。
- 补充只读查询的gpu.md、active.md、queue.md路径不存在，未作为事实或故障判断依据；该查询没有项目写入。当前事实均取自本轮canonical begin及其报告。
- 已在本轮租约内更新七段研究记忆和动作/决策notes，保留历史未知值与冻结证据。当前执行状态为等待finish校验，不能预称COMPLETED；完成与释放必须以实际CLOSURE_POSTFLIGHT_PASS/COMPLETE及独立锁检查为准。
- 本轮预期实际研究产出为新增候选0、新证据引用0、新准入0、新训练0，最终以收据为准。本轮收据用量窗口及调用/token计数在finish前未知；finish后仅在任务回复补充，不在释放租约后改写项目报告。缓存计入输入、推理计入输出，不重复相加，不视作账单。

## 5. Decisions Made
- 准入目标已满足且无到期候选，因此按真实停止条件收尾本轮event；不是路线耗尽或新的资源故障，不建立空转决策日志或无证据冷却，不等待训练全部结束来填满deadline。
- 保留Y37和Y39原训练、源代码与checkpoint，继续固定30 epochs/seed1999，不重启、不早停、不进行验证集调参，也不额外占用GPU。GPU0授权不等于卡空闲，后续诊断或dispatch仍需新鲜占用与租约核验。
- 当前目标0.515未达，自动CSV数值champion Y36 0.5110并不构成科学晋升；冻结比较对象W2保留，Phase2仍未授权。Y39日志最佳提高至0.5069不证明训练终态、因果必要性、无伤害或Pareto优势。
- 在后续canonical终态或实际崩溃事件出现时处理；终态后按已冻结9508表达式/141场景、候选/W2/B1各两次、FIRST主结果、场景bootstrap10000/seed1999、TRAIN切点1487/3325、伤害门控、重复噪声与训练后成本评审。不得把未执行评估记为PASS。
- 仅操作AdaRES_project，保持Windows后台调度Disabled，不操作其他项目或调度、不新建任务、不嵌套codex exec、不恢复后台。下次触发时间未知，以既有自动化为准。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
