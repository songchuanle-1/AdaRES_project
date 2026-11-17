# Research Pipeline Report - AdaRES_project
快照时间：2026-10-05T10:51:42.914927+00:00；生成报告时快照年龄：192 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-05 18:51:42
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
================================================== 2026-10-05 18:51 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y27_relative_peer_res] 进程未见 (deathwatch 2/3) — 暂不处置

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y27 | configs_song/model_paper5/adares_y27_relative_peer_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 本轮快照为2026-10-05 18:51:42.914927+08，event；实际任务ID与execution_mode归属一致，后台Pipeline-AdaRES_project保持Disabled，closure_policy已启用。begin仅执行一次，先读取compact view，再按ID核对AdaRES-Y27；没有重复preflight。
- 脚本监控未见训练进程，Y27规范队列仍为running/deathwatch 2/3；活跃1、准入预算0、due_count0、新终态ID0。这是终态确认窗口，不是物理训练仍在运行的证明。本轮未重新采集GPU1利用率、显存或compute进程，均未知；18:10的空闲证据属于上一轮，不能当作当前占用。
- 稳定候选AdaRES-Y27仍为complete，65项证据未改动。上一轮已完成固定30轮、Y27/W2/B1各两次9508表达/141场景评估、10000次整场景bootstrap和训练后串行成本测量；所有冻结分组区间跨零，效果仍不确定、目标未达标、无champion或Pareto提升。本轮没有重复任何已完成科学测试。
- 仅更新七段研究记忆的本轮快照、deathwatch 2/3和上一轮已确认的收尾事实，并复用脚本0-3段事实表生成完整六段报告；未改候选、队列、准入、模型、冻结协议或资源白名单。新候选0、新测试0、新证据0、新训练0。
- 写入此notes时本轮执行状态为持有租约、report/finish待执行，不能提前称COMPLETED。本轮收据用量窗口、token统计及程序实际产出计数尚未知；以finish后收据为准。用量只覆盖begin至收据的本地窗口，缓存包含在input、推理包含在output，不重复相加，也不是账单。

## 5. Decisions Made
- 本轮没有新科学决策。延续原有处置：不重启已完成Y27，不再恢复已完成的旧Y23，不将规范running标签误作当前训练，不为获得完成标记制造测试或修改历史证据。
- 无到期可执行候选且准入预算为0，保留正常deathwatch确认流程；不重复monitor以人为推进至3/3，不手动腾队列。等规范终态确认和容量开放后，在同一授权项目继续独立、可证伪的新设计。
- GPU0仍归项目外保留，未查询或使用；不操作其他项目、server-6、调度或后台，不放宽准入及冻结规则。下一次时间未读取已保存自动化，未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
