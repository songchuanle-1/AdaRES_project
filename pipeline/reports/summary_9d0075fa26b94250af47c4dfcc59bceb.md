# Research Pipeline Report - AdaRES_project
快照时间：2026-10-11T08:48:39.547873+00:00；生成报告时快照年龄：152 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-11 16:48:39
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
================================================== 2026-10-11 16:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y39_point_distribution_res] Best=0.5078 28/30 no_improve=4/30 gpu=1 ? 序列: 0.5078, 0.5021, 0.5045, 0.5057, 0.5021
  [adares_y41_boundary_probability_res] 1/30 尚无验证 - gpu=0

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y39 | configs_song/model_paper5/adares_y39_point_distribution_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y41 | configs_song/model_paper5/adares_y41_boundary_probability_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 本轮仅核查 AdaRES_project，当前任务由环境 CODEX_THREAD_ID 核实，与 foreground 归属一致；最新 GPU0/GPU1 授权、冻结 goal、closure_policy 启用及 Windows Pipeline-AdaRES_project 禁用已确认。规范 begin 一次，monitor/goal 返回码均为 0，先读取小于 12KB 的 compact view，未重复 preflight。
- 快照为 2026-10-11 16:48:39.547873+08，模式 event；active_queue_changed/research_artifact_changed/research_work_changed 是上轮 Y41 启动及 Y37/Y40 科学处置的状态对齐，不是本轮新增终态或研究发现。原始监控 Y39/GPU1=28/30、best=0.5078、no_improve=4/30；Y41/GPU0=1/30、尚无验证。规范表格缺失的利用率、显存、进程及结构化指标仍记未知，不推断空闲。
- 活跃/运行均为 2，认领/等待均为 0，目标 2，准入预算 0；新终态 0、到期候选 0、遗漏到期项 0。没有待补启动证据、故障或可续做项，保留现有训练，不新增诊断/实验/准入、不重启或移动原训练。未重复读取候选证据或复测已关闭 Y37/Y40。
- 复用上轮已验证的 Y41 六门准入、GPU0 原生步数增长及 Y39/GPU1 保留证据；不把 16:34:17 的 PID/显存采样冒充本轮实时占用。科学状态保持 Y39/Y41 training、Y37 complete_inconclusive、Y40 rejected；目标未达，训练启动不证明有效性。
- 七段研究记忆更新了当前快照、原始训练进度及上轮真实 FINALIZED/PASS 收据，不改写旧评测或历史科学证据。六段事实表由脚本复用，只补本段动作及第5段无新决策的说明；没有另写 no-op 决策日志。
- 本轮报告写入时执行状态为待 postflight，只有 finish 成功并独立核实匹配锁释放后才称 COMPLETED。本轮收据用量窗口/产出计数尚未知；不得挪用上轮 111 条响应的统计，不将缓存再次相加或当作账单。

## 5. Decisions Made
- 本轮无新科学或资源决策。延续原计划：保持 GPU0 的 Y41 和 GPU1 的 Y39 固定30轮原生训练，待真实终态/故障事件后执行冻结比较；不为填充时间重复诊断或设计变体。
- W2 科学比较基准、目标 0.515、Phase2 未批准及所有准入约束均保持。没有其他项目操作、调度修改、新任务、嵌套 codex 或恢复后台；下一调度时间未知。
- 由于状态仍为正常训练且无需要用户处理的新事件，常规轮询不发提醒；完整0–5六段和规定表格保存在本轮报告，收尾结果以实际 finish、收据与锁核验为准。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
