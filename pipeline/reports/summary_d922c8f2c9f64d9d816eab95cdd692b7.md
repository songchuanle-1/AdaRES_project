# Research Pipeline Report - AdaRES_project
快照时间：2026-10-10T14:48:42.459141+00:00；生成报告时快照年龄：84 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-10 22:48:42
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
================================================== 2026-10-10 22:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y39_point_distribution_res] Best=0.3858 3/30 no_improve=0/30 gpu=1 ? 序列: 0.3200, 0.3858
  [adares_y37_pair_distribution_res] Best=0.5018 13/30 no_improve=1/30 gpu=0 ? 序列: 0.4747, 0.4926, 0.4920, 0.5018, 0.4935

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
- 仅处理 AdaRES_project，实际任务 ID 已与前台配置核对；目标、冻结协议、GPU0/1 授权和已启用 closure_policy 不变，Windows Pipeline 为 Disabled。begin 仅一次，先读紧凑视图，不重复 preflight、不读取完整快照或候选历史。
- 快照为 2026-10-10 22:48:42.459141+08，真实截止为 23:48:42.459141+08。event 原因仅为 research_artifact_changed；运行 2、认领 0、等待 0、目标容量 2、准入预算 0，新终态 0、到期候选 0。
- 原始监控：Y37 在 GPU0 为 13/30，舍入最佳 mIoU 0.5018，无改善 1/30；Y39 在 GPU1 为 3/30，舍入最佳 0.3858，无改善 0/30。两项仍为 training，训练后有效性未知；结构化表未采集的指标、当前 PID、利用率及显存仍未知，不推断空闲或用历史证据冒充实时核实。
- 保持两项原有固定 30 轮训练，不新增候选、准入、启动或重复测试，不改写历史科学证据。更新七段短记忆，仅补报告第 4/5 段并保留脚本第 0 至 3 段事实表。执行状态尚待 finish；成功并独立核实锁释放后才称 COMPLETED，失败则保留证据、按真实原因 abort 并核实释放。
- 收据仅统计本轮 begin 到收据记录的本地窗口，排除更早读取、迟到用量和最终回复；目前用量及实际产出统计未知，finish 后读取真实收据。本轮实际工作为监控确认、短记忆和六段报告，无新增实验。缓存属于输入，推理属于输出，不重复相加、不累计前轮、不视为账单；释放后不修改项目报告。

## 5. Decisions Made
- 无新增科学决策。目标容量已达到 2/2 且没有新终态或到期工作，按 event 合同收尾，不硬撑截止、不造测试或冷却理由。训练中监控数值不能替代冻结的训练后全量比较、子组伤害与成本审查。
- 目标 0.515 尚未达到；Y36 自动数值冠军舍入为 0.5110，不构成科学晋级，保留 W2 科学比较基准和 Phase 2 未授权状态。保留 Y37/Y39 的既定后续评估，真实终态、崩溃或到期工作出现后再执行。
- 不操作其他项目或调度，不新建任务、不恢复后台、不嵌套 codex exec。下一调度时间未知，无需用户操作。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
