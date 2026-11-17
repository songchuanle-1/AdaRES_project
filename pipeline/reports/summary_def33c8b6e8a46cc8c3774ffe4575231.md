# Research Pipeline Report - AdaRES_project
快照时间：2026-10-10T08:49:20.734407+00:00；生成报告时快照年龄：68 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-10 16:49:20
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
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-10 16:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y37_pair_distribution_res] Best=0.4615 6/30 no_improve=0/30 gpu=0 ? 序列: 0.3360, 0.3686, 0.4222, 0.4472, 0.4615
  [adares_y36_local_adjacency_res] Best=0.5099 29/30 no_improve=7/30 gpu=1 ? 序列: 0.5028, 0.5039, 0.5070, 0.5064, 0.5061

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y36 | configs_song/model_paper5/adares_y36_local_adjacency_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y37 | configs_song/model_paper5/adares_y37_pair_distribution_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 只操作 AdaRES_project。已一次性核对实际任务归属、最新配置和项目授权：foreground owner 一致、server-2 GPU0/1 每卡单任务、closure_policy 启用、Windows 调度 Disabled。未操作其他项目、调度或后台，未读取密码或使用嵌套 codex。
- begin --auto-steady 获取本轮租约 def33c8b6e8a46cc8c3774ffe4575231，首先读精简视图；快照为 2026-10-10 16:49:20.734407+08:00，截止为 17:49:20.734407+08:00。event 仅 research_artifact_changed，终态及到期候选均 0；监控和目标检查已执行一次，无重复 preflight 或历史倾倒。
- 保留第 0-3 段脚本事实表。原始日志 Y36/GPU1 29/30、舍入 best 0.5099、未改善 7/30；Y37/GPU0 6/30、舍入 best 0.4615。结构化缺项及实时 GPU 利用率、显存和进程仍未知。29/30 不等于已完成，早期指标增长不等于有效性；未等待整场训练或重复健康测试。
- 准入目标已满足，无待处理终态、故障或到期诊断；仅在租约内更新七段记忆及报告第 4/5 段。没有新增候选、测试、复核、证据引用、准入、派发或无变化决策日志；没有改动两个原始 fixed30 训练。
- 科学状态：Y36/Y37 training，W2 champion，Phase 1 未达标，Y35 complete_inconclusive。执行状态：写入时待 finish；实际 CLOSURE_POSTFLIGHT_PASS、COMPLETE 和独立确认锁消失后才称 COMPLETED。失败则保存证据、abort 并核实释放。
- 本轮收据窗口和实际产出写入时未知，finish 后只读核对并在回复补充。用量仅为 begin 至收据的局部日志窗口，排除早期读取、晚到记录和最终回复；缓存已含于输入、推理已含于输出，不重复相加、不当账单。产出计数不是科学收益，释放后不改写项目文件。

## 5. Decisions Made
- 没有新科学决策。继续保留既定 30 epoch 训练，不据未改善次数或训练中 best 调参、提前停止、重启或声称达标；没有新增容量，不设计凑数变体或重开已关闭工作。
- 终态出现后按冻结协议继续 candidate/W2/B1 双次全验证、场景 bootstrap 与子组风险、重复噪声及训练后成本核验。没有新的阻塞或经测试支持的穷尽证据，不伪造 closure/cooldown；下一次准确调度时间未知，自动化未修改。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
