# Research Pipeline Report - AdaRES_project
快照时间：2026-10-05T13:49:31.148024+00:00；生成报告时快照年龄：186 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-05 21:49:30
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
================================================== 2026-10-05 21:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y28_mask_readback_res] Best=0.3820 3/30 no_improve=0/30 gpu=1 ? 序列: 0.3474, 0.3820

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y28 | configs_song/model_paper5/adares_y28_mask_readback_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- AdaRES_project本轮event快照2026-10-05 21:49:31.148024+08，截止22:49:31.148024；归属一致，Pipeline-AdaRES_project仍Disabled，closure_policy启用。begin执行监控和目标检查各一次；活跃1/运行1/等待0/目标1/准入预算0/到期0，无新增终态。
- 复用本轮规范监控事实：Y28日志当前3/30、Best=0.3820、no_improve=0/30，已记录序列0.3474、0.3820。3/30是当前轮标记，不表示固定30轮完成；早期验证值不是泛化收益、冠军提升或训练后全量冻结评估结论。
- 没有新增终态、故障、待核验启动或到期研究候选；不重复preflight、远程健康抓取、已闭合诊断或新变体测试。上轮20:50的步数3341至3350及GPU1设备17500MiB/35%为历史观测，本轮显存、利用率、计算PID、Acc50/spmIoU未知，不当作当前占用或零值。GPU0未查询或使用。
- 候选AdaRES-Y28仍为training、证据28条，本轮不改账本、准入收据、既有证据、源码或冻结协议；仅刷新七段记忆中的本轮预检事实和上轮真实收尾，保持119行。0至3段直接复用脚本表格，原表缺失字段保留未知。
- 写报告时本轮执行状态为待postflight，科学状态training/效应未知；收据窗口、用量、最终产出尚未知。仅finish明确PASS/COMPLETE并核验实际headless租约释放后称完成。收据只是begin至记录时的局部可见窗口，缓存含于输入、推理含于输出，不重复相加，不当账单。

## 5. Decisions Made
- 没有新的科学决策。唯一授权GPU1仍由Y28训练占用，准入预算0；保持同一固定30轮、seed1999、min_epochs30/patience30，不额外排队、停训、重新启动或回放旧Y23恢复。
- 目标0.515未达，历史W2冠军0.5100不变；不依据早期验证曲线调轮数、阈值、温度、损失权重、特征或选择子组。训练结束后按既有启动前冻结方案进行全量重复、完整场景bootstrap、分组损害和训练后成本评估。
- 当前只是正常训练中的例行观察，没有需用户处理的实质变化，heartbeat保持安静；完整六段报告保存于本轮报告路径。不操作其他项目、调度或后台，不新建任务、不嵌套codex exec。下一准确调度时间未读取，记未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
