# Research Pipeline Report - AdaRES_project
快照时间：2026-10-05T16:49:34.838905+00:00；生成报告时快照年龄：165 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-06 00:49:34
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
================================================== 2026-10-06 00:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y28_mask_readback_res] Best=0.4684 7/30 no_improve=0/30 gpu=1 ? 序列: 0.3820, 0.4316, 0.4525, 0.4606, 0.4684

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
- AdaRES_project本轮event快照2026-10-06 00:49:34.838905+08，截止2026-10-06 01:49:34.838905+08；归属一致，Pipeline-AdaRES_project仍Disabled，closure_policy启用。begin执行监控/目标检查各一次：容量1/目标1/活跃1/运行1/等待0/准入预算0/到期0/新增终态0。
- 模式原因仅research_artifact_changed。复用本轮规范监控：Y28当前7/30、日志Best=0.4684、no_improve=0/30，最新五项序列0.3820、0.4316、0.4525、0.4606、0.4684。7/30是当前轮标记，不是固定30轮已完成；早期验证指标不证明最终收益、冻结全量评估或冠军晋升。
- 无新增终态、故障、待核验启动或到期研究候选；不重复preflight、额外远程健康抓取、已闭合诊断或新变体测试。当前GPU1利用率/显存/计算PID、Acc50/spmIoU未知；2026-10-05 20:50设备17500MiB/35%及3341至3350步仅为历史观测，不冒充当前状态。GPU0未查询或使用。
- 只刷新119行七段操作记忆中的快照、当前运行事实和上轮真实收尾，跨午夜使用完整日期；候选账本、源码、准入收据、既有证据及冻结协议不修改。0至3段复用脚本完整事实表，缺失字段保留未知；本轮无新增研究候选、证据引用或科学阶段转换。
- 报告写入时执行状态待postflight，科学状态training/最终效应未知；本轮收据用量、窗口及最终产出尚未知。仅finish明确PASS/COMPLETE并核实实际headless租约释放后称完成，不在释放后写项目。收据仅begin至记录时的局部可见用量，缓存含于输入、推理含于输出，不重复相加，不当账单。

## 5. Decisions Made
- 无新的科学决策。当前为正常训练中的例行变化，无需用户操作，heartbeat保持安静；继续同一Y28固定30轮、seed1999、min_epochs30/patience30，不额外排队或准入，不停止或重启训练。
- 目标0.515未达，历史W2冠军0.5100不变。不依据早期验证曲线调轮数、阈值、温度、损失权重、特征或子组；训练完成后才按启动前冻结方案执行全量重复、完整场景bootstrap、分组损害及训练后成本。
- 仅操作AdaRES_project，不操作其他项目或调度，不新建任务、不嵌套codex exec、不恢复后台、不回放已履行的旧Y23恢复。GPU0维持用户保留状态。下一准确调度时间未读取，记未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
