# Research Pipeline Report - AdaRES_project
快照时间：2026-09-30T19:49:56.502432+00:00；生成报告时快照年龄：296 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-01 03:49:56
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
================================================== 2026-10-01 03:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y17_target_equivalent_pairing_res] Best=0.4425 7/30 no_improve=0/30 gpu=1 ? 序列: 0.3557, 0.3967, 0.4119, 0.4360, 0.4425
  [adares_y15_query_completion_res] Best=0.1734 8/30 no_improve=1/30 gpu=0 ? 序列: 0.1221, 0.1627, 0.1714, 0.1734, 0.1683

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y17 | configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y15 | configs_song/model_paper5/adares_y15_query_completion_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 核对实际当前任务ID与execution_mode归属一致，授权仅server-2 GPU0/1，每卡一任务；Windows后台调度仍Disabled。未修改目标、冻结协议或调度。
- 本轮event快照2026-10-01 03:49:56+08，begin已执行一次监控/目标检查。活跃2、运行2、排队0、到期候选0、准入预算0，未重复preflight。
- 03:51:08+08只读核验：Y15 GPU0原PID18796，Train[8/30]1670→1673/4686；Y17 GPU1原PID109632，Train[7/30]3640→3643/4686。显存16728/22782MiB、利用率56%/28%；有限日志样本未见OOM或崩溃，两项原生训练有实际推进。
- 保存本轮独立证据pipeline/analyses/native_progress_4b1bd39ab5ec4c49a906faf7677fdcc4.md并分别追加Y15/Y17哈希引用，更新七节记忆；历史证据未覆盖。没有新设计、科学测试、准入、启动或重启。
- 科学状态仍training。早期best Y15=0.1734、Y17=0.4425，Y15最新0.1683、no-improve1/30，不据此调参/早停或宣称最终效果。Acc25/50、spmIoU、checkpoint内容及最终30epoch效果本轮未知。
- 本轮执行状态待finish与独立锁检查确认；收据用量当前未知，窗口仅begin至收据，不含前置读取、晚到记录及最终回复；缓存包含在输入、推理包含在输出，不重复累加，不当作账单。实际产出为一次双任务运行核验、一份新证据及两项引用，无新增训练。

## 5. Decisions Made
- 无新科学决策或用户操作需求。保留两项固定30epoch原单卡任务，不以早期指标波动改变阈值、温度、权重、特征或epoch协议。
- 继续等待实际终态或故障事件，再按已冻结全ScanRefer评估、场景聚类、噪声、损害/Pareto方案处理；Y17机制归因仍需完全相同数据重组的pair-off对照。
- 本轮闭环成功不代表方法有效或达到mIoU0.515；冠军仍是完成30epoch的W2 0.5100。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
