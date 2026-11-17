# Research Pipeline Report - AdaRES_project
快照时间：2026-10-01T02:49:26.996266+00:00；生成报告时快照年龄：131 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-01 10:49:26
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
================================================== 2026-10-01 10:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y17_target_equivalent_pairing_res] Best=0.4923 17/30 no_improve=1/30 gpu=1 ? 序列: 0.4810, 0.4731, 0.4833, 0.4923, 0.4910
  [adares_y15_query_completion_res] Best=0.2054 14/30 no_improve=0/30 gpu=0 ? 序列: 0.1599, 0.1890, 0.1971, 0.1959, 0.2054

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
- 当前任务实际ID与execution_mode归属一致；授权仅server-2 GPU0/1，每卡一任务，Windows后台Disabled；目标、冻结条件及调度未修改。
- 本轮event快照2026-10-01 10:49:26+08；begin执行一次监控/目标检查，容量2、运行2、等待0、到期候选0、准入预算0、终态0，未重复preflight。
- 10:49:53+08只读核验：Y15 GPU0原PID18796，Train[14/30]2416→2419/4686；Y17 GPU1原PID109632，Train[17/30]2751→2754/4686。显存16730/22784MiB、利用率86%/79%；两项原生训练持续推进，有限样本未见崩溃/OOM。
- 保存独立证据pipeline/analyses/native_progress_a92b9b9d745f433e860731fec4ecb8c2.md，分别追加Y15/Y17哈希引用并更新七节记忆；历史证据未覆盖，没有新科学测试、准入、派发或重启。
- 科学状态仍training；早期best Y15=0.2054、Y17=0.4923，Y17最新0.4910、no-improve1/30。保留不利观测，不作最终效果或达标判断；Acc25/50、spmIoU、checkpoint内容及最终30epoch有效性本轮未知。
- 执行状态待finish及独立锁释放核验。收据用量当前未知，窗口仅begin至收据，不含前置读取、晚到记录和最终回复；缓存属于输入、推理属于输出，不重复累加或当作账单。实际产出一次双任务核验、一份新证据及两项引用，无新增训练。

## 5. Decisions Made
- 无新科学决策或用户操作需求，保留两项原单卡训练及固定30epoch协议，不调整阈值、温度、权重、特征或epoch。
- 真实故障或终态后按冻结全ScanRefer评估、场景聚类、噪声、损害/Pareto方案处理；Y17因果归因仍需相同重组数据pair-off对照。
- 本轮执行闭环不代表科学有效性，完成30epoch的冠军仍W2 0.5100，距目标0.515差0.0050。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
