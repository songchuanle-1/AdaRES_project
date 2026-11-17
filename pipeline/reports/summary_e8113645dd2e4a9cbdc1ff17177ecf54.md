# Research Pipeline Report - AdaRES_project
快照时间：2026-09-30T21:48:42.061713+00:00；生成报告时快照年龄：132 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-01 05:48:41
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
================================================== 2026-10-01 05:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y17_target_equivalent_pairing_res] Best=0.4615 10/30 no_improve=1/30 gpu=1 ? 序列: 0.4360, 0.4425, 0.4538, 0.4615, 0.4566
  [adares_y15_query_completion_res] Best=0.1734 10/30 no_improve=3/30 gpu=0 ? 序列: 0.1714, 0.1734, 0.1683, 0.1684, 0.1599

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
- 已核对实际任务ID与execution_mode归属，授权仍仅server-2 GPU0/1，每卡一任务；Windows后台调度Disabled，目标及冻结条件未改动。
- 本轮event快照2026-10-01 05:48:42+08，begin执行一次监控和目标检查；运行2/容量2、等待0、到期候选0、准入预算0，无新增终态；没有重复preflight。
- 05:49:04+08只读核验：Y15 GPU0原PID18796，Train[10/30]379→383/4686；Y17 GPU1原PID109632，Train[10/30]2483→2486/4686。显存16728/22784MiB、利用率83%/38%；两个原进程均有步数推进，有限日志样本未见崩溃或OOM。
- 本轮新证据保存为pipeline/analyses/native_progress_e8113645dd2e4a9cbdc1ff17177ecf54.md，分别追加Y15/Y17哈希引用，更新七节研究记忆；未覆盖历史证据或新增训练。
- 科学状态仍training，无新科学测试或准入。Y15早期best0.1734、最新0.1599、no-improve3/30；Y17早期best0.4615、最新0.4566、no-improve1/30。保留不利观测，不作30epoch效果判断或验证集调参。Acc25/50、spmIoU、checkpoint内容和最终效果本轮未知。
- 执行状态待finish及独立锁释放检查确认。收据用量当前未知，仅begin至收据的局部窗口，不含前置读取、晚到记录和最终回复；缓存是输入子集，推理是输出子集，不重复累加，不当作账单。实际产出为一次双任务核验、一份新证据及两项引用，无新增变体、派发或科学收益结论。

## 5. Decisions Made
- 无新科学决策或用户操作需求，保留两个原单卡任务和固定30epoch协议；不修改阈值、温度、损失权重、特征或epoch，不以早期下降作停训依据。
- 真实故障或终态出现后按冻结全ScanRefer评估、场景聚类、噪声、损害/Pareto方案处理；Y17因果归因仍需相同重组数据的pair-off对照。
- 本轮执行完成不等于方法有效或目标达成；完成30epoch的冠军仍W2 0.5100，距目标0.515差0.0050。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
