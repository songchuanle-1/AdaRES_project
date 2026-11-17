# Research Pipeline Report - AdaRES_project
快照时间：2026-09-30T16:49:14.484078+00:00；生成报告时快照年龄：183 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-01 00:49:14
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
================================================== 2026-10-01 00:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y17_target_equivalent_pairing_res] Best=0.3557 3/30 no_improve=0/30 gpu=1 ? 序列: 0.2778, 0.3557
  [adares_y15_query_completion_res] Best=0.1627 5/30 no_improve=0/30 gpu=0 ? 序列: 0.0760, 0.1239, 0.1221, 0.1627

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
- 本轮2026-10-01 00:49:14+08快照，event，deadline01:49:14；所有权/仅server-2GPU0/1/固定30epoch核实，旧Windows调度Disabled。begin执行一次monitor/goal，2运行0等待/预算0，无到期候选或终态。
- 00:49:51只读PID/GPU/原生日志核验：Y17GPU1PID109632 Train[3/30]2409->2412/4686，Y15GPU0PID18796 Train[5/30]3267->3270/4686。各单卡，未重启/改配置。显存16590/22782MiB，利用率67/29%。
- Monitor早期bestmIoU Y17=0.3557、Y15=0.1627，均无连续不刷新；非30epoch终态、未达到目标、不能推断因果收益或调参。Acc25/50、spmIoU及checkpoint内容未独立核验，未知。
- 保存本轮真实运行证据并追加两在训候选、更新七段短记忆。无新准入/训练推送/模型代码变更/研究决策，不重复诊断或凑实验。
- 执行结果待finish；科学状态两项training，后续固定终态评估。收据仅begin到receipt本地用量窗口，缓存输入不重复相加，非账单；实际产出本轮新进度证据，无新训练。

## 5. Decisions Made
- 无新科学决策。两任务正常进展且容量满、无到期研究，保留30epoch协议和冻结评估计划，监测故障/终态；无需用户操作。
- 保持已保存每小时:47前台任务，下一次2026-10-01 01:47北京时间；不恢复Windows旧后台、不操作其他项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
