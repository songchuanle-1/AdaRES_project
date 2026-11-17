# Research Pipeline Report - AdaRES_project
快照时间：2026-10-01T22:49:00.532007+00:00；生成报告时快照年龄：155 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-02 06:49:00
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
================================================== 2026-10-02 06:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y18_point_measure_res] Best=0.4882 10/30 no_improve=1/30 gpu=1 ? 序列: 0.4617, 0.4678, 0.4670, 0.4882, 0.4874
  [adares_y15_query_completion_res] 进程未见 (deathwatch 2/3) — 暂不处置

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y18 | configs_song/model_paper5/adares_y18_point_measure_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y15 | configs_song/model_paper5/adares_y15_query_completion_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 快照2026-10-02 06:49:00+08，event；核实实际线程与foreground归属一致、closure已启用、Windows后台Disabled、server-2 GPU0/1限制未变。begin监控和目标检查仅各一次，返回码均0。
- 本轮无新终态、到期候选或准入预算。Y15 deathwatch从1/3到2/3，队列仍running；保留上一轮rejected科学结论，不重复闭合路线验证、不强制终态。
- 06:49:33至06:49:36+08只读核实server-2主机身份及Y18原PID7449；原唯一日志Train10/30步数3654至3658，lr5.8e-05。两次1200byte原日志尾部SHA256已保存，不仅凭PID认定训练正常。
- 后续实测GPU0 1MiB/util0%、GPU1 22720MiB/util81%，Y15原PID18796未见。设备显存不是独立实验峰值；无GPU诊断、重训、kill、新准入、其他项目或调度操作。
- 新增进度证据并哈希绑定Y18，保持training，更新七段memory；科学状态Y15 rejected/Y18 training，最终效果未知。停止本轮新增研究的依据是快照活跃2/目标2、预算0、due0，不声称路线耗尽或资源全部占满。
- 执行状态在报告生成时待finish；以实际finish/锁核验为准，失败保留证据并abort。收据用量窗口与产出在finish后于任务补充，当前未知，不把缓存重复加总或当作账单。

## 5. Decisions Made
- 无新增科学决策；继续原Y18固定30epoch训练及既有冻结终态评价，下一次正常monitor继续Y15生命周期确认，不重复preflight加速deathwatch。
- 本轮为例行训练进度与既有证据收尾，无需用户操作；不因空闲GPU凑变体，不放宽准入或验证集调参。下一次调度时间未读取，记未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
