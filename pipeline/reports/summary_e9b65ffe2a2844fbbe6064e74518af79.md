# Research Pipeline Report - AdaRES_project
快照时间：2026-10-01T20:48:02.623783+00:00；生成报告时快照年龄：146 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-02 04:48:02
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
================================================== 2026-10-02 04:47 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y18_point_measure_res] Best=0.4678 7/30 no_improve=0/30 gpu=1 ? 序列: 0.3864, 0.4419, 0.4372, 0.4617, 0.4678
  [adares_y15_query_completion_res] Best=0.2303 30/30 no_improve=2/30 gpu=0 ? 序列: 0.2116, 0.2249, 0.2303, 0.2282, 0.1977

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
- 本轮event；快照2026-10-01T20:48:02.623783Z，即北京时间2026-10-02 04:48:02。触发项是上轮研究产物/证据变化；复用begin事实表，不重复preflight。新增终态0、到期诊断0、准入预算0。
- 北京时间2026-10-02 04:48:31.6156609至04:48:31.8685035只读SSH核实server-2主机dl01-MS-7C60、原PID与原唯一日志。GPU0显存17282MiB/利用率43%，GPU1显存22718MiB/利用率82%；两卡均有原训练进程，设备总显存不是候选独立峰值。
- Y15原PID18796 Train[30/30]2571→2574/4686，Y18原PID7449 Train[7/30]4607→4610/4686。有限900字节日志尾未见崩溃标记，不宣称全日志健康。Y15进入第30轮但尚未完成全部步数，不能把30/30标签当终态。保存新增共享进度证据，以SHA256绑定两稳定候选，更新七段记忆；未重启或修改训练。
- 科学状态training：监控Y15 best=0.2303/latest=0.1977，Y18 best/latest=0.4678；训练后效力、分组危害、目标达成与准确率/成本前沿仍未知。执行状态等待finish/postflight，尚不能称COMPLETED。
- 实际产出为一份共享训练进度证据、两候选新增引用、报告与记忆更新；新增候选/准入/训练启动均为0。finish前收据用量未知；统计窗口仅覆盖begin后至收据生成前本地可见响应，不含之前阅读、延迟记录或最终答复。缓存包含在输入、推理包含在输出，不重复相加，不作为账单。

## 5. Decisions Made
- 活跃2/目标2且无到期诊断，继续Y15/Y18原进程固定30epoch/seed1999，等待真实终态而非本轮等待整个训练；不额外准入，不据中间验证读数调整结构、权重、阈值或温度。
- 终态后按各自冻结计划执行官方全量ScanRefer/W2/B1比较、重复噪声、场景聚类区间、预注册分组危害及准确率/成本前沿评估。Y15中间指标较低是待终态核验的科学风险，不是已证实的运行故障，不盲目重启或抹除不利证据。
- 仅AdaRES_project与server-2 GPU0/1；Windows后台保持Disabled，不操作其他项目/调度/server-6。finish成功并独立核实锁消失才称本轮完成；失败保留证据并用原token abort。下一调度时刻未读取，记未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
