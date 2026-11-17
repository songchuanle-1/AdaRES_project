# Research Pipeline Report - AdaRES_project
快照时间：2026-10-20T20:43:05.148690+00:00；生成报告时快照年龄：61 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-21 04:43:04
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
================================================== 2026-10-21 04:42 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y56_unique_competitors_res] Best=0.4695 8/30 no_improve=1/30 gpu=0 ? 序列: 0.4378, 0.4503, 0.4569, 0.4695, 0.4647
  [adares_y57_peer_second_moment_res] Best=0.4646 8/30 no_improve=0/30 gpu=1 ? 序列: 0.4176, 0.4316, 0.4570, 0.4493, 0.4646

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y56 | configs_song/model_paper5/adares_y56_unique_competitors_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y57 | configs_song/model_paper5/adares_y57_peer_second_moment_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 111：complete=16 / rejected=79 / closed=12 / training=2 / blocked=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 35（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 仅AdaRES_project，owner、实验优先授权及server-2 GPU0/1单卡单实验限制核验；后台仍Disabled。begin先监控、检查目标及尝试派发既有合格任务，队列无waiting/claiming可派发；无新终态或操作故障。
- 实际执行python -B scripts/check_ef_execution.py --token 38e4ac5762484ebbb03c97b99dda5c93。Y58/Y59各7文件的源码/YAML、冻结smoke payload、Git测试提交a83449f、current goal和原生训练源仍一致；过期goal/篡改源码负控制拒绝。仅复查执行前完整性与资源，不重复CPU构造、不声称原生后向通过。
- 最新采集上海2026-10-21 04:43:24：Y56原PID83638/GPU0第8/30轮step1618至1709，best mIoU .46950018405914307（最近验证.4646662175655365）；Y57原PID95947/GPU1第8/30轮step1075至1169，best .46456223726272583。各7次注册原生验证，实际cwd/config及原生递增步数核实；仍是训练中间结果，不按波动调参或改fixed30。
- 实际GPU/全部compute与running/claiming槽位检查无授权空整卡。两个模拟生命周期控制不是实际资源释放；保持原训练，不借剩余显存共跑。Y58/Y59仍等两步native smoke，未准入、未入队或启动新训练，queue保持2running/0waiting/0claiming。
- 本轮没有新增变体、候选、代码修改、提交、失败或修订；仅保存当前真实resource_check/blocked证据与七段记忆。未跨项目、未改调度/goal/数据协议/授权边界，不重复请求常规许可。报告尚待canonical finish，不预称其结果。

## 5. Decisions Made
- 本轮无新研究决策，保持已有独立Y58/Y59、其原生YAML及精确测试版本，等待整卡资源，不为配额制造重复候选或更改运行训练。
- 授权整卡实际空闲并核验当前目标/版本/租约后，继续既有两步真实TRAIN native smoke，通过即当轮入队并尝试现有门控派发；无需额外常规授权，无全局研究冷却。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
