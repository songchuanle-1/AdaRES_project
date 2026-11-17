# Research Pipeline Report - AdaRES_project
快照时间：2026-10-21T10:50:35.327339+00:00；生成报告时快照年龄：294 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-21 18:50:34
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
================================================== 2026-10-21 18:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y56_unique_competitors_res] Best=0.5080 23/30 no_improve=3/30 gpu=0 ? 序列: 0.5020, 0.5080, 0.5040, 0.5050, 0.5080
  [adares_y57_peer_second_moment_res] Best=0.5030 26/30 no_improve=5/30 gpu=1 ? 序列: 0.5030, 0.5010, 0.5020, 0.5000, 0.5020

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
- 候选累计记录 112：complete=16 / rejected=79 / closed=12 / training=2 / blocked=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 35（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 实际CODEX_THREAD_ID与配置owner一致，后台Pipeline-AdaRES_project保持Disabled；18:50:35新快照experiment_first启用，准入预算2，仅server-2 GPU0/1，一卡一独立实验。begin已检查监控、目标与既有队列，本轮无新终态或可派发waiting。
- 复用scripts/check_prepared_resources.py执行一次到期资源复查。Y58/Y59/Y60三项各7个已测源/YAML的冻结payload、当前工作区、Git blob/提交、goal及原生train入口LF SHA一致；既有原生检查程序未变。未重复CPU诊断或将身份检查当原生任务PASS。
- 18:50:59确认原Y56 PID13738/Train23of30 step4256至4333，原Y57 PID16317/Train26of30 step1670至1760；cmd/cwd匹配、原生步数增长、尾部无异常。当前续训片段日志best mIoU分别.5084453821/.5028846860；Y56较16:47刷新。训练未完成，这不是冻结双遍完整评估、显著收益或新冠军结论。
- 实际compute和running/claiming槽位仍占据两张授权整卡，无独占空卡。三项候选保留同ID资源证据、18:50起1小时复查与队列开槽提前唤醒条件，tick_attempt=Y58 resource_check/blocked。未执行GPU smoke、准入、入队、部署或共跑。
- 同次GPU0为280W/84C/风扇79%/util41%/22684of24576MiB，但SW Thermal Slowdown Active；GPU1为390W/66C/风扇52%/util71%/17098of24576MiB，SW Power Cap Active且无热降频。内核读取成功、无新Xid/fallen-off，boot未变。数据盘96%/可用155262074880bytes。本轮仅回读功耗等设置，未读取私密凭据或进行任何硬件设置变更。
- 全部本轮新观察见pipeline/checks/prepared_resources_b5f2edcc5dc94607b1a8478c07a096a6.json；完整七段memory已更新。本轮无新增模型、候选、实现修订、Git提交、CPU诊断、原生两步检查、训练启动或全局研究冷却。

## 5. Decisions Made
- 继续保留两项固定30轮原训练和三项已实现候选，尊重GPU0既有280W/GPU1 390W设置。GPU0虽核心温度降低但仍出现热降频，不宣称散热或掉卡根因已解决，不自动改风扇、继续降低或回升功耗。
- 三项下一步仍是独占空卡上的真实原生YAML/TRAIN两步检查。空卡后优先Y58/Y59再Y60，用当轮有效租约重核版本/goal/资源，检查通过即同轮入队并尝试门控派发，不再索取已覆盖的常规操作授权。
- 本轮只是到期资源复查和训练进展核验，没有新路线决策，不重复写无变化other日志，不为配额创建重复候选或修改调度；随后canonical finish，结果以真实收据为准。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
