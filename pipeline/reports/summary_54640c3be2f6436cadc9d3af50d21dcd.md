# Research Pipeline Report - AdaRES_project
快照时间：2026-10-02T05:47:43.197592+00:00；生成报告时快照年龄：141 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-02 13:47:42
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
================================================== 2026-10-02 13:47 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y23_valid_token_support_res] Best=0.4257 4/30 no_improve=0/30 gpu=0 ? 序列: 0.3327, 0.3713, 0.4257
  [adares_y18_point_measure_res] Best=0.5070 20/30 no_improve=6/30 gpu=1 ? 序列: 0.5007, 0.5048, 0.5043, 0.4971, 0.4981

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y18 | configs_song/model_paper5/adares_y18_point_measure_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y23 | configs_song/model_paper5/adares_y23_valid_token_support_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 核实实际当前任务与foreground所有者一致；仅操作AdaRES_project。最新execution_mode/goal/SERVERS/closure_policy及冻结协议未变，Windows后台Disabled；未访问其他项目、更改调度、新建任务或嵌套codex exec。
- 快照北京时间2026-10-02 13:47:43，event仅因上一轮research_artifact/research_work变化；无新终态、到期候选0、活跃2/2、等待0、准入预算0。begin已执行monitor/goal各一次，未重复preflight。
- 北京时间13:48:06至13:48:17复用动态token有界采集，密钥BatchMode SSH验证server-2主机dl01-MS-7C60、原PID/config/GPU/独有日志。Y18 GPU1 PID7449 Train20/30步数1695→1705/4686；Y23 GPU0 PID125903 Train4/30步数131→139/4686，均真实递增。
- 独立GPU0/1设备利用率57%/62%、显存22718/22720MiB，不是候选隔离峰值。仅读各次至多24000字节日志尾，窗口中未见Traceback/OOM，不代表全日志无故障；保留远端时钟偏差，以本地观测时间为准。
- 新增证据pipeline/analyses/native_progress_54640c3be2f6436cadc9d3af50d21dcd.json SHA256=d863be874e85d374e0ceb50768df6c15b1c9df8e4e8e2219fce640d3ca4bd5a0及原始采集日志，追加至Y18/Y23稳定ID并更新七段短记忆，保留历史失败、冻结评估与重复不稳定性证据。
- 科学状态training；预检Y18 best0.5070/20轮/no_improve6，Y23 best0.4257/4轮/no_improve0仅为训练期间读数，最终达标、有效性、Pareto未知。本轮Acc@0.25/Acc@0.5/完整延迟/隔离峰值未独立采集，记未知，不根据中期验证读数改协议。
- 执行状态须以finish实际postflight和锁释放为准，报告生成时尚未完成。实际产出为1份双训练新证据、1份原始日志、2个候选证据更新、七段记忆和六段报告，新候选/准入/启动0。
- 收据统计窗口仅为本地begin至收据，排除前置阅读、晚到记录及最终答复；收据生成前用量未知，finish后只在任务中读取报告，不释放后写项目。缓存包含在输入，推理包含在输出，不重复累加，不将用量当账单。

## 5. Decisions Made
- 无新科学决策。两张授权卡各运行一个原始独立实验，无到期可执行研究项且准入预算0，继续Y18/Y23固定30轮/seed1999；不重启、抢占训练、填槽造变体或重复闭合路线审查。
- Y18到20轮不代表应截断或重新扩展，本轮保留原进程直至冻结30轮终态；终态后按Y18/Y23冻结计划执行全ScanRefer、W2/B1、重复、场景聚类区间、子群损害和成本评估，不验证集调参或提前提升champion。
- 正常训练进展无需用户介入；未新增无信息增益/外部阻塞判定，也未更改历史design_cycle。研究仍有价值，保留自动化；下次精确调度未知，以保存的自动化为准。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
