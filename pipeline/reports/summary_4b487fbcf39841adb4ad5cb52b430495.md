# Research Pipeline Report - AdaRES_project
快照时间：2026-10-22T18:48:57.228333+00:00；生成报告时快照年龄：185 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-23 02:48:56
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
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-23 02:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y59_frozen_text_mode_res] Best=0.4540 6/30 no_improve=0/30 gpu=0 ? 序列: 0.3400, 0.3675, 0.4377, 0.4339, 0.4540

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y59 | configs_song/model_paper5/adares_y59_frozen_text_mode_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 3；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 113：complete=16 / rejected=79 / closed=12 / blocked=5 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 37（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 上海02:49:24实核原Y59 PID28712/命令/cwd/starttime/GPU0归属一致，Train6/30 step3424至3447；02:50资源采样实际尾部step3504至3585增长，未见尾部错误。前五轮训练日志验证序列.3400429785/.3675022423/.4376808703/.4338700771/.4540018737，best .4540018737仅为训练过程指标，不是冻结全量评估或冠军晋升。
- GPU0保持280W，实测83C/22718MiB，热降频仍Active；内核读取成功未见新Xid/fallen-off。GPU1仍有外部负载，未干预；Y57继续按用户指令暂停，当前仅GPU0获准，不扩大GPU或改变功耗/风扇/调度。
- 本轮实际执行修正后的recheck_gpu0_lifecycle_v2.py，仅让旧采集器复查未准入Y60；七文件/原生YAML/冻结计划/goal/commit一致。随后分别保留Y56/Y58固定30完成后的冻结评估待办、Y60原生smoke、Y61版本准备；Y61七文件与原CPU payload及decoder goal一致。新资源记录没有把完成实验写回训练前准入。
- 唯一获准GPU0正在运行Y59，实际空槽列表为空；Y60/Y61未做原生两步GPU检查，未新增候选、模型/YAML修改、Git提交、同步、准入、排队或训练启动。未重复CPU构造或前轮10项回归，不将旧结果冒充本轮实验。tick_attempt=Y60/resource_check/blocked，相关证据同时归入候选。

## 5. Decisions Made
- 保护Y59的固定30轮/seed1999原训练及280W限制，不重启、不迁移；Y57暂停到期不产生恢复授权、不重复询问。
- 已有两项准备实现，下一步均需独占GPU0；本轮真实资源检查后收尾，不为配额再造候选。空槽出现时优先已提交Y60按未改native_smoke_plan做原生YAML/TRAIN至少两步有限loss/backward及资源检查，通过即当轮准入并尝试现有门控派发；保留Y56/Y58冻结全量后评和成本比较，再推进Y61精确版本及原生两步。
- 本轮只有训练进展与资源/版本/生命周期复查，没有新的科学效益结论。继续使用v2 observer和recheck_gpu0_lifecycle_v2.py，不重放已完成实验finalizer，不改快照/预算/禁区或调度；按canonical report/finish收尾并核验context提交和锁释放，当前不预称通过。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
