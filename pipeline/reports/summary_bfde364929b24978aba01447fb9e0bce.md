# Research Pipeline Report - AdaRES_project
快照时间：2026-10-23T18:45:11.843578+00:00；生成报告时快照年龄：457 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-24 02:45:11
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
================================================== 2026-10-24 02:45 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y60_stage_specific_confidence_res] Best=0.4709 9/30 no_improve=0/30 gpu=0 ? 序列: 0.4463, 0.4634, 0.4647, 0.4702, 0.4709

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y60 | configs_song/model_paper5/adares_y60_stage_specific_confidence_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 3；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 113：complete=16 / rejected=79 / closed=12 / blocked=5 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 38（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 上海2026-10-24 02:45:11取得design快照，核实实际CODEX_THREAD_ID与owner一致、后台调度Disabled；experiment_first启用，capacity1/running1/waiting0，无新增终态或连接部署故障。仅server-2 GPU0/280W，保留Y57暂停和外部GPU1任务。
- 02:46:00实核Y60原PID17186/命令/cwd/starttime/GPU0归属，Train9/30步1019至1041增长。GPU0为84至86C/280W/22744MiB，当次未见热降频、训练尾部错误或新kernel Xid/fallen-off，不代表历史散热问题解决。证据pipeline/checks/ef_gpu0_runtime_bfde364929b24978aba01447fb9e0bce.json。
- 02:46:51通过生命周期v3实查GPU0仍由Y60占用；Y56/Y58/Y59已完成训练的终态与不可变冻结后评证据一致，Y61七文件/9bea提交/原生计划/current goal一致。tick_attempt为Y56/resource_check/blocked，证据同时列入候选；没有GPU原生检查或新准入。证据pipeline/checks/gpu0_lifecycle_v3_bfde364929b24978aba01447fb9e0bce.json。
- 发现现有专用准入派发helper仍绑定Y60，补充scripts/admit_dispatch_y61_gpu0.py及test_y61_admission_handoff.py，不改hash绑定旧工具、Y61模型/YAML或已冻结原生检查计划。Y61入口要求真实当前轮native raw/log、精确goal/commit/YAML、两步有限loss/backward、七阶段共享头路径、当前独占GPU0/280W及Y60 canonical done_complete后才可准入，沿用既有门控launcher和原生进度核验。
- 本地15项CPU schema/负控制及Python3.8语法检查通过；覆盖缺真实结果、NaN/少步、错ID/版本/goal/YAML、未TRAIN、错误peer分支、运行任务/忙卡拒绝相应操作。正schema fixture是合成回归输入，不是真实训练或准入证据。实际check复核Y61精确版本/冻结计划，原始smoke未生成、admitted/native_smoke_executed/queue_changed均false，queue/running哈希未变。证据pipeline/checks/y61_admission_handoff_bfde364929b24978aba01447fb9e0bce.json。
- 本轮无新候选、模型修订、Git提交、GPU工作负载、同步、准入、入队或训练启动；HEAD仍9bea5e906be3a478be50442db521e766c446a1ec，Y60不重启/迁移，Y57继续暂停。更新七段记忆，按canonical report/finish收尾，未预称校验成功。

## 5. Decisions Made
- 本轮先补齐已准备Y61的原生检查到准入派发交接入口，不借用Y60身份签收据，不改旧hash证据。新入口本地预检通过不能触发实际准入或代替GPU两步；未满足原生/资源条件时继续保留Y61原ID和计划。
- 继续保护Y60原部署6ac/固定30轮/seed1999/280W及Y57暂停。独占GPU0空闲后按原冻结计划后评Y56/Y58/Y59，Y61使用当前token执行原两步计划，真实PASS后再调用Y61专用admit及dispatch并核验原生步数。不扩大资源、改调度或重复索取已有常规授权。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
