# Research Pipeline Report - AdaRES_project
快照时间：2026-10-23T20:44:31.803701+00:00；生成报告时快照年龄：157 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-24 04:44:31
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
================================================== 2026-10-24 04:44 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y60_stage_specific_confidence_res] Best=0.4965 11/30 no_improve=0/30 gpu=0 ? 序列: 0.4647, 0.4702, 0.4709, 0.4880, 0.4965

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
- 上海2026-10-24 04:44:31取得design快照，核实实际CODEX_THREAD_ID与AdaRES owner一致、后台调度Disabled；experiment_first启用，running1/waiting0，无新增终态或操作故障。仅server-2 GPU0/280W，保留Y57暂停和外部GPU1任务。
- 04:44:59实核Y60原PID17186/命令/cwd/starttime/GPU0归属，Val11/30步158至205增长，准确为第11轮验证阶段，不写成训练batch。GPU0为84至85C/280W/22746MiB，当次未见热降频、训练尾部错误或新kernel Xid/fallen-off，不代表历史散热问题解决。证据pipeline/checks/ef_gpu0_runtime_81a2c3ef30724df5bebc4aa514a3bd38.json。
- 04:46:07通过生命周期v3实查GPU0仍由Y60占用，无独占空闲授权卡；Y56/Y58/Y59既有done_complete及不可变冻结后评证据一致，Y61七文件/9bea提交/原生smoke计划/current goal一致。tick_attempt为Y56/resource_check/blocked，证据同时列入对应候选；GPU1外部任务未干预。证据pipeline/checks/gpu0_lifecycle_v3_81a2c3ef30724df5bebc4aa514a3bd38.json。
- 本轮没有新候选、模型/YAML/代码修改、Git提交、CPU或GPU测试、生产同步、准入、入队或训练启动；Y61先前交接入口与CPU回归未重跑、不当成本轮成果。Y60原训练不重启或迁移，Y57继续暂停，已完成训练不重复归档或晋升，无新科学效益结论。
- 更新七段紧凑记忆，按canonical report/finish收尾；实际结果以命令和收据为准，未预称校验成功，不在释放后补写。

## 5. Decisions Made
- 本轮无新增研究或操作决策。继续执行既有GPU0-only/280W、Y57暂停、固定30轮/seed1999和不可变后评/原生检查计划。真实独占GPU0空闲后再执行Y61原生两步，实际PASS后才准入派发；不修改资源授权、队列、调度或历史结论。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
