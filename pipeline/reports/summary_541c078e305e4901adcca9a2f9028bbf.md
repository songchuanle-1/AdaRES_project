# Research Pipeline Report - AdaRES_project
快照时间：2026-10-20T05:26:03.270582+00:00；生成报告时快照年龄：156 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-20 13:26:02
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
================================================== 2026-10-20 13:25 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y52_point_target_res] 24/30 尚无验证 - gpu=1
  [adares_y54_fine_support_res] 25/30 尚无验证 - gpu=0

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y52 | configs_song/model_paper5/adares_y52_point_target_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y54 | configs_song/model_paper5/adares_y54_fine_support_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。
- 研究储备 2 / 2；储备准备预算 0，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 0、未分类阻塞 2。
- 非终态候选与排队项合计 2；总上限 8。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 109：complete=14 / rejected=79 / closed=12 / training=2 / blocked=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 33（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮为2026-10-20上海13:26:03.270582规范新快照的event，处理previous_tick_unhealthy及ledger/research_artifact/research_work变化；begin已各执行一次monitor和goal，均返回0，operational_issues为空且没有新终态，不重复preflight。
- 新快照active2/running2/waiting0/claiming0、容量2，准入预算0、储备2/2及准备预算0、due0/blocked_not_due2。当前Y56/Y57仍等待独占空闲卡，不把旧retry当作释放GPU或新增授权，不重跑历史CPU诊断。
- 本轮monitor看到Y52 24/30和Y54 25/30尚无新验证，未采集双时间点原生步数、GPU利用率或显存。上一轮13:04:51-13:04:57的Y52步骤127至138、13:05:38-13:05:44的Y54步骤125至138均为缓存证据，不能冒充13:26新验证。两项历史checkpoint best分别.5094767808914185和.5062541365623474，不是本轮新评估；本轮新best、Acc@50和spmIOU未知。
- 前轮81e295aec4c1457c9689d4b62dd21b62的规范finish因旧快照active0而实际恢复后running2误判，已按原token abort；13:22:18已独立核实锁/pending不存在，前轮收据INCOMPLETE/context未提交，训练继续。旧失败及六份恢复证据均保留，本轮不改旧快照或检查器。
- 更新七段紧凑记忆和本轮六段审计报告，明确历史原生采样与当前登记观测。本轮没有新科学测试、候选ID、模型修订、Git提交/推送、准入、队列追加、训练启动、GPU诊断或调度修改，没有新科学收益结论。

## 5. Decisions Made
- 无新增研究决策；沿用Y52/Y54已恢复的同一fixed30/seed1999训练，模型/优化器/学习率状态不重置，不停止训练给候选腾卡。完成后仍按冻结全9508表达/141scene双FIRST与子组伤害规则评估；历史冠军Y36 .5110、目标.5150、差.0040不变。
- Y56/Y57保持原inconclusive/exclusive_gpu_slot、冻结测试和既有负证据。当前无到期任务或储备准备预算，等待实际资源/队列变化；空闲后须在新当前租约核对source/checkpoint/GT/实时占用再执行原冻结观测。不重新索要已有授权，不制造全路线耗尽或科学冷却。
- 规范finish后以返回值/收据和锁/pending独立核验判定本轮是否完成；当前不预称PASS。新快照与实际队列一致不等于已修复旧检查器。用量只代表begin至receipt本地可见窗口而非账单，未知不是零；缓存已含input、推理已含output，不重复相加。释放后不写项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
