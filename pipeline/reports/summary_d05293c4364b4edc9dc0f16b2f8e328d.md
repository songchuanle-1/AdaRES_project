# Research Pipeline Report - AdaRES_project
快照时间：2026-11-03T08:49:53.867468+00:00；生成报告时快照年龄：493 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-03 16:49:53
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
| server-1 | 0 | 未知 | 未知 | 未知 |
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-11-03 16:49 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y79_query_language_feedback_res] Best=0.5003 22/30 no_improve=6/30 gpu=0 ? 序列: 0.4907, 0.4992, 0.4987, 0.4962, 0.5002

--- server-2 (192.168.1.163) ---
  [adares_y81_query_mask_refinement_res] Best=0.4737 8/30 no_improve=0/30 gpu=1 ? 序列: 0.4255, 0.4441, 0.4624, 0.4651, 0.4737
  [adares_y80_query_visual_feedback_res] Best=0.5011 13/30 no_improve=1/30 gpu=0 ? 序列: 0.4794, 0.4936, 0.4982, 0.5011, 0.5005

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y80 | configs_song/model_paper5/adares_y80_query_visual_feedback_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y81 | configs_song/model_paper5/adares_y81_query_mask_refinement_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y79 | configs_song/model_paper5/adares_y79_query_language_feedback_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 135：complete=16 / rejected=79 / closed=12 / blocked=25 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 59（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 运行：AdaRES-Y79 Train22/30 step1918/4686；AdaRES-Y80 Train13/30 step4231/4686；AdaRES-Y81 Train8/30 step4362/4686，原PID/source/YAML/receipt/marker/功耗与native计数递增核验通过。
- 仅当前资源/进度与Y82/Y83原native准备完整性复核，不重放CPU/prepare。
- 新候选0/准入0/启动0/归档0，无正式FIRST/cost；三卡fresh busy+reserved，Y82优先Y83等待原native GPU检查，记忆与备份保存。

## 5. Decisions Made
- 保留原Y79/Y80/Y81训练，不挤占GPU；原Y82准备不重放，完成准备目标后按资源条件继续。
- 原native任务必须使用匹配计划和fresh授权独占空槽；实测PASS后才按原完整冻结后评/receipt/精确source契约准入派发并核步数，CPU不得代替GPU检查或效益。
- 保留科学goal/原receipt/数据协议/私有环境/精确mapping/禁用/seed1999-fixed30与350W-280W-390W；Y57单独暂停，禁server6/跨项目/调度变更，无新增用户操作。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
