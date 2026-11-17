# Research Pipeline Report - AdaRES_project
快照时间：2026-11-01T12:52:03.890864+00:00；生成报告时快照年龄：932 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-01 20:52:03
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
================================================== 2026-11-01 20:51 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  无训练进程
  [adares_y72_visual_mask_refinement_res] 进程未见 (deathwatch 1/3) — 暂不处置

--- server-2 (192.168.1.163) ---
  [adares_y74_language_update_gate_res] Best=0.5026 19/30 no_improve=4/30 gpu=0 ? 序列: 0.5026, 0.4993, 0.4990, 0.4968, 0.4962
  [adares_y75_language_query_init_res] Best=0.4796 8/30 no_improve=0/30 gpu=1 ? 序列: 0.4384, 0.4415, 0.4627, 0.4674, 0.4796

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y74 | configs_song/model_paper5/adares_y74_language_update_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y75 | configs_song/model_paper5/adares_y75_language_query_init_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y72 | configs_song/model_paper5/adares_y72_visual_mask_refinement_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 129：complete=16 / rejected=79 / closed=12 / blocked=19 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 53（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- Y72自然完成固定30轮：完整Train/Val1..30、稳定finite last30/best/model/optimizer544/scheduler30和14、原源/YAML/receipt/freeze及20生命周期守卫核验后canonical done_complete归档，原checkpoint留远端。训练best0.500363588在epoch14，不是正式FIRST。
- 确认server1GPU0独占空闲且既有350W后，Y76原生YAML/tools.train真实TRAIN两步loss4.61551094/6.31082487通过；七stage conditioning/query_norm身份和真实消费、有限loss/梯度、全部七offset行非零梯度与参数更新均核。16准入/资源/源版本证据拒绝守卫通过。
- 深复制并核等原Y64完整training/evaluation，Y76独立准入并入队；主master仅选择七测试A文件，安全additive同步仅server1，已门控启动PID3489002，原生Train1步数递增至415。server2原Y74/Y75 PID49891/65454持续推进，源码/YAML/receipt保留。
- 三授权卡现均busy+reserved，Y77原准备及两GPU原生计划有效，等待匹配独占空卡；17项完整冻结后评/cost仍待资源。保存handoff与七段记忆，Y57继续暂停，正式FIRST/成本未更新。

## 5. Decisions Made
- 优先推进最近可入队的Y76，实际native PASS后本轮完成准入、门控派发和进度确认；本轮准入1/启动1/完成1在snapshot预算2内，不要求凑满额度。
- Y72只归档一次，不重训；Y76实际两批native验证不代替效益/全量显存/原完整协议。保留原数据/seed1999/fixed30/禁用项、350/280/390W和Y57专用暂停要求。
- 后续observer使用独立y76精确版本契约；Y77须真实native PASS和498e精确新契约才可准入派发。不触碰server6/其他项目，不改调度频率或科学goal。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
