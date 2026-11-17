# Research Pipeline Report - AdaRES_project
快照时间：2026-10-29T08:49:41.404182+00:00；生成报告时快照年龄：375 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-29 16:49:40
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
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-29 16:49 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y66_peer_state_feedback_res] Best=0.4648 7/30 no_improve=0/30 gpu=0 ? 序列: 0.3659, 0.4230, 0.4430, 0.4497, 0.4648

--- server-2 (192.168.1.163) ---
  [adares_y65_confidence_feedback_res] Best=0.5010 15/30 no_improve=2/30 gpu=0 ? 序列: 0.4868, 0.4991, 0.5010, 0.5005, 0.4999

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y65 | configs_song/model_paper5/adares_y65_confidence_feedback_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y66 | configs_song/model_paper5/adares_y66_peer_state_feedback_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 119：complete=16 / rejected=79 / closed=12 / blocked=10 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 44（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 上海16:50核Y65原Train15/30从1328至1338、Y66原Train7/30从4550至4560；真实六源码/YAML/cwd/config/marker/原准入/GPU0独占通过，server-2保持280W。
- 为既存Y67准备实际native PASS后的准入/原冻结协议保存/门控派发接续helper，并实际执行14项CPU负对照：身份/goal/版本/YAML、两步有限loss、七阶段路径/消费、完整14仿射梯度与真实optimizer更新、公式观察、TRAIN数据/场景、显存/独占空闲和280W。14项全部通过，未执行native/admit/dispatch。
- 复核Y67当前HEAD=db35/冻结七文件与提交及goal一致，原Y64训练/全量后评协议的深复制原型不变；本轮不生成实际准入freeze。Y65/Y66原运行/收据和Y57暂停保留，手动测试fixtures不冒充原生结果。
- 两服canonical物理证明GPU0仍分别有Y65/112356和Y66/1973300及预约，显存均约22GiB；server-1即使瞬时util0仍不是空闲。Y67原生两步及八项原冻结后评继续待资源，GPU1仍用户暂停。

## 5. Decisions Made
- 继续最接近准入的Y67，先完成不占GPU的接续实现，不挤占原运行、不改名重开候选。当前无授权空槽，真实native未执行，不准入、不派发、不宣称收益。
- GPU释放且canonical终态归档/预约清除后，用当前token运行既有Y67实际原生YAML/TRAIN两步；真实PASS再按准备好的helper同轮签收/冻原协议/入队/门控派发并核递增原生步数。
- 保持server-2 GPU0 280W、server-1默认350W，Y57/GPU1暂停及专用恢复、数据/禁用项/调度不变。内部接续准备未改变用户可行动状态，例行训练健康，无需用户处理。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
