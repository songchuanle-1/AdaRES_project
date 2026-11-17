# Research Pipeline Report - AdaRES_project
快照时间：2026-10-29T09:37:21.720153+00:00；生成报告时快照年龄：678 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-29 17:37:21
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
================================================== 2026-10-29 17:37 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y66_peer_state_feedback_res] Best=0.4699 8/30 no_improve=0/30 gpu=0 ? 序列: 0.4230, 0.4430, 0.4497, 0.4648, 0.4699

--- server-2 (192.168.1.163) ---
  [adares_y65_confidence_feedback_res] Best=0.5022 16/30 no_improve=0/30 gpu=0 ? 序列: 0.4991, 0.5010, 0.5005, 0.4999, 0.5022

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
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 3；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 119：complete=16 / rejected=79 / closed=12 / blocked=10 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 44（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 按本会话“server2 GPU1也可以使用”新授权，记录server2_gpu1_20261029授权，SERVERS白名单扩至GPU0/1，AGENTS说明Y57单独暂停保留；科学goal字节精确不变以保留原运行签名，旧资源句由最新明确grant覆盖。新canonical快照容量3。
- Y67只适配物理GPU1/既有390W和新授权绑定，七模型/YAML/HEADdb35和原科学检查body核可逆逐字节等同。上海17:39:14至17:39:50原生YAML/TRAIN/tools.train真实两步通过，loss7.886719703674316/8.231637001037598，551有限梯度、七stage/实际消费/14仿射更新/公式通过，峰6593445888B。
- 经14项GPU1准入守卫，深复制并核等同原Y64完整训练/全量后评协议，签收/入队/门控限定增量同步并启动Y67 PID100811；实际Train1/30从108至117，源码/YAML/marker/准入/独占/390W确认。
- Y65原PID112356/code882d和GPU0 280W保持；因Y67七文件增量部署使共享marker升级db35，新严格additive观察器核ancestor/准确七文件差异/原Y65六源码与YAML等同，4项负对照通过；实核Y65 Train16从2675至2684。Y66原server-1 PID1973300继续Train8从4672至4676。
- 用应用工具更新adares-tick资源提示并实际读保存配置核验，ACTIVE/原会话/每两小时频率和通知意图保持；canonical提示参考仅AdaRES段同步。三卡canonical占用及预留核实，Y57实验仍暂停、epoch27原恢复要求保留，八项原冻结后评待资源。

## 5. Decisions Made
- GPU1资源新授权用于已准备的Y67；不自动解释为恢复被单独暂停的Y57。每卡一项，GPU0 280W、GPU1既有390W和server-1既有350W均不作硬件改变，server-6仍禁用。
- 原生真实PASS后同轮完整准入/派发和递增步数确认，未以CPU/schema/进程存在代签；三项原运行源码/协议/数据/seed1999/fixed30保留。新共享marker仅反映增量部署，不更改Y65原run commit或receipt。
- 原完整val9508表达/141场景、方法与Y36/W2/B1各两遍、FIRST主分析/scene bootstrap10000/seed1999/TRAIN切点1487/3325与伤害/成本协议保留；初始化与训练步数不代表效益。Y57如需恢复仍从原epoch27完整model/optimizer/scheduler续跑，不可从预训练重启。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
