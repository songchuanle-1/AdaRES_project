# Research Pipeline Report - AdaRES_project
快照时间：2026-10-29T00:52:03.249381+00:00；生成报告时快照年龄：89 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-29 08:52:02
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
================================================== 2026-10-29 08:51 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y64_directed_containment_res] Best=0.5029 29/30 no_improve=9/30 gpu=0 ? 序列: 0.4943, 0.5008, 0.4949, 0.5009, 0.4972

--- server-2 (192.168.1.163) ---
  [adares_y65_confidence_feedback_res] Best=0.3561 3/30 no_improve=0/30 gpu=0 ? 序列: 0.3297, 0.3561

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
| AdaRES-Y64 | configs_song/model_paper5/adares_y64_directed_containment_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 118：complete=16 / rejected=79 / closed=12 / blocked=9 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 43（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 已核实际归属/Disabled调度/前轮已释放，canonical begin先监控/目标/派发已有任务；本轮无waiting/claiming、新终态或operational issue。
- Y64原server-1 PID2752219 Train29/30从1608至1615，Y65原server-2 PID112356 Train3/30从4469至4479，均实际递增；六源码/原生YAML/cwd/config/各原marker/准入/独占通过，Y65功率280W。Y65用户时间采用本地runner，远端时钟偏移不冒充上海当前时间。
- 两服canonical物理证明绑定本轮token/snapshot，GPU0分别为Y64和Y65且预留，授权资源均忙；Y66和原七项冻结全量后评继续资源阻塞，GPU1用户暂停未使用。

## 5. Decisions Made
- 保留两个现有实验同PID/模型优化器/seed1999/fixed30，进度不是训练效益；不重放已完成准入、派发、监测修复或旧归档。
- Y66已有CPU/限定版本/原生plan和未激活server-1备选，下一缺项需真实GPU；当前无授权空槽，不重复等价CPU检查或凑候选。Y64真实完整终态并canonical归档后，若server-1真实idle则按ID激活Y66并原生两步，PASS后冻结原后评/准入/当轮派发。
- 保留Y57/GPU1暂停、server-2 GPU0 280W、原数据/禁用/恢复协议和七项全量冻结后评，不改调度。无新增科研结论、启动或用户待办。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
