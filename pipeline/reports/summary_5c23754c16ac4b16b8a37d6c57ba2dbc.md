# Research Pipeline Report - AdaRES_project
快照时间：2026-10-29T04:52:10.382146+00:00；生成报告时快照年龄：678 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-29 12:52:09
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
================================================== 2026-10-29 12:52 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y66_peer_state_feedback_res] Best=0.3659 3/30 no_improve=0/30 gpu=0 ? 序列: 0.3314, 0.3659

--- server-2 (192.168.1.163) ---
  [adares_y65_confidence_feedback_res] Best=0.4808 9/30 no_improve=0/30 gpu=0 ? 序列: 0.4451, 0.4569, 0.4584, 0.4688, 0.4808

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
- 候选累计记录 118：complete=16 / rejected=79 / closed=12 / blocked=9 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 44（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 上海12:52核Y65原Train9/30从3029至3039、Y66原Train3/30从1583至1591；六源码/YAML/cwd/config/marker/原准入/GPU0独占通过，server-2保持280W。上轮Y64终态已完整归档，不重放。
- 按快照待跑0/creation2新增独立Y67：七阶段mask/score读出各自学习LayerNorm仿射参数，保留原七次共享confidence头/损失及数据/seed1999/fixed30/Q128/selector/Boolean gate，+3072参数；不预证收益。
- Y67首轮CPU检查因诊断脚本在参数干预/恢复后复用旧前向图后向导致autograd版本错误，原失败log及程序保留。仅新脚本重新计算私有梯度前向，模型/YAML七文件未改；上海12:57:44至12:58:08实际重检通过。
- 实际CPU原生YAML/完整模型/发布骨干/真实TRAIN collate162738点8表达，以及float32/64独立norm/初值输出/独立公式/私有梯度隔离、合成完整DEC七阶段输出与输入梯度/RNG锚点、各stage非零梯度和末阶段干预/恢复负对照全部通过。尚无实际稀疏criterion/optimizer或GPU检查。
- 仅提交Y67七文件至db35并保存待执行原生计划，8项native路径/梯度/优化器更新守卫、旧goal和源码篡改负对照通过。Y65/Y66原882d准入/队列/运行保留，无push或生产同步。
- 两服canonical占用证明显示Y65/server-2 GPU0和Y66/server-1 GPU0均有原compute PID/预约，无授权空槽。Y67原生两步及八项原冻结全量后评/成本等待，Y57/GPU1暂停保留。

## 5. Decisions Made
- 准备1项结构候选，不强行凑满预算，不使用CPU结果签原生PASS或准入。优先保留两项正常原训练；资源释放后用已保存计划做真实原生YAML/TRAIN两步并在通过后当轮准入/派发。
- Y67只解除已消费的预测读出仿射参数共享；与Y60独立confidence头、Y26旧norm重连、Y25 soft gate和Y65/Y66反馈不同，不重开历史拒绝、不改验证边界或扫参。
- 保持原冻结全量评估/配对成本和不确定性协议，CPU机制检查及训练日志不替代效益证据。只用server-1 GPU0/server-2 GPU0，保持280W/默认350W；Y57/GPU1暂停，server-6/跨项目/驱逐/重启/改调度禁止，无用户待办。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
