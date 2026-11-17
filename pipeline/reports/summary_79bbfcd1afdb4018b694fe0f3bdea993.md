# Research Pipeline Report - AdaRES_project
快照时间：2026-11-07T00:49:43.614974+00:00；生成报告时快照年龄：1990 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-07 08:49:42
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
================================================== 2026-11-07 08:49 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y90_query_peer_affinity_res] Best=0.3260 2/30 no_improve=0/30 gpu=0 ?

--- server-2 (192.168.1.163) ---
  [adares_y89_stage_evidence_norm_res] Best=0.4950 11/30 no_improve=1/30 gpu=0 ? 序列: 0.4660, 0.4739, 0.4795, 0.4950, 0.4876

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y89 | configs_song/model_paper5/adares_y89_stage_evidence_norm_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y90 | configs_song/model_paper5/adares_y90_query_peer_affinity_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 143：complete=16 / rejected=79 / closed=12 / blocked=34 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 67（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |
| AdaRES-Y67 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.676817+00:00 |
| AdaRES-Y70 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.677616+00:00 |
| AdaRES-Y73 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.678374+00:00 |
| AdaRES-Y75 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679091+00:00 |
| AdaRES-Y78 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679826+00:00 |
| AdaRES-Y81 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.680927+00:00 |

## 4. This Tick's Actions
- 本轮快照生成于2026-11-07T00:49:43Z。实际所属会话匹配，后台调度保持禁用，科学goal和GPU1暂停授权保持。先由正式begin查询进展并派发已有合格队列；当前没有就绪待跑任务或连接、同步、派发故障。Y88新增终态事件是上一轮已经精确核验归档的历史结果，本轮不重放其checkpoint加载或归档。
- 先核验原Y89和Y90实际训练、原生YAML、代码、收据、部署标记及功耗。收尾重新采样确认Y89 PID60578在原server-2 GPU0/280W持续推进至Train12/30、5/4686，Y90 PID2442962在原server-1 GPU0/350W持续推进至Train3/30、2241/4686。没有重启、迁移、同步或改动两个现有训练。
- 完成Y91显式加法身份绑定：将完全相同的七个既有CPU测试文件保留为直接位于当前Y90上的独立提交59baf94e9d3e406afa13b2ddc96968cdcc976cae；原CPU证据、原提交、原计划和原助手保持。14项实际源码、负控制、运行环境及当前占用者守卫通过；只更新冻结提交元数据并新增严格绑定当前Y90/Y89占用者的原生执行助手，未重放CPU科学实验、未执行GPU原生任务或准入。
- 新实现一个独立Y92视觉侧非线性mask读出变体：仅在原始x_mask输出后、原始batching之前加入一次零初始化视觉通道平方投影，保留原始全部点、通道、查询、分数与共享置信度消费者，不叠加Y88查询侧变换或Y91置信度门控。实际原生YAML、完整CPU模型、发布骨干、原TRAIN162738点/8表达加载、float32/64标量和置换/无RNG/零初始对照、七阶段W2与输入梯度对照、初始mask干预/初始分数不变/精确旁路、原消费者及两次合成更新首轮通过。CPU不是原生GPU任务反向或效果证据。
- 将Y92实际测试的七个文件独立保留为直接位于Y90上的提交ec332f57dacc09459df8d6d52f21424c4d5ea3de，不包含未训练Y91父版本，不移动生产HEAD或其他索引/工作区。10项真实CPU证据守卫和6项运行环境守卫均首轮通过；13个CPU/原生核心回调完全一致并已在CPU实际执行。冻结原server-1 GPU0/350W及server-2 GPU0/280W原生检查计划，仅采用既有精确数据与路径映射；尚未执行GPU原生任务、准入、入队或训练。
- 首次收尾守卫拒绝一条超过300秒的训练进度证据。保留原助手，记录同一守卫真实诊断日志，机械地顺序刷新两项原生进度采样后用相同300秒条件通过，模型、数据、CPU证据和原生计划均未改变。有效闭环证据为pipeline/checks/y92_preparation_closeout_79bbfcd1afdb4018b694fe0f3bdea993.json，诊断与刷新证据为pipeline/checks/y92_closeout_observation_refresh_79bbfcd1afdb4018b694fe0f3bdea993.json。
- 最后顺序刷新两台授权服务器的正式占用证据，保存七节记忆与本轮报告。完整收尾以正式finish收据为准，不提前声明postflight结果。

## 5. Decisions Made
- 本轮新增独立科学候选1、准入0、训练启动0、归档0、GPU原生检查0。Y91先于Y92；二者的原生任务均等待匹配的当前训练自然完成、完整checkpoint精确核验和规范归档，以及新鲜独占、未预留的合法GPU0。当前两卡已占用，不挤占训练或外部任务，不凑满额度新增第三个准备候选，不以CPU通过代替原生准入。
- Y91不重放既有CPU/版本/绑定/准备；后续使用新的run_y91_bound_native.py。Y92仅在生产HEAD仍匹配当前Y90和精确原占用者时使用已冻结计划；若Y91先推进HEAD，必须新增同七个文件的显式测试身份/运行环境/占用者绑定，不编辑旧hash绑定助手、不重置源码、不组合机制或绕过原生守卫。
- 原完整冻结协议保持：9508表达/141场景、候选与Y36/W2/B1各两次、FIRST为主、场景聚类bootstrap10000/seed1999、TRAIN分界1487/3325、原有害子组和mIoU/Acc25/50/查询数/延迟/显存/同卡成本。正式冻结FIRST最佳仍为Y36 0.510725438294，目标0.5150，本轮没有后评或成本更新。
- server-2 GPU1全部使用与Y57实验分别继续用户暂停，原GPU1冻结后评/成本不迁移或替换资源；Y86保持原数值控制挂起，无不变重试或门槛放松。server-6、跨项目、功耗与调度修改均未执行；原goal、收据、数据和运行环境保持，无需用户新增确认。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
