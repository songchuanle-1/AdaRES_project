# Research Pipeline Report - AdaRES_project
快照时间：2026-10-27T20:44:00.309763+00:00；生成报告时快照年龄：386 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-28 04:43:59
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
================================================== 2026-10-28 04:43 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  无训练进程
  [adares_y63_peer_bilinear_res] 进程未见 (deathwatch 1/3) — 暂不处置

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y63 | configs_song/model_paper5/adares_y63_peer_bilinear_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 3；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 117：complete=16 / rejected=79 / closed=12 / blocked=9 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 41（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 核验当前实际会话与项目owner、旧Windows调度Disabled及前dab0收据FINALIZED/PASS/context_committed/released。canonical begin/design快照初始仍登记Y63 running；实际SSH只读核实其原PID3071037已退出，原生Train30/30、Val30/30和最后保存记录完整。
- 新增scripts/finalize_y63_fixed30.py及test_y63_terminal_handoff.py，实际运行python -B scripts/finalize_y63_fixed30.py --token afe43c6c4f504b6d9d5052c0af50e4c9。12项CPU保护测试通过（错ID/GPU/服务器/PID/状态、改暂停、不完整epoch、重置scheduler、空optimizer、错检查点路径、非有限、活进程、错误或重复映射、无别名精确清理）。日志pipeline/checks/y63_server1_fixed30_afe43c6c4f504b6d9d5052c0af50e4c9_tests.log，SHAc5797af83fef88ad37f069fbdefb9c3a04151cf345cc6aa00bc7923ec2988e01。
- 真实CPU检查Y63最后和最佳检查点：last epoch30/scheduler30/optimizer544，best epoch21/scheduler21/optimizer544，模型及优化器张量全部有限、文件size/mtime稳定并计算SHA；两份best_miou均0.5074337124824524。CUDA未初始化、未执行全量冻结复评。last SHA b4f23ec71979b4996076af9c48820b4531a19bfc0781211d93098715842ca92f；best SHA ecaa9070908c3cecff86394fc413a09a347906170c0eee5c2677efc4d2b44eb0。
- 远端六源码、原生YAML确定性映射、marker82f、原准入/冻结协议验证后，canonical finalize_task明确gpu=task['gpu']归档Y63为done_complete/30/30，完整日志拉至Checkpoints/server-1/adares_y63_peer_bilinear_res/afe43c6c4f504b6d9d5052c0af50e4c9/fixed30.log，checkpoint留远端。只移除Y63 queue/running，Y57暂停和其他记录保留，archive_pending空。终态证据pipeline/checks/y63_server1_fixed30_afe43c6c4f504b6d9d5052c0af50e4c9.json(SHAb7b0b4d5d9f334713f2d88c011f0048dbb8ca7cdac755dee2aca170945f002d2)，archive_binding同前缀(SHA55435c92976838c8af9e14be2156f58969db5af52770183d87656b138f81e560)。
- 实际canonical occupation复查：server-1 GPU0为未登记PID2489339、1578MiB、0%，无本项目预留；仍不是独占空闲。server-2 GPU0为未登记PID13813、24138MiB、97%；直接只读hostname/280W核验。occupation证据分别pipeline/checks/gpu_occupation_server-1_afe43c6c4f504b6d9d5052c0af50e4c9.json(SHAc38859ddeda121954adf309d4656be6d14281d93e52ebc638899088120148125)及server-2同前缀(SHA291257ddc7a7beb3039c2b1019e3d86778d3cf3b9131cb9a6c437cc609d9e923)。Y62/Y63为冻结后评资源挂起，Y64/Y65仍需原生GPU检查；资源描述随实际改变更新。
- 当前无已登记running/waiting/claiming，只有Y57 paused；没有新模型源码/commit/push/生产sync/native GPU检查/准入/入队/启动。更新七标题memory，报告等待canonical finish核验，不预称收尾通过。

## 5. Decisions Made
- Y63训练best .5074仅为训练期间验证指标，尚未超过历史CSV .5110；原冻结全量FIRST复评/成本协议仍待GPU，不能据此直接判机制无效、晋升或调参重训。
- Y63实际完成不代表GPU空闲。保留server-1未登记PID2489339与server-2 PID13813，不将util0当独占空闲，不驱逐或干扰这些任务，不恢复Y57/GPU1。
- 授权独占空槽出现后优先推进Y64。server-1的canonical Y63归档已完成，无需重放；按已有备选activate和runner重新核hostname/350W/路径/goal/版本/compute/lease。server-2按原plan/280W。实际原生两步PASS并冻结后评后才准入/入队并当轮尝试启动，再推进Y65。
- 不跨项目、不使用server-6、不改调度，不重开拒绝/禁用路线，不改原数据/seed1999/fixed30/selector/损失等协议。新归档是实质完成，通知用户；无需新增操作授权或用户处理。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
