# Research Pipeline Report - AdaRES_project
快照时间：2026-10-22T04:48:08.838097+00:00；生成报告时快照年龄：534 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-22 12:48:08
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
================================================== 2026-10-22 12:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y58_fixed_backbone_statistics_res] Best=0.4993 17/30 no_improve=3/30 gpu=0 ? 序列: 0.4878, 0.4993, 0.4951, 0.4918, 0.4966

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1 waiting=1
    待推: [AdaRES-Y59] configs_song/model_paper5/adares_y59_frozen_text_mode_res.yaml gpu=0
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y58 | configs_song/model_paper5/adares_y58_fixed_backbone_statistics_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 1 | AdaRES-Y59 | configs_song/model_paper5/adares_y59_frozen_text_mode_res.yaml | waiting | - | server-2 |
- 活跃 2（运行 1 / 认领 0 / 等待 1）；目标 3；准入预算 1。
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 1；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 113：complete=16 / rejected=79 / closed=12 / blocked=4 / training=1 / deployment=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 37（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 上海12:48:08取得本轮实验优先design快照，实际会话归属与owner相符，后台Disabled；继续只用server-2 GPU0、保留280W上限及Y57暂停。没有新终态；Y58运行、Y59已准入FIFO waiting。
- 12:49:22两次实采确认Y58原PID77094/命令/cwd/starttime/GPU0归属不变，Train17/30原生步2699至2723增长，尾部无报错。GPU0=280W/84至85C，短采样无热降频、新Xid/fallen-off；GPU1仍有外部compute，未干预。这不抹去前轮热降频历史或证明硬件修复。证据pipeline/checks/ef_gpu0_runtime_3d166dcb16cb4aa5a5bf6b03069cac7e.json。
- 实际复查Y60冻结源码/YAML/提交/当前goal匹配，唯一授权GPU0仍由Y58占用，无独占空槽、未执行native smoke；此时Y58 Train17 step2733至2818，日志前16轮best仍0.4992775022983551，仅中途指标。资源证据pipeline/checks/prepared_resources_3d166dcb16cb4aa5a5bf6b03069cac7e.json绑定Y60及Y61。实际另核验Y58/Y59各7文件/准入收据与当前HEAD一致，Y59无版本阻塞、保持waiting。
- 12:53:28完成原Y61的完整解码器CPU集成：相同初始state、原生YAML六层加初始预测、512/513超点及Q128；七阶段eval/train masks/scores/indis逐位等W2，实测共享头调用7次、query_norm输入requires_grad交替True/False。全部后向有限，七阶段mask输出梯度差均0，最终查询梯度max差1.92245e-6，头参数梯度在预定算术界内相同。证据pipeline/analyses/y61_decoder_cpu_3d166dcb16cb4aa5a5bf6b03069cac7e.json及.log。
- 本轮Y61模型/YAML和原CPU payload未改，新增的是集成程序及测试证据，不是新候选或模型修订。合成CPU输入和CE+mask-square替代损失不等于稀疏TRAIN原生criterion/optimizer两步；native_data_loaded/native_task_backward/native_smoke_passed均false。Y61保持实际资源阻塞，tick_attempt绑定本轮implementation_test/passed仅CPU范围。没有Git提交、生产同步、准入、入队、训练启动或新完成项。

## 5. Decisions Made
- 继续保护Y58固定30轮，GPU0释放后由既有门控按FIFO派发已准入Y59；不占GPU1、不共卡、不驱逐外部任务、不恢复Y57、不改硬件或调度。
- 优先续做原Y61未测的七阶段连接而非堆积新ID。本轮测试只验证计算图控制，未证明训练收益；Y60原生检查和Y56冻结后评按原计划等待独占GPU0。保留当前HEAD及Y59收据，后续再保存Y61精确版本、绑定并实际执行原生两步后申请准入，常规操作无需重复用户授权。决策pipeline/decisions/20261022-125559_other.md。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
