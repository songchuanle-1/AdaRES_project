# Research Pipeline Report - AdaRES_project
快照时间：2026-10-22T16:50:18.048001+00:00；生成报告时快照年龄：550 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-23 00:50:17
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
================================================== 2026-10-23 00:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y59_frozen_text_mode_res] Best=0.3675 3/30 no_improve=0/30 gpu=0 ? 序列: 0.3400, 0.3675

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y59 | configs_song/model_paper5/adares_y59_frozen_text_mode_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 3；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 113：complete=16 / rejected=79 / closed=12 / blocked=5 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 37（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 00:51:30至00:51:39上海实核Y59原PID28712、命令/cwd及GPU0归属，Train3/30步4520至4543，后续资源采样至4559，持续增长且尾部无错误。前两轮训练日志验证为0.3400429785/0.3675022423，仅训练过程记录，尚无冻结全量效益结论。
- GPU0保持280W，实测84C/22718MiB，热降频再次激活；内核读取成功未见新Xid或fallen-off。GPU1仍有外部负载，未干预。硬件根因未解决，不将sample_healthy=false误报为训练停滞。
- 复核Y60七文件/原生YAML/冻结程序/当前goal/提交6ac一致；Y61七文件仍与原CPU payload一致，完整decoder CPU证据保留。本轮无原生GPU smoke、无新增候选/模型修改/Git提交/准入/排队/启动。唯一获准GPU0忙，未挤占Y59或启用GPU1。
- 旧资源脚本误把已完成Y58的下一步改为训练前检查；原脚本/原始输出保留，未改其哈希绑定证据。通过新生命周期分支核验Y56/Y58 canonical done_complete、epoch30 scheduler与原冻结后评引用，恢复只做后评不重训；Y60保留原生smoke、Y61保留版本准备。新入口及空卡阶段衔接合计10项CPU回归通过，模拟空卡不是实测空卡。
- Y58完成属于前轮已有事实，本轮不重复归档、不重放finalizer；保留其best24/last30及原后评。Y57继续按用户指令暂停，不请求重复授权，不恢复、不迁移。模型/数据/优化器/seed1999/fixed30及调度均未改变。

## 5. Decisions Made
- 下一轮资源复查用scripts/recheck_gpu0_lifecycle_v2.py --token CURRENT_TOKEN；不直接使用旧check_prepared_resources_gpu0_trial.py改写完成实验。新入口内部只让旧采集器处理未准入Y60，再按生命周期分别留待后评、原生smoke和版本准备。
- 继续保护Y59唯一GPU0原训练和280W限制。独占GPU0可用后优先已提交Y60真实原生YAML/TRAIN两步检查，通过即当轮准入并尝试既有门控派发；保留Y56/Y58冻结全量后评及成本比较待办，Y61随后保存精确版本再绑定两步检查。不把CPU通过写成native PASS或训练效益。
- 本轮已有真实资源检查和流程修复证据；不为填满配额再造新候选、不修改快照/预算/禁用项、不延长deadline。按原六段报告和七标题记忆留档后完成canonical finish；收尾结果待实际校验，当前不预称PASS。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
