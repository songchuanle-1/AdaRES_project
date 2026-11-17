# Research Pipeline Report - AdaRES_project
快照时间：2026-10-24T10:49:30.931651+00:00；生成报告时快照年龄：1113 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-24 18:49:30
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
================================================== 2026-10-24 18:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y60_stage_specific_confidence_res] 进程未见 (deathwatch 1/3) — 暂不处置

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y60 | configs_song/model_paper5/adares_y60_stage_specific_confidence_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 3；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 113：complete=16 / rejected=79 / closed=12 / blocked=5 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 38（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 仅AdaRES_project，实际会话归属和Disabled后台已核验；复用本轮lease、experiment_first授权及原快照，保持server-2 GPU0-only/280W和Y57暂停，不改调度、硬件设置或其他项目。
- Y60原PID消失且原生Val30/30完整363/363，完整last/best模型、optimizer、scheduler和日志SHA核验。canonical归档done_complete：last epoch30/optimizer603/scheduler30，best epoch16/mIoU0.5075249075889587。完整日志拉入Checkpoints；pipeline/checks/y60_fixed30_fd874a5eafed4a5b84c6c77d939b6b3f.json及.log留档，不作为冻结后评或晋升。
- Y60归档已提交后末尾断言发现running.json仍指旧Y60而exit1，保留真实失败。专用reconcile_y60_terminal_mapping经6项CPU守卫，实核终态、旧PID消失、无新GPU0主人和Y57暂停后只移除旧映射，不重放归档、不改remote。失败/恢复证据y60_terminal_mapping同token。
- GPU0独占空闲36C/280W后，生命周期实核四项完成终态/原冻结后评及Y61七文件、9bea提交、原生YAML/current goal/冻结program一致；gpu0_lifecycle_v3同token保存，未修改模型/YAML/Git HEAD。
- 18:55:35至18:56:10 Y61首次真实原生YAML/TRAIN/tools.train两步optimizer通过：loss7.886719703674316/8.233489990234375、539梯度张量有限，每步共享头七次、query_norm输入True/False交替，可训练LN梯度非零、mask/query输入梯度有限。两批4场景16表达，峰6599737344B，无验证集/权重保存；pipeline/analyses/y61_smoke_fd874a5eafed4a5b84c6c77d939b6b3f.json及.log，不是效益或全数据显存上界。
- 同轮Y61签准入收据及冻结后评，沿用canonical安全同步和门控派发9bea版本至GPU0，固定30/seed1999、PID78290、launch_verified=true，y61_gpu0_dispatch同token保留实际启动证据。
- 新交接观察器首次在任何SSH/写入前被validation阶段残留exclusive_gpu_slot字段触发schema断言；保留失败，仅接受完成后评的None/exclusive_gpu_slot，经10项CPU守卫后真实观察。19:02:54实核Y61身份/compute/日志，Train1/30步440至465/4686增长，GPU0=280W/77C/16896MiB，无本次热降频、内核Xid/fallen-off或尾部错误。GPU1外部负载不干预，非整机单卡隔离，不宣称硬件根因修复；pipeline/checks/y61_training_handoff_fd874a5eafed4a5b84c6c77d939b6b3f.json。
- 四项Y56/Y58/Y59/Y60完成终态/冻结计划复核后，仅按真实Y61占卡标记后评等待GPU0空槽，原模型/optimizer/scheduler和负结果保留；Y61真实训练确认后清理旧资源字段。更新七段紧凑记忆；无新模型变体/修订/Git提交、无GPU1训练诊断或server-6/调度变化。

## 5. Decisions Made
- Y60已canonical归档，不重放finalizer；原归档末尾旧映射失败以独立有证据的局部恢复处理，保留原失败和hash绑定工具。
- Y61真实原生两步PASS，按现有授权及本轮预算准入并正式派发；保留发布SPFormer、原生YAML、固定30/seed1999、单GPU0/280W，不预证或宣称性能收益。决策及收据见pipeline/admissions/AdaRES-Y61.json。
- Y57继续按用户指令暂停，不用旧许可或心跳代替恢复授权；GPU1外部任务不驱逐、不改设置。
- 后续观察Y61原PID/步数和热风险；四项完成实验仅独占GPU0空槽时执行原冻结全量后评/成本比较，训练best不晋升、不调参。本轮未执行后评。
- 本轮已完成真实原生检查、准入和验证启动，按experiment_first收尾，不为额度强造路线；下轮按实时资源续已有工作。报告不预称postflight成功，finish后只读核验receipt/context/lease/pending，不补写共享状态。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
