# Research Pipeline Report - AdaRES_project
快照时间：2026-10-22T10:50:53.756615+00:00；生成报告时快照年龄：204 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-22 18:50:53
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
================================================== 2026-10-22 18:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y58_fixed_backbone_statistics_res] Best=0.5034 26/30 no_improve=1/30 gpu=0 ? 序列: 0.5014, 0.5010, 0.4968, 0.5034, 0.4980

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
- 本轮上海18:50:53取得实验优先design快照，实际会话归属匹配，后台Disabled。仅server-2 GPU0获准、280W上限及Y57用户暂停保持；Y58/Y59队列min_epochs=30、patience=30实核，固定30轮不改。
- 18:51:22两次实采核验原Y58 PID77094/命令/cwd/starttime/GPU0归属，Train26/30 step159至183持续增长、尾部无报错。GPU0=280W/84C/22720MiB，本次短采样无热降频、新Xid/fallen-off；GPU1仍有外部compute，未干预。证据pipeline/checks/ef_gpu0_runtime_3ad9da4ad2bb4066903da7ae027a8296.json；本次结果不抹去前轮热降频，也不证明硬件根因已修复。
- 18:51:45到期复查Y60七文件/原生YAML/冻结payload/commit/当前goal一致，唯一授权GPU0仍由Y58占用，无空槽、未执行native smoke。Y58当次Train26 step125至226，前25轮日志best更新为0.5034376382827759，来自epoch24，仅中途指标，不是冻结后评或冠军晋升。资源证据pipeline/checks/prepared_resources_3ad9da4ad2bb4066903da7ae027a8296.json。
- 18:52:13复用v2等待核验入口，Y58/Y59现有准入收据、各7文件及当前HEAD一致；Y59仍waiting/gpu0、没有dispatch_hold/admission_error。Y61各7文件与原CPU不可变payload一致，stage未改，当前resource和attempt证据完整归入候选的断言通过。证据pipeline/checks/ef_waiting_recheck_v2_3ad9da4ad2bb4066903da7ae027a8296.json；tick_attempt=Y59/resource_check/blocked。
- 本轮没有新候选、CPU/GPU模型测试、模型/YAML或操作脚本修改、Git提交、同步、准入、队列新增、训练启动或完成项。Y60/Y61既有CPU测试不重复执行，不冒充原生稀疏TRAIN两步。记录和七段记忆已更新，接下来canonical report/finish，不预称收尾成功。

## 5. Decisions Made
- 沿用既有顺序并保护Y58固定30轮。唯一获准GPU0真实释放后由既有门控按FIFO派发已准入Y59；Y60/Y61原生两步与Y56冻结后评仍须独占GPU0。保持HEAD及有效收据，资源等待不是缺用户授权。
- 本轮无新科学路线决策、无额外用户确认；不改功耗、风扇、训练协议或调度，不恢复Y57或转用GPU1/server-6，不抢占外部任务，不因中途验证指标调整实验。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
