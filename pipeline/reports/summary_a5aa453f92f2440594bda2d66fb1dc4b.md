# Research Pipeline Report - AdaRES_project
快照时间：2026-10-21T12:52:43.967375+00:00；生成报告时快照年龄：578 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-21 20:52:43
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
================================================== 2026-10-21 20:52 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y56_unique_competitors_res] Best=0.5110 26/30 no_improve=1/30 gpu=0 ? 序列: 0.5050, 0.5080, 0.5070, 0.5110, 0.5060

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y56 | configs_song/model_paper5/adares_y56_unique_competitors_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 3；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 112：complete=16 / rejected=79 / closed=12 / training=1 / blocked=4。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 35（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 20:52:43上海begin核验实际CODEX会话与项目owner一致、后台调度Disabled；按当前SERVERS.yaml仅GPU0执行。监控确认1项Y56 running、1项Y57 paused、0 waiting，无新终态或派发错误。最新单卡用户指令继续优先，未恢复GPU1。
- 本轮围绕刚开始的单卡试运行执行实际负载观察，而非重复未到期的准备候选资源检查。新增scripts/check_gpu0_single_card_trial.py，复用当前租约与项目锁、仅只读SSH/proc/nvidia-smi；8项CPU控制通过，涵盖进程身份、PID复用、步数未增、GPU1外部占用、未知内核、热降频及功耗变化，Python3.8远端语法也通过。它不是模型或原生训练smoke。
- 20:58:13两次间隔约12秒实测原Y56 PID13738命令/cwd/starttime及GPU0 UUID一致，Train26/30 step3286到3308且日志长度增加，尾部无异常。GPU0原280W保持，温度83至84C、功耗254.45至243.89W；当次SW/HW热降频未激活，内核读取成功且无Xid/fallen-off。证据pipeline/checks/gpu0_single_card_a5aa453f92f2440594bda2d66fb1dc4b.json已绑定Y56及本轮resource_check。训练未重启、迁移或改参。
- 同次实际compute发现GPU1出现PID50448、占用24158MiB，因此检查器的gpu1_compute_empty=false/sample_healthy=false，原记录完整保留；这不是Y56崩溃或新的科学失败，而是不满足整机单卡隔离条件。随后20:59:45仅核验该PID元数据，确认其owner为另一用户、入口是非AdaRES语音训练程序，不是已暂停的Y57；没有读写其他项目文件或干预外部任务。记录pipeline/checks/gpu1_external_occupant_a5aa453f92f2440594bda2d66fb1dc4b.json。
- Y57保持paused/dispatch_hold，完整epoch27检查点、原日志及后续恢复须另获用户指令的条件均未改。Y58/Y59/Y60的原生两步检查仍待独占GPU0；三项20:28资源证据的复查尚未到期，未重复执行、改stage或延长retry。本轮没有新模型、候选、Git提交、准入、入队、生产同步、训练启动、功耗/风扇或调度改动。

## 5. Decisions Made
- 继续保护Y56/GPU0原进程及280W限制，保持项目allowed_gpus=[0]、Y57用户暂停。其他用户在GPU1的任务不属于本项目授权，不停止、迁移、重置或调整它；本项目没有自动补回GPU1。
- 本轮实测可确认AdaRES仅使用GPU0且原生步数继续增长；不能称整机单GPU负载试验，也不能把温度变化归因于停Y57或宣称掉卡根因已修复。sample_healthy=false的具体原因是GPU1外部占用，而不是内核/GPU0热降频故障。
- 如需严格检验整机仅GPU0负载下的散热/掉卡稳定性，需要协调GPU1外部任务的空闲窗口；不默认有驱逐授权。当前Y56可继续，用户不必再次授权普通训练或重复批准Y57暂停。
- 后续GPU0真实空闲时，优先推进已实现Y58/Y59再Y60的原生YAML/TRAIN两步loss/backward及路径/峰值检查，通过才准入/入队并尝试现有门控派发。未把硬件观察或CPU单元测试冒充这些训练检查；不为填满额度制造新候选，不改目标/数据/固定30轮及禁用项。当前报告不预称收尾通过，随后canonical finish并核验收据、context及租约释放。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
