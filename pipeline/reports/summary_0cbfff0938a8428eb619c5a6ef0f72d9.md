# Research Pipeline Report - AdaRES_project
快照时间：2026-10-21T22:47:23.052117+00:00；生成报告时快照年龄：185 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-22 06:47:22
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
================================================== 2026-10-22 06:47 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y58_fixed_backbone_statistics_res] Best=0.4800 9/30 no_improve=0/30 gpu=0 ? 序列: 0.4309, 0.4440, 0.4647, 0.4601, 0.4800

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
- 候选累计记录 112：complete=16 / rejected=79 / closed=12 / blocked=3 / training=1 / deployment=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 37（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 06:47:23上海begin核验实际会话所属、后台Disabled、当前goal和GPU0-only资源限制，先监控并尝试已有队列。Y58运行/Y59合法FIFO等待/Y57暂停，未见连接、准入或派发故障；无新增终态。
- 06:48:39复用现有只读观察器，实核Y58原PID77094命令/cwd/starttime/GPU0归属及Train9/30 step368到389递增、日志增长。GPU0仍280W/83C/22718MiB，本次未见热降频、尾部异常、新Xid/fallen-off。GPU1外部compute仍在，未干预；观察器sample_healthy=false仅说明不满足整机单卡隔离，不是Y58失败。证据pipeline/checks/ef_gpu0_runtime_0cbfff0938a8428eb619c5a6ef0f72d9.json。
- 项目锁/当前租约下对Y58/Y59各7文件、冻结payload、原生YAML、Git提交、current goal及实际准入收据一致性复核通过。GPU0仍被已确认推进的Y58占用，Y59不能并跑，继续waiting；resource_check/blocked和当前证据绑定Y59，保留deployment阶段。证据pipeline/checks/ef_waiting_recheck_0cbfff0938a8428eb619c5a6ef0f72d9.json。
- Y57人工复查到期但没有新用户恢复指令，保持暂停、检查点和日志，不把计时器当授权或重复索权。Y56既有终态/冻结后评及Y60原生检查的原条件不变。本轮无新候选、模型/YAML/工具修改、Git提交、原生smoke、准入、同步、训练启动、硬件或调度变化；仅状态/资源检查，没有新的研究结论或需处理事项。

## 5. Decisions Made
- 继续Y58原固定30轮训练及GPU0的280W限制，不抢占或重启；Y59在GPU0真实开槽后由既有门控接续。不扩展GPU1授权、不恢复Y57或干预外部任务。
- Y60原生测试与Y56全量冻结后评需要独占授权GPU0；保留现有实现和计划，不以CPU/资源核验冒充smoke或科学收益，不为凑待跑目标增加占位变体。
- 83C且本次无热降频仍仅是短采样，不宣称长期稳定或硬件修复。无实质状态变化及可行动的新事项，不重复发送研究简报。按canonical report/finish核验context提交和租约释放，释放后不补写。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
