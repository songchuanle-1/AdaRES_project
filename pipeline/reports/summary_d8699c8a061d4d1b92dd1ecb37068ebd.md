# Research Pipeline Report - AdaRES_project
快照时间：2026-10-21T18:53:28.719161+00:00；生成报告时快照年龄：207 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-22 02:53:28
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
================================================== 2026-10-22 02:53 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y58_fixed_backbone_statistics_res] Best=0.3383 3/30 no_improve=0/30 gpu=0 ? 序列: 0.3223, 0.3383

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
- 02:53:28上海begin已核验实际会话所属、后台Disabled和最新GPU0-only条款。监控现有Y58运行/Y59 waiting/Y57 paused，无派发或连接故障；快照列出Y56终态是前轮已核验归档的结果，本轮不重复finalize或将它计作新完成。
- 02:54:21复用既有capture_ef_gpu0_runtime.py，只读两次实测Y58原PID77094/命令/cwd/starttime/GPU0归属及Train3/30 step2678至2701增长，尾部无异常。GPU0仍280W/83C/22718MiB，当次无热降频、新Xid或fallen-off；GPU1外部compute仍在。证据pipeline/checks/ef_gpu0_runtime_d8699c8a061d4d1b92dd1ecb37068ebd.json，sample_healthy=false仅因GPU1非空，不是Y58故障。
- 按当前租约和项目锁，对Y58/Y59各7文件、冻结payload、原生YAML、Git提交、current goal及现有真实准入收据做实际一致性复核，全部通过。Y59已有合法FIFO等待任务，但唯一授权GPU0被已验证步数增长的Y58占用，不能启动第二实验或借GPU1；未重复原生smoke。当前resource_check/blocked与证据绑定Y59，保留deployment/waiting，不伪造授权缺口或资源PASS。证据pipeline/checks/ef_waiting_recheck_d8699c8a061d4d1b92dd1ecb37068ebd.json。
- Y56冻结后评和Y60原生检查的既有资源条件保留；Y57继续用户暂停、epoch27检查点/日志及另获恢复指令条件不变，不重复索权。本轮没有模型/YAML/测试程序修改、新候选、Git提交、新准入、同步、训练启动、硬件设置或调度变化；只完成状态与资源检查，没有新的研究结论。

## 5. Decisions Made
- 保持原Y58固定30轮训练和280W资源条件，Y59继续合法FIFO等待，既有门控在GPU0真实开槽时再派发；不为补齐待跑目标制造占位候选、挤占运行任务或把未测Y60直接准入。
- 继续仅server-2 GPU0，不恢复GPU1/Y57，不干预外部任务。当前83C无热降频只是短采样，不能认定历史硬件问题解决；无新增需用户处理或授权的事项，未扩大操作权限。
- 保留已有准入、原生负结果和冻结数据/seed/固定30轮协议。资源检查不冒充新科学成果；本轮无实质状态变化，面向用户不重复发布研究结论。随后canonical report/finish并核验收据/context提交及租约释放，不预称成功，释放后不补写。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
