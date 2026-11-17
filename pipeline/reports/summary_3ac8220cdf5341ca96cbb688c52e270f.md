# Research Pipeline Report - AdaRES_project
快照时间：2026-10-27T04:48:20.805476+00:00；生成报告时快照年龄：120 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-27 12:48:20
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
================================================== 2026-10-27 12:48 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y63_peer_bilinear_res] Best=0.4934 14/30 no_improve=0/30 gpu=0 ? 序列: 0.4790, 0.4891, 0.4889, 0.4906, 0.4934

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
- 核验AdaRES_project实际会话、AGENTS/goal/SERVERS/execution_mode，Pipeline-AdaRES_project保持Disabled。canonical begin为design，running1/waiting0/ready0，无新终态或连接/部署故障。前975b轮收据已核FINALIZED/PASS/context_committed=true/lease_release=released。
- 上海2026-10-27 12:49:02至12:49:07核实Y63/server-1 GPU0同PID3071037，第14/30轮4534至4542/4686步增长。精确cwd/config、六模型源码、原生YAML确定性路径适配、marker82f与GPU0独占compute通过；单次GPU100%/22756MiB/68°C/350W。证据pipeline/checks/y63_server1_progress_3ac8220cdf5341ca96cbb688c52e270f.json。
- canonical gpu_occupation.py生成本轮两服原始hostname/GPU/compute证据：server-1 GPU0仍独占Y63；server-2 GPU0仍外来PID13813、98%/24138MiB，无授权空槽。另次只读观测GPU0为100%/24138MiB/85°C/280W。保留Y57暂停、GPU1限制和外来任务。六项冻结后评与Y64/Y65原生GPU检查继续等待资源。
- 复核Y64已冻结原生计划：当前goal、七文件工作区及提交e278、原训练入口均一致，身份记录pipeline/checks/y64_plan_identity_3ac8220cdf5341ca96cbb688c52e270f.json绑定当前资源证明。未重复旧CPU控制、prepare helper或Git提交；没有原生GPU运行、生产同步、准入或入队。
- 本轮无新候选、源码修改、push、部署、训练启动/重训或效果结论，Y65既有CPU准备保留。已保存本轮resource_check/blocked证据并更新七节记忆；当前仅正常训练推进及未变资源阻塞，无新增异常或用户待办。

## 5. Decisions Made
- 保留已准备的Y64/Y65，当前真实原生下一步依赖独占授权GPU空槽，不为填额度制造重复变体或重复CPU测试，不挤占Y63或外来任务。
- server-2 GPU0空闲后先核当前goal/源/版本/训练入口/数据与资源，再用当前token及明确--gpu --gpu-id 0执行Y64冻结plan。至少两步真实native criterion/backward/optimizer、每步七调用、新梯度及第二步实际响应通过后，冻结原全量后评协议、准入并当轮门控派发。
- 保留原数据协议、seed1999/fixed30/发布SPFormer/禁用项；Y57/server-2 GPU1用户暂停，server-2 GPU0保持280W，禁止server-6、跨项目或改调度。按foreground执行report/finish并核释放，当前未变且无用户可行动事项，心跳静默，释放后不补写。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
