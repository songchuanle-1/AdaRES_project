# Research Pipeline Report - AdaRES_project
快照时间：2026-10-28T06:45:02.486000+00:00；生成报告时快照年龄：2178 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-28 14:45:01
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
================================================== 2026-10-28 14:44 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y64_directed_containment_res] Best=0.4764 9/30 no_improve=1/30 gpu=0 ? 序列: 0.4462, 0.4615, 0.4683, 0.4764, 0.4735

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
| AdaRES-Y64 | configs_song/model_paper5/adares_y64_directed_containment_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 3；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 118：complete=16 / rejected=79 / closed=12 / blocked=10 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 42（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 按实际会话归属核Disabled旧调度/前轮已释放并canonical begin；本轮无新终态或operational issue，队列无waiting/claiming。
- 上海15:20:19至15:20:24确认Y64在server-1 GPU0原PID2752219第10/30轮2855至2863/4686，六源码/原生YAML/9706/原准入和独占通过。
- canonical物理占用证明绑定本轮token/snapshot：server-1 GPU0为Y64并已预留，server-2 GPU0为未登记13813/24138MiB，授权GPU均忙，已记录resource_check/blocked。

## 5. Decisions Made
- Y65/Y66已有CPU测试、限定保存版本和原生计划，下一缺项需真实GPU；准备数量足够，不凑重复候选/重复等价CPU检查，不以CPU PASS代签原生两步。
- 真实空槽后优先Y65再Y66，核当前goal/FROZEN/七文件/HEAD882d后原生两步，实际PASS才冻结原后评/准入/入队并当轮派发；跨server-1须合法路径/runtime/350W绑定。
- 保留Y64原运行、七项全量冻结后评、Y57/GPU1暂停、server-2 GPU0 280W与禁用项，不干预外部任务/改变数据或调度。本轮仅运行和资源核验，无新增科研结论、启动或用户待办。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
