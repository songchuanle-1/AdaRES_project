# Research Pipeline Report - AdaRES_project
快照时间：2026-10-26T04:44:54.030818+00:00；生成报告时快照年龄：49 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-26 12:44:53
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
================================================== 2026-10-26 12:44 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y62_overlap_density_res] Best=0.5058 17/30 no_improve=2/30 gpu=0 ? 序列: 0.4952, 0.4981, 0.5058, 0.4982, 0.5038

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
  server-1: running=1 waiting=1
    待推: [AdaRES-Y63] configs_song/model_paper5/adares_y63_peer_bilinear_res.yaml gpu=0
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y62 | configs_song/model_paper5/adares_y62_overlap_density_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 1 | AdaRES-Y63 | configs_song/model_paper5/adares_y63_peer_bilinear_res.yaml | waiting | - | server-1 |
- 活跃 2（运行 1 / 认领 0 / 等待 1）；目标 4；准入预算 2。
- 研究储备 1 / 3；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 1；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 116：complete=16 / rejected=79 / closed=12 / blocked=7 / training=1 / admission=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 41（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 已读取当前前台/实验优先规则并核实际会话所属、资源和Disabled后台。快照2026-10-26T04:44:54.030818+00:00，canonical begin先查进展及检查既有派发，running1/waiting1/ready1，无新终态或运行故障；前86eb轮收据FINALIZED/PASS/committed/released。
- 上海12:45:06至12:45:11确认Y62原PID2540376/server-1 GPU0，第17/30轮4328至4336/4686增长，六文件/原生YAML/marker一致，bounded尾部无错误。证据pipeline/checks/server1_training_progress_bdb21d5e51a643f98fe17b3d918c452d.json，SHA256 0cad3bc5f83482b8e6ea4e5d019e61b876f9f6776908bb2a0ab5b442208b9be7。
- 于2026-10-26T04:45:04.230762+00:00canonical实测server-2 GPU0被外来PID13813占用，100%/24138MiB，无授权空槽。证据pipeline/checks/gpu_occupation_server-2_bdb21d5e51a643f98fe17b3d918c452d.json，SHA256 0f70a4bf1e2708fa5c6bbb9cc55e831a6b7b5fe5219deedf7d36ee10855f8a67，绑定五项后评及Y64并保存resource_check/blocked；更新七段记忆。

## 5. Decisions Made
- 继续Y62现有训练和Y63同卡waiting；保留Y64已完成CPU准备，授权独占空槽出现后再完成源manifest/限定版本/原生计划及真实两步TRAIN检查，通过才准入与门控派发。本轮无新增变体、重复CPU测试、提交、同步、准入或启动。
- server-2 GPU0保持280W，Y57/GPU1保持用户暂停，禁止server-6和干扰外来任务，不改调度。五项fixed30完成实验按既有冻结后评协议等待资源；未提取验证指标，不把正常步数增长当成训练收益。
- 当前状态仍为正常训练与资源等待，无需新增用户操作，心跳静默。完整六段报告生成后finish并核FINALIZED/PASS/committed/released及锁/pending缺失，释放后不补写。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
