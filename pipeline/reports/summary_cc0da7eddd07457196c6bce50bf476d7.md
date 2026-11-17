# Research Pipeline Report - AdaRES_project
快照时间：2026-10-25T14:51:42.519143+00:00；生成报告时快照年龄：270 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-25 22:51:41
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
================================================== 2026-10-25 22:51 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y62_overlap_density_res] Best=0.3463 2/30 no_improve=0/30 gpu=0 ?

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
- 候选累计记录 115：complete=16 / rejected=79 / closed=12 / blocked=6 / training=1 / admission=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 41（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 快照2026-10-25T14:51:42.519143+00:00：确认实际会话属于AdaRES_project，后台调度保持Disabled；前轮412be收据已FINALIZED/PASS/context_committed=true/lease_release=released，旧审计修复已完成。
- 2026-10-25T14:52:19.965264+00:00使用canonical gpu_occupation.py只读核验server-2；GPU0仍被外来compute PID13813占用，100%利用率、24136/24576MiB，无授权空槽。证据pipeline/checks/gpu_occupation_server-2_cc0da7eddd07457196c6bce50bf476d7.json，SHA256 92f6c1a769e0a7497eed0c8b37d7c00bf28e73dce795c21e7dd1e0d13bdf9293；原始日志保留，已记录resource_check/blocked。
- 上海22:52:22至22:52:27独立确认Y62原PID2540376/server-1 GPU0，第2/30轮324到331/4686；六模型文件及原生YAML语义一致，源码marker82fbd9fb1fc56da6afee5b145f59cfa533464707，bounded尾部无错误。进度证据pipeline/checks/server1_training_progress_cc0da7eddd07457196c6bce50bf476d7.json，SHA256 b1b3d10b5fd4a76dcf116e19ad4d2724f8b3a74fa5385239eeee173c9853eb0c。
- 更新七段紧凑记忆，区分前轮完成的审计修复与本轮常规资源/进度观察；本轮未新增模型、训练检查、Git提交、准入或启动。

## 5. Decisions Made
- Y62继续server-1 GPU0现有训练；Y63已准入并等待同卡，不重复派发运行任务或推进HEAD使Y63现有收据失效。server-2 GPU0维持280W，Y57及GPU1保持用户暂停，禁止server-6并保留外来任务。
- 五项fixed30完成实验继续保留原检查点和不可变冻结后评协议，授权资源被占用时不启动后评、不重训、不重复准入；实测资源阻断按experiment-first规则收尾，不为填预算重复旧检查。
- 当前训练正常推进且队列、资源占用和暂停状态无实质变化；未提取验证主指标，不推断效益或第1轮是否已有验证。用户无需新增操作，本次心跳静默。
- 规范report/finish后核验本轮FINALIZED、postflight PASS、context_committed及lease_release；释放后不再写入项目状态。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
