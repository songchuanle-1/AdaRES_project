# Research Pipeline Report - AdaRES_project
快照时间：2026-10-03T14:48:43.959574+00:00；生成报告时快照年龄：373 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-03 22:48:43
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-W2 — mIoU=0.5100 (gap to target: +0.0050)
       Epochs: 30/30, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0080

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5100, target=0.5150, gap=+0.0050
================================================================================
```

## 1. GPU Overview
- 仅列授权卡；未采集利用率、显存或进程时标为未知，不推断空闲。
| Server | GPU | Util | Memory | Process |
|---|---|---|---|---|
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-03 22:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y25_mask_feedback_res] Best=0.3396 2/30 no_improve=0/30 gpu=1 ?

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y25 | configs_song/model_paper5/adares_y25_mask_feedback_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 本轮仅处理 AdaRES_project 的 event tick；快照 2026-10-03 22:48:43.959574+08，截止 23:48:43.959574+08；归属、冻结协议、closure_policy、GPU1 白名单及后台调度 Disabled 已核实，begin 与 preflight 各执行一次。
- 复用已核验 SHA256 的只读 capture_y25_launch.py，经 run_foreground_probe.py 保存两次有界采样：22:49:23.856870 至 22:49:29.308937+08，Y25 原 PID98611/config/log/cwd/GPU1/source94918c1 均核验一致，Train3/30 原生步骤 27->37/4686。
- GPU1 本次设备占用 22254MiB、利用率 52%，不是独立候选峰值；仅在两段各至多24KB日志尾部未见 Traceback/OOM/RuntimeError，未声明全日志无故障。GPU0 未查询、未使用。
- 证据 pipeline/analyses/y25_progress_999ec32f451543fa9fad0f31fbe9eb19.json 的 SHA256 为 adc9f5209b6b912aaf2aea49fb921dafd80406492cfc1d9589039e0abd3556aa；已绑定 Y25 ledger，更新七段短期记忆，保留早期进度、对照及不利结果。
- 事实表0-3仍对应 begin 快照：监控原文 epoch2/30、Best=.3396、no_improve=0/30；后续采样已进入 epoch3，这不是矛盾或终态。科学状态 training、最终效力未知；报告写入时 finish 尚待验证，不能预称完成。
- 本轮实际产出为一次双采样进度证据、Y25下一动作及记忆更新；新设计、准入、启动均为0。收据用量及窗口在 finish 前未知，最终以收据与命令/锁状态为准；窗口不含此前读取及最终回复，缓存含于输入、推理含于输出，不能重复相加或当账单。

## 5. Decisions Made
- 当前容量1、活跃1、准入预算0、到期候选0，无新终态；保持同一Y25固定30轮/seed1999，不重启、不调参、不新增凑数变体，也不重开已结案路线。
- 阶段性验证 .3396 不能证明最终收益或伤害；六道准入门槛既有证据不变，保留训练软门控/推理硬门控分布偏移及对照损失上升风险，不宣称新champion、Pareto优势或新颖性结论。
- 等待真实固定30轮终态后执行冻结 exp_design_Y25_ed4dc965 的9508表达式/141场景重复Y25/W2/B1评估、场景bootstrap及尺寸组/成本/实质伤害分析；GPU0继续保留，不操作其他项目或调度。下一调度时间未知，以已保存自动化为准。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
