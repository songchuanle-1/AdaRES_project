# Research Pipeline Report - AdaRES_project
快照时间：2026-10-04T08:47:57.368014+00:00；生成报告时快照年龄：119 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-04 16:47:57
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
================================================== 2026-10-04 16:47 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y25_mask_feedback_res] Best=0.4941 29/30 no_improve=12/30 gpu=1 ? 序列: 0.4864, 0.4867, 0.4884, 0.4880, 0.4864

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
- 本轮event快照北京时间2026-10-04 16:47:57.368014，原截止17:47:57.368014；已核实实际任务归属、GPU1授权和禁用后台，begin/preflight仅一次。
- 复用阶段感知脚本853d4336，于16:48:20.615777至16:48:26.082788核实原Y25 PID98611/config/log/cwd/GPU1/source94918c1，Train29/30步数3882→3892/4686，日志37976641→37979661字节，实际训练推进。GPU1设备22260MiB、利用率66%，非隔离峰值；两个24KB日志尾无崩溃标记，不能推断完整历史。
- 保存新证据pipeline/analyses/y25_phase_92fe7fbc0d2040cca59eb338e4b9645b.json，SHA256 a6ab16e2f43b531690c828805aa1e8ef26b2fc09def43769ae13aa40bd232d54；更新同一Y25下一步及七段记忆，保留历史失败、反向对照和冻结设计，未操作其他项目。
- 科学状态TRAINING，最终效力与Pareto未知；此前六门准入不重复验证。新增候选0、新训练派发0、重启0，GPU0未查询或使用。收据用量在报告写入时未知，finish后读取本地begin至receipt部分窗口；缓存已包含在输入、推理已包含在输出，不重复相加，不作为账单。postflight尚待执行，不预称本轮完成。

## 5. Decisions Made
- 容量1已满、活跃目标满足，预算0/到期0/新终态0；保持原固定30轮seed1999，不等待整个训练来耗尽本轮时间，不重启、调参、追加准入或复查已关闭路线。
- best .4941/latest .4864仅为中途验证，不能替代最终效力判定；真实终态后按冻结方案完成9508表达式/141场景Y25/W2/B1各两遍、10000次整场景bootstrap、固定TRAIN尺寸分组及成本比较。
- GPU0继续用户保留，GPU1占用时不额外开展GPU测试；生成六段报告并验证收尾，完成状态必须由finish成功及独立确认锁消失共同证明。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
