# Research Pipeline Report - AdaRES_project
快照时间：2026-10-03T15:51:10.894989+00:00；生成报告时快照年龄：122 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-03 23:51:10
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
================================================== 2026-10-03 23:51 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y25_mask_feedback_res] Best=0.4228 4/30 no_improve=0/30 gpu=1 ? 序列: 0.3396, 0.3591, 0.4228

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
- 本轮 AdaRES_project event 快照为2026-10-03 23:51:10.894989+08，真实截止2026-10-04 00:51:10.894989+08；已核验实际任务归属、foreground配置、冻结目标、closure_policy及项目GPU1白名单，Windows后台任务仍Disabled。首次读取误用根目录SERVERS.yaml路径，随后读取项目实际文件成功，发生在begin之前且未产生项目写入。
- begin已执行一次monitor/goal，未重复preflight；活跃1/运行1/容量1/准入预算0/到期候选0/新终态0。监控原文Best=.4228、4/30、no_improve=0/30；这是阶段性读数，不是30轮训练完成或最终收益。
- 复用已核验SHA256的原只读进度脚本：23:51:35.850308至23:51:41.242148+08，同一Y25 PID98611/config/log/cwd/GPU1/source94918c1核验一致，Train4/30步骤2674->2683/4686。两段各至多24KB尾部未见Traceback/OOM/RuntimeError，仅限采样范围。
- GPU1设备显存22256MiB、利用率100%，不是独立候选峰值；GPU0未查询或使用。科学状态training，最终效力未知，没有执行GPU诊断、重启、调参或新启动。
- 保存pipeline/analyses/y25_progress_b07bd29caafb448ab04d7cbd873f58f6.json，SHA256=e325b7503a2864972aee04df67274312b21b6821dd5d4713caa23456965d4bd8；新增绑定1条进度证据，更新Y25下一动作及七段短期记忆，既有不利结果和冻结设计不变。报告写入时finish待验证，实际执行状态以finish/收据/锁释放核验为准。
- 本轮收据窗口及用量在finish前未知；窗口仅覆盖begin至收据，不含此前读取、晚到记录和最终回复，缓存已经包含在输入、推理已经包含在输出，不重复相加或当账单。上一轮已FINALIZED/PASS，窗口22:48:41至22:55:10，输入1416498（含缓存1334656）/输出8797（含推理210）；实际新候选0/变更候选1/证据引用1/新训练0，产出计数不代表科学收益。

## 5. Decisions Made
- 当前目标容量已满足，无到期可执行候选或新终态，维持同一Y25固定30轮及seed1999；不为了填满tick新增变体，不重复已关闭路线研究，六道准入门槛不放宽。
- 继续保留TRAIN软门控/EVAL硬门控分布偏移、原即时损失上升及数值重复/恢复噪声；阶段性Best=.4228不支持最终收益、伤害、champion或Pareto结论。
- 等待真实30轮终态，按已冻结exp_design_Y25_ed4dc965执行9508表达式/141场景重复Y25/W2/B1比较、whole-scene bootstrap10000/seed1999、冻结TRAIN尺寸切点1487/3325和成本/实质伤害检查；不提前停训。GPU0继续保留，Y18/Y23历史已完成不重复resume；无其他项目或调度操作。下一调度时间未读取，记未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
