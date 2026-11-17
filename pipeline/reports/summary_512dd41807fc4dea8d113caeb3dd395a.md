# Research Pipeline Report - AdaRES_project
快照时间：2026-10-03T16:48:48.250438+00:00；生成报告时快照年龄：219 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-04 00:48:48
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
================================================== 2026-10-04 00:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y25_mask_feedback_res] Best=0.4297 5/30 no_improve=0/30 gpu=1 ? 序列: 0.3396, 0.3591, 0.4228, 0.4297

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
- 本轮仅AdaRES_project，实际归属、前台配置、冻结目标、closure_policy与GPU1白名单核验一致，后台调度Disabled。event快照为2026-10-04 00:48:48.250438+08，截止01:48:48.250438+08；begin/preflight各一次，活跃1/容量1/准入预算0/到期0/新终态0。
- 首个只读Train进度采样失败：远端stdin第38行对应原脚本第30行的两次Train步骤严格递增断言，不是训练Traceback证据。保存原失败log及SHA256=8d0dd58704be381924c71afb29af73317b801f704a898588472f11102a2c9886。失败采样未输出具体两次步骤，不能回填或断言精确故障阶段。
- 新建独立有界phase-aware只读观测脚本capture_y25_phase.py（SHA256=853d4336f13ef10f7e0fc5804db4452a73c257c1f2f434ce61de050589f3b9b7），不覆盖原脚本，不改模型或门槛；观测完成与观察到训练进展分别记录，步骤未变不会伪报进展。
- 实际恢复观测00:50:32.806184至00:50:38.967036+08：原Y25 PID98611/config/log/cwd/GPU1/source94918c1一致，Val5/30步骤363/363转入Train6/30步骤7/4686；日志字节6593895->6596006，两段各至多24KB尾部crash_markers均为空，确实观察到原生阶段进展。GPU1设备22256MiB/利用率53%，不是独立峰值；GPU0未查询或使用。
- 新证据pipeline/analyses/y25_phase_512dd41807fc4dea8d113caeb3dd395a.json，SHA256=d387f295edfb36cca808e4d7f1b610f15137327384a13f1ac1b4d31c7f56be59，连同失败log及新脚本绑定ledger并更新七段短期记忆。科学状态training/最终效力未知，快照监控Best=.4297、5/30/no_improve0为中间读数；表0-3保持begin事实，后续进入第6轮仅记本段。执行状态写入时finish待验证。
- 本轮实际产出为1个失败原始记录、1个阶段感知观测脚本及1份成功双采样证据，新增设计/准入/训练启动均0。本轮收据用量与窗口在finish前未知；上一轮23:51:08至23:53:21输入640751（含缓存625408）/输出2988（含推理47）、新候选0/变化1/新证据引用1。收据窗口不含此前读取、晚到记录或最终回复，缓存和推理不重复相加，不是账单。

## 5. Decisions Made
- 训练正常跨轮，首个监测断言失败已由具体观测方法修复及真实新证据解决；不将短时Train步骤不变当成训练崩溃，不重启原任务，不修改科学准入检查。
- 保持同一Y25固定30轮/seed1999，当前容量满足且无到期候选，不重复已结案路线，不新造变体或GPU诊断。保留原即时loss上升、TRAIN软/EVAL硬分布偏移风险及重复/恢复噪声，不把阶段Best=.4297或进度当成最终收益/Pareto/champion结论。
- 后续有界监测使用capture_y25_phase.py；真实30轮终态后按冻结exp_design_Y25_ed4dc965评估9508表达式/141场景重复Y25/W2/B1、场景bootstrap10000/seed1999、冻结尺寸切点1487/3325和成本/实质伤害。Y18/Y23已完成不重复resume，GPU0继续保留，不操作其他项目、调度或后台。下一调度时间未知，以保存的自动化为准。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
