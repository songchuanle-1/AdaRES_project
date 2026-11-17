# Research Pipeline Report - AdaRES_project
快照时间：2026-10-08T08:48:12.245032+00:00；生成报告时快照年龄：167 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-08 16:48:11
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
================================================== 2026-10-08 16:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y31_mask_shape_res] 进程未见 (deathwatch 2/3) — 暂不处置

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y31 | configs_song/model_paper5/adares_y31_mask_shape_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- AdaRES_project本轮事实快照为2026-10-08 16:48:12.245032+08，模式event，截止17:48:12.245032+08未修改。核实实际线程与foreground owner一致、后台Pipeline-AdaRES_project仍Disabled、closure_policy启用；最新SERVERS与用户授权仅GPU1。begin --auto-steady执行一次，先读紧凑视图，monitor/goal均返回0；未重复preflight。D:/project/AGENTS.md不存在，使用已提供的根级用户指令与实际项目AGENTS；不视作closure故障，不创建该文件。
- 脚本监控本轮未见训练进程，Y31规范队列仍running、deathwatch2/3，活跃1/目标1/预算0/到期候选0/新增终态0。event由上轮研究产物与台账变化触发，不代表新增科学结果。未强改队列或deathwatch计数；未查询GPU0或其他项目，未把GPU利用率/显存未知替换为零，也未把上轮16:08:50的空闲观测当作当前资源预约。
- 按ID读取已complete的同一Y31，终态分析SHA256仍为17c5b5d023420e7a516ebd502c744193cdc01abd22e58485c1e1456a89008c25。上轮已完成固定30轮及冻结三模型各两遍、整场景统计、训练后成本和事后审计，本轮不重复这些测试或复审关闭路线。Y31效益不确定、不晋升、未触发冻结伤害淘汰的结论保持；W2历史champion0.5100、目标0.5150、Phase1未达标和Phase2未启动均保持。未改研究台账、证据、准入、目标、门槛或冻结协议。
- 仅更新七段短记忆中的当前快照与上轮成功收尾事实，并复用脚本第0-3段事实表补充本轮动作/无新决策，保存完整六段报告。实际产出仅为一次监控和目标检查、既有结果完整性核对及运行状态记录；新增候选、科学测试、准入和训练启动均无。没有可执行到期候选且预算0，不伪造资源阻塞、设计耗尽或冷却，不等到截止。当前执行状态须以finish及锁释放为准，撰写时尚待核验。
- 当前收据窗口与用量撰写时未知，finish后在任务回复核对；不在租约释放后改写报告。收据仅统计begin至生成前本地可见响应，排除更早读取、晚到用量和最终回复，缓存已含于输入、推理已含于输出，不重复相加，也不是账单。上轮收据window15:48:53.680889至16:29:05.334868，FINALIZED/PASS/context committed且16:29:15.9457881确认锁已释放；仅作为历史记录，不累计成当前用量。

## 5. Decisions Made
- 本轮无新的科学决策。保持Y31 frozen protocol complete/效益不确定/不晋升，complete不等于方法有效、等效或无害；原不利控制与Y30伤害结论保留。等待规范队列按既有确认流程收敛后再按原政策继续独立可证伪诊断，不强改计数、重复preflight或启动新训练。GPU0继续保留，下一触发以既有自动化为准，精确时间本轮未读取。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
