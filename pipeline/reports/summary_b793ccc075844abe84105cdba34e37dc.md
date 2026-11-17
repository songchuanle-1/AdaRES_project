# Research Pipeline Report - AdaRES_project
快照时间：2026-10-02T14:47:41.992477+00:00；生成报告时快照年龄：91 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-02 22:47:41
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
================================================== 2026-10-02 22:47 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y18_point_measure_res] 进程未见 (deathwatch 2/3) — 暂不处置

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y18 | configs_song/model_paper5/adares_y18_point_measure_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 仅AdaRES_project的event tick；快照北京时间2026-10-02 22:47:41.992477、截止23:47:41。实际CODEX_THREAD_ID与绑定任务一致，closure_policy显式启用，Pipeline-AdaRES_project为Disabled，授权仅server-2 GPU1。begin监控/目标检查各执行一次。
- 本轮脚本事实：原Y18配置训练进程仍未见，canonical deathwatch由1/3推进到2/3，队列仍running；活跃1/运行1/等待0/准入预算0，无新terminalID，无due候选，Y23 blocked_not_due1。没有满足正式done_complete，不重复monitor凑次数。
- 复用上一轮不可变终态/六遍全量评估/场景bootstrap证据，未重复已完成的评估、守卫或GPU探测。Y18训练及冻结评估已完成、mIoU.507605986且目标未达/效力inconclusive属于上一轮证据，不冒充本轮新增产出。22:03 GPU1空闲仅是历史快照，本轮实时占用未知；GPU0未查询/使用。
- 本轮仅更新七段短记忆的当前租约/快照/正式确认计数，以及同一Y23记录的阻塞说明与next_action，保留retry_after23:00、恢复条件、所有哈希及冻结科学证据。未修改正式队列、goal、门禁、训练轮数、SERVERS或调度；无其他项目操作、后台恢复或新任务。
- 执行状态待finish验证；科学状态Y18 complete但效力inconclusive、Y23用户暂停/blocked。实际新增产出为状态说明更新与本轮六段报告；无新增候选、证据测试、准入、训练启动或科学阶段推进。收据精确用量待finish生成，未知不写零；仅统计本地begin至收据窗口，排除此前读取、晚到记录与最终回复，缓存是输入子集、推理是输出子集，不重复累加或作账单。下次准确调度时间未知。

## 5. Decisions Made
- 没有新的科学决策。继续原用户安排：canonical Y18 done_complete后，才用现有守卫在实时独占空闲GPU1恢复同一Y23 epoch4模型/AdamW/PolyLR至epoch5、总30轮；GPU0保留，不调用从头启动的waiting路径。
- 既有依赖未解除且候选未到期，不重复同等探测/已完成评估、不生成填充变体、不等待截止或擅自推进deathwatch。保留既定unblock条件与retry_after，后续正常tick继续监控。
- finish成功并独立核实释放才称本轮COMPLETED；如失败保留证据按原token abort，不改快照、截止或历史证据。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
