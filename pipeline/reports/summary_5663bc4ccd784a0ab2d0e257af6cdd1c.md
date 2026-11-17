# Research Pipeline Report - AdaRES_project
快照时间：2026-10-04T09:48:58.669361+00:00；生成报告时快照年龄：1100 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-04 17:48:58
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
================================================== 2026-10-04 17:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y25_mask_feedback_res] 进程未见 (deathwatch 1/3) — 暂不处置

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
- 本轮event快照北京时间2026-10-04 17:48:58.669361，原截止18:48:58.669361；任务归属、GPU1唯一授权和后台禁用已核实，仅一次begin/preflight。表0–3为该时刻快照，不能冒充后续终态。
- 17:50:45至17:50:48核实原Y25 PID98611及同配置进程均不存在，最后Val30/30达到363/363，last meta30/best meta16、model/optimizer/scheduler及SHA均核实且读取未变化；完整30轮训练已实际结束，流水线仍为deathwatch1/3，不手工改变生命周期。
- 17:50:59至18:01:37完成Y25/W2/B1各两遍原冻结全量native评估，均9508表达式/141场景、身份对齐、seed1999及TRAIN尺寸切点1487/3325；Y25 mIoU .494117094/.494020751，W2 .510107995/.510007015，B1 .502121793/.502223014。
- 18:01:49完成10000次整场景bootstrap：Y25相对W2差值-.015990902，95%CI[-.023798195,-.008143030]；总体/小/中目标满足冻结损害规则，大目标跨零。相对B1总体CI[-.016802654,.000638898]且各尺寸区间均跨零，不宣称可靠损害B1。大幅逐表达式重复波动保留，两遍不是通用噪声界。
- 18:01:48至18:02:51完成冻结TRAIN8及最大场景728/6780串行native batch1成本采集，各2warmups/3readings：Y25/W2参数164629637、查询128/选中1，最大场景allocated1.243689/reserved1.757813GiB相同；train8 median .075167709/.077175695s、描述性p95 .117561844/.117949866s，最大场景median .191020748/.189681914s、描述性p95 .225458593/.224446009s。三次计时不能证明尾延迟或Pareto优势。
- 四项重复噪声统计控制测试、四个适配脚本语法检查通过；终态、三个全量评估、聚类分析、成本及脚本哈希写入同一Y25记录，并保存终态分析及七段记忆。未改原模型、checkpoint、冻结设计、目标或门槛；新候选0/新训练派发0/重启0。
- 18:03:20只读确认hostname dl01-MS-7C60及GPU1 1MiB/util0/无computePID；GPU0未查询或使用。科学状态rejected（有害于W2的此实现），非崩溃、非全面机制否定。报告写入时执行postflight尚待验证；当前收据用量未知，finish后读取begin至receipt部分窗口，缓存包含在输入、推理包含在输出，不重复相加，不作为账单。

## 5. Decisions Made
- 拒绝Y25这次固定30轮实现：目标.515未达，整体和小/中目标相对W2退化越过冻结场景级损害规则；不晋升champion、不调参或重启、不改名重开。TRAINsoft/EVALhard错配是风险但不是已隔离证明的唯一原因；相对B1及大目标差异仍不确定。
- 保留不可变快照的active1/budget0/deathwatch1/3，等待下一正常tick生命周期确认，不重复preflight、不伪造done_complete或填充新变体；下次合法容量空缺研究须基于独立证据，不重审此关闭实现。
- GPU0继续用户保留，额外测试均仅GPU1串行且先校验空闲/独占租约；本轮原冻结训练后验证工作已实际完成，生成六段报告后调用finish，只有通过且独立确认锁消失才称本轮完成。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
