# Research Pipeline Report - AdaRES_project
快照时间：2026-10-17T11:50:39.721544+00:00；生成报告时快照年龄：147 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-17 19:50:39
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
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-17 19:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2:
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| 无 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 快照内无运行任务 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 2；准入预算 2。
- 研究储备 2 / 2；储备准备预算 0，不增加训练并发或启动授权。
- 候选累计记录 107：complete=14 / rejected=79 / closed=12 / blocked=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 30（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮快照2026-10-17 19:50:39上海时间，design。归属任务、foreground、goal、仅server-2 GPU0/1单实验单卡授权和后台Disabled已核实；begin仅执行一次monitor/goal，返回码均0。紧凑视图项目运行/认领/排队均0、无新终态；GPU实时利用率、显存、外来作业未知，不据此分配。
- due0、blocked_not_due2、储备2/2、准备预算0。Y52/Y54既有retry_after分别为2026-10-17T20:53:10.188457+08:00及2026-10-17T21:53:58.735844+08:00，本轮均未到期；未读取两者许可文件或重复候选查询、原生/资源/运维测试，未修改retry/wake。
- 19:51:30只读核实账本SHA256仍d5bef086d1949b149c90e97b40f7046b00109c561fcc9e6f7573398b3313bbc6，与上轮相同；canonical脚本SHA仍a78da0060b280300359adc58a20f6c325cabb487313492112e8fc0bf4535bee6。复用14:52:55已报告冷却反例，证据closed_stage_cooldown_regression_c15addc5f96f406593c63f1dd9aa11f0.json的哈希仍6203b9d508bae2b43f391a3b0d9b981a2553289eeb9bf3ebb56ca3d076060aff。未重跑回归，未把缓存当新科学证据。
- 本轮只完成状态核对，没有新的研究结论。新科学候选/测试、模型/配置变化、修订、Git提交、准入和训练启动均0；历史CSV最佳Y36 mIoU 0.5110、基线B1 0.5020、目标0.515，差0.0040，无新训练或评估指标。Y52/Y54训练收益未知。
- 原研究账本、design_cycle、历史closed/rejected、共享程序、目标、资源和调度完全保留。复用既有20261017-145336_question.md故障决策，不新增同义问题或催问；共享冷却缺陷仍未修复，17:58成功收尾不代表它已修复。
- 本轮未调用finish，不重复同一不变状态的已知失败校验。缓存14:55:58精确错误为CLOSURE_INCOMPLETE: Capacity-gap design made no durable candidate progress; create a stable diagnosis candidate and run its next test before stopping，见closed_stage_postflight_failure_c15addc5f96f406593c63f1dd9aa11f0.json；这不是本轮新postflight或PASS。完整六段审计和七节记忆留档后直接abort当前token，按INCOMPLETE保留，实际释放以命令/收据/独立核验为准，不称完成。
- 收据用量仅本轮begin至收据前的本地可见响应窗口，实际token在收据前未知；不含更早阅读、迟到记录、最终答复，不是账单。缓存已含input、推理已含output，不重复相加。本轮实际产出是一次规范监控/目标/状态核对与审计，不是新科学实验或训练成果。

## 5. Decisions Made
- 没有新到期工作或许可变化，不将旧授权依赖/冷却故障包装为新科学diagnosis、不凑变体/第三个储备、不改名重开旧路线或借用上轮进展。无真实独立新测试，不能宣称路线穷尽、无信息增益早停或deadline接近。
- 按已知校验与不重复等价探针规则，保留缓存精确故障并直接abort本轮租约，不重复finish、不硬撑截止或为过检查修改历史/准入。只有finish真实成功才称完成，本轮不会如此宣称。
- 仅wake证据改变或retry到期后续做原稳定ID，可复用已冻结recheck_candidate_commit_dependency.py处理到期提交依赖；候选限定提交许可、共享修复许可和GPU授权分开，不扩展Y50旧许可，不跨项目或改调度/后台。
- 本轮六段与七节记录仅在当前租约内写入，abort后只读核实foreground/pending消失，不进行释放后写入。既有阻塞无实质变化，无新增研究结果或新需确认事项，保持静默。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
