# Research Pipeline Report - AdaRES_project
快照时间：2026-10-17T07:49:21.910861+00:00；生成报告时快照年龄：175 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-17 15:49:21
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
================================================== 2026-10-17 15:49 : AdaRES_project ==================================================

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
- 本轮快照2026-10-17 15:49:21上海时间，design；归属任务ID、foreground模式、当前目标、server-2 GPU0/1单实验单卡授权及后台Disabled均已核实。begin仅执行一次monitor/goal，返回码均0；项目运行/认领/排队均0，无新终态。GPU实时利用率、显存及外来作业未知，不据此派发。
- 紧凑视图due0、blocked_not_due2、储备2/2、储备准备预算0。Y52/Y54既有提交依赖未到期，未重新读取候选完整证据或授权文件、未重跑原生/资源/诊断门禁、不改retry/wake。
- 15:50:06上海时间仅做字节核对：共享foreground_tick.py SHA256 a78da0060b280300359adc58a20f6c325cabb487313492112e8fc0bf4535bee6与上轮相同，研究账本也与上轮回归时哈希相同。复用2026-10-17 14:52:55已实测运维反例closed_stage_cooldown_regression_c15addc5f96f406593c63f1dd9aa11f0.json，证据哈希6203b9d508bae2b43f391a3b0d9b981a2553289eeb9bf3ebb56ca3d076060aff；本轮未重跑该回归，不把缓存算作新成果。
- 已知故障仍未解除：12条历史closed被共享冷却判断计入pending，使两项真实human_authorization阻塞不能保存冷却；原研究账本、历史closed、共享程序、调度、目标和准入规则均未修改。既有20261017-145336_question.md已经提出限定修复请求，本轮不重复催问或新增同义决策。
- 本轮只完成状态核对，没有新的研究结论。新科学候选、科学测试、模型/配置改动、修订、Git提交、准入与训练启动均0；没有新训练或评估mIoU。历史CSV最佳Y36 0.5110、基线B1 0.5020、目标0.515，差0.0040，均非本轮新增结果；Y52/Y54训练收益未知。
- 本轮不重复调用已被同一不变条件阻断的finish。上轮14:55:58缓存精确错误为CLOSURE_INCOMPLETE: Capacity-gap design made no durable candidate progress; create a stable diagnosis candidate and run its next test before stopping，见closed_stage_postflight_failure_c15addc5f96f406593c63f1dd9aa11f0.json。这不是本轮新执行的postflight错误，也不是PASS；本轮拟按INCOMPLETE保留审计并abort，实际释放须以本token命令、收据和独立核验为准。
- 收据用量窗口仅本轮begin至收据前的本地可见响应记录，实际token计数在收据前未知；排除更早阅读、迟到记录与最终答复，不是账单。缓存已含input、推理已含output，不重复相加。实际产出是本轮状态/哈希核对和完整审计留档，不是新增研究或实验成功数。

## 5. Decisions Made
- 遵守不重复等价探针和不凑变体：不把未变化的提交许可依赖或运维故障包装成新的科学diagnosis，不改名重开旧路线、不额外准备第三个训练原型，也不把上轮研究进展当作本轮候选变化。
- 已知校验无法接受当前真实阻塞状态且源码/账本均未改变，按foreground契约保留此前精确失败并直接abort当前拥有的token，不重复finish、不等待截止、不降低准入或修改历史。未执行新finish就不能称本轮完成。
- 仅在已有wake证据改变或retry到期后续做原稳定Y52/Y54；限定提交许可与共享脚本修复许可互不替代，Y50许可不扩大。共享修复仍待既有明确请求的答复，本轮不触碰其它项目、调度或后台。
- 完整六段报告和七节记忆在当前租约内写入，随后abort并只读核实foreground及pending消失；释放后不再写项目。因已报告阻塞未变化，无新的研究结果或需重复提示的用户决定，会话保持静默。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
