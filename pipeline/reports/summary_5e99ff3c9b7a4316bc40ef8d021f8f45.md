# Research Pipeline Report - AdaRES_project
快照时间：2026-10-15T15:48:50.704714+00:00；生成报告时快照年龄：135 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-15 23:48:50
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
================================================== 2026-10-15 23:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y48_mask_equivalence_res] Best=0.5034 16/30 no_improve=1/30 gpu=1 ? 序列: 0.4944, 0.4936, 0.4976, 0.5034, 0.5027

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y48 | configs_song/model_paper5/adares_y48_mask_equivalence_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。
- 研究储备 2 / 2；储备准备预算 0，不增加训练并发或启动授权。
- 候选累计记录 101：complete=14 / rejected=72 / closed=12 / blocked=2 / training=1。
- 否决范围：unclassified=72；实现修订累计 1；收据文件 29（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮快照2026-10-15T15:48:50.704714+00:00（本地23:48:50），真实截止2026-10-15T16:48:50.704714+00:00；begin/preflight各一次，0-3段沿用脚本事实表；finish尚未执行，执行状态待规范收尾确认。
- 规范begin读取原Y48日志：第16/30轮，阶段best mIoU仍0.5034，最近验证0.5027、一次未刷新。新证据是实际训练日志推进，不是新科学测试；无终态、冻结全量比较或可靠baseline提升结论，历史champion与0.515目标不变。未新增双时间点步数/源码哈希/GPU占用测试，Acc@50/spmIoU等未知。
- 本次处置2026-10-15T23:50:50.692462+08:00早于两候选原资源retry 2026-10-15T23:52:44.904536+08:00。产物变化导致的紧凑due列表不是新目标/服务器/资源队列或解除阻塞证据；保留Y46/Y49全部字段、证据数组、原retry和冻结测试，不提前重复探针、不延后重查或等待截止。
- 最新完整GPU现场证据为2026-10-15 22:52:38至22:52:44+08：本任务Y48占GPU1/PID8877，其他任务占GPU0/PID19723，当时空闲0。本轮当前GPU0占用/显存/利用率和可用卡数未重新采集，均未知，不把缓存当实时；GPU1训练归属见当前begin摘要。
- 科学状态Y48=training_not_efficacy、Y46/Y49=blocked；储备2/2、准备预算0。保留Y46同ID修订1的生产原生/实际训练更新/完整恢复/资源待办和成本风险，保留Y49同冻结native测试与原生同component分区差异见证为0等限制，无新的native/resource/efficacy PASS。
- 实际产出1份本轮monitor观察与有界延期证据（仅绑定原Y48）、7节短记忆、完整六段报告；新候选0、准入0、新训练0、资源重查0、GPU诊断0、重复CPU控制0。attempted_ids仅实际观察的新Y48日志，Y46/Y49标为deferred，不把延期或文档冒充科学进展。
- 沿用既有外部阻塞的有界延期收尾，原retry或真实变化后检查现场空闲并获取独占诊断租约再执行同ID冻结GPU测试。下一保存自动化时刻未知。收据尚未生成，用量/调用数暂未知；窗口仅begin至收据本地可见记录，不含早期读取/最终回复/迟到记录，缓存含在输入、推理含在输出，不重复加总或当账单；finish后只读，不在释放租约后回写。

## 5. Decisions Made
- 无新科学结论或设计；未到原retry不因上一轮自身产物唤醒而提前重复资源测试，保留原候选和原门槛，不凑变体、造路线穷尽或等待耗尽截止。
- 实际空卡先续Y46同ID修订1生产原生路径与Y49同冻结native程序，完整验证后决定准入；授权、计时器或逻辑空位不是空闲或独占证据。
- Y48保持released-SPFormer/seed1999/单卡/固定30轮及冻结终态9508表达、141场景、每方法两次FIRST主结果、W2/B1配对及子组风险评价，不据早期验证波动改协议或晋升。
- 仅AdaRES_project/server2 GPU0、1；后台Pipeline-AdaRES_project仍Disabled，未碰其他项目、server6、调度、新任务或嵌套执行。
- finish明确PASS/COMPLETE且独立确认租约消失才称COMPLETED；若不接受真实延期，保留证据并abort，不伪造到期测试或改检查器。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
