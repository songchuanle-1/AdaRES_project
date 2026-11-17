# Research Pipeline Report - AdaRES_project
快照时间：2026-10-15T09:48:48.163522+00:00；生成报告时快照年龄：209 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-15 17:48:47
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
================================================== 2026-10-15 17:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y48_mask_equivalence_res] Best=0.4696 9/30 no_improve=2/30 gpu=1 ? 序列: 0.4384, 0.4623, 0.4696, 0.4659, 0.4668

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
- 本轮快照2026-10-15T09:48:48.163522+00:00（本地17:48:48），真实截止2026-10-15T10:48:48.163522+00:00；begin/preflight各一次，0-3段复用脚本事实表。报告保存时finish未执行，执行状态待规范收尾确认。
- Y46和Y49原复查时间16:58:28+08已过，先按稳定ID续做，两者共享一次真实资源复查；研究储备2/2、准备预算0、准入预算1，不因逻辑队列空位增加候选或降低门槛。
- 实际现场窗口2026-10-15T17:50:25.701998+08:00至2026-10-15T17:50:32.713633+08:00：原Y48/PID8877/cwd/配置/argvGPU1成立，7项源码/配置哈希与原准入提交逐项相同；原生第9/30轮步数569至578/4686，训练持续推进，未重启或迁移。
- 现场授权2张卡：本任务Y48占GPU1（PID8877，65%，22718/24576MiB），其他任务占GPU0（PID19723，100%，23482/24576MiB），空闲0。未分配诊断CUDA上下文、未中断或接管其他任务；0-3段GPU未知值属于较早preflight，本段标明稍后现场证据。
- begin监控日志显示Y48第9/30轮、阶段best mIoU=0.4696、最近两次未刷新；这是未完成训练的学习曲线，不是终态提升或历史champion变更。目标0.515未达成，不据验证波动调参、早停或重启；Acc@50/spmIoU等未新增采集，未知。
- Y46修订1及Y49的冻结下一程序/清单哈希均未变，语法编译通过但GPU测试未运行。生产native backward/tools.train/恢复/资源等未通过项保持待办；保留Y46运行成本风险和Y49原生同component分区差异见证为0等限制，资源复查不是科学PASS。
- 科学状态Y48=training_not_efficacy、Y46/Y49=blocked。按现场外部占用以external_resource_blocked收尾，下次有界复查2026-10-15T18:51:59.461206+08:00；实际资源/目标/证据变化可提前唤醒，仍须现场空闲和独占诊断租约。
- 本轮实际产出1次共享资源/训练进度检查、1份哈希绑定的双候选阻塞处置、候选证据及7节短记忆更新、完整六段报告；新候选0、准入0、新训练0、GPU诊断0、实现修订0，没有重复CPU对照或终态评价。
- 收据尚未生成，用量/调用数暂未知；仅统计begin至收据的本地可见窗口，不含早期读取、最终回复和迟到记录。缓存含在输入、推理含在输出，不重复相加或当账单；finish后只读核对，不在租约释放后回写。

## 5. Decisions Made
- 本轮无新科学结论或新设计。真实外部资源阻塞时保留原稳定ID及原未通过门槛，不制造占位候选、路线穷尽或deadline耗尽，不重做无变化CPU试验。
- 下一次真实空卡先续Y46同修订1生产原生测试与Y49冻结native测试，然后才按完整门槛决定准入；计时器或白名单授权本身不是空卡证据，保护GPU0其他任务与GPU1原训练。
- Y48保持原released-SPFormer/seed1999/单卡/固定30轮和冻结全9508表达、141场景、每方法两次FIRST主结果、W2/B1比较与子组风险评价；不以早期验证分数挑选改动或重启。
- 操作仅AdaRES_project/server2 GPU0、1；后台Pipeline-AdaRES_project保持Disabled。未操作其他项目、server6、调度、新任务或嵌套执行；下一自动化保存触发时间未知。
- finish明确PASS/COMPLETE且独立核实本轮租约已消失后才能称本轮COMPLETED；否则保留证据并abort。科学blocked不等于本轮执行失败，也不等于准入PASS。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
