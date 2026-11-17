# Research Pipeline Report - AdaRES_project
快照时间：2026-10-15T14:48:41.484528+00:00；生成报告时快照年龄：384 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-15 22:48:41
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
================================================== 2026-10-15 22:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y48_mask_equivalence_res] Best=0.5034 15/30 no_improve=0/30 gpu=1 ? 序列: 0.4846, 0.4944, 0.4936, 0.4976, 0.5034

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
- 本轮快照2026-10-15T14:48:41.484528+00:00（本地22:48:41），真实截止2026-10-15T15:48:41.484528+00:00；begin/preflight各一次，0-3段沿用脚本事实表；finish尚未执行，本轮执行状态待规范收尾核验。
- 初步处置22:52:01早于原retry22:52:20，已保留其真实monitor/延期证据。报告准备期间原retry到时，随后在2026-10-15T22:52:38.982807+08:00至2026-10-15T22:52:44.904536+08:00执行两项候选共享的唯一一次资源检查，最终处置以该新现场证据为准；没有睡眠等截止、提前探针、重复preflight或删除初步证据。
- 原Y48/PID8877/cwd/配置/argvGPU1成立，7项源码/配置哈希与原准入提交逐项相同；第15/30轮原生步数3232至3241/4686，原训练持续推进，无重启、迁移或新训练启动。
- 现场授权2张：本任务Y48占GPU1（PID8877、69%、22720/24576MiB），其他任务占GPU0（PID19723、95%、23598/24576MiB），空闲0。未分配诊断CUDA上下文或干预当前/外部训练；0-3段未知项是更早preflight，本段给出标明时间的后续事实。
- begin日志显示Y48第15/30轮、阶段best mIoU=0.5034（上轮0.4976）。虽数值略高于原B1历史数值，未做同冻结口径全量终态比较，不能宣称可靠提升、显著性或晋升；仍未超历史champion或达0.515。Acc@50/spmIoU等本轮未新增采集，未知。
- Y46修订1与Y49冻结清单/下一原生程序哈希未变，语法编译通过但GPU分支未执行；生产native/tools.train/完整恢复/资源等门槛保持待办，保留Y46成本风险及Y49原生同component分区差异见证为0等限制；资源检查不是科学PASS。
- 科学状态Y48=training_not_efficacy、Y46/Y49=blocked；按真实外部占用以external_resource_blocked收尾。下一有界可复查2026-10-15T23:52:44.904536+08:00（新现场结束后1小时，不是自动化触发承诺），真实资源/目标/证据变化可提前唤醒，但GPU测试须现场空闲与独占租约。
- 实际产出1次共享资源/训练进度检查、1份最终双候选阻塞处置及哈希、保留1份初步monitor/延期证据、候选证据与7节短记忆和完整六段报告。储备2/2、准备预算0；新候选0、准入0、新训练0、GPU诊断0、修订0，无重复CPU控制或终态重测。
- 收据尚未生成，用量/调用数暂未知；窗口仅begin至收据的本地可见记录，早期读取/最终回复/迟到记录不计。缓存含于输入、推理含于输出，不重复相加或当账单；finish后只读核对，不在释放租约后回写。

## 5. Decisions Made
- 无新科学结论或新设计；候选到时后执行真实资源复查，实际占用仍阻塞则保留同ID/原门槛，以外部资源阻塞收尾，不凑变体、降准入或造路线穷尽。
- 下一真实空卡先续Y46同修订1生产原生路径及Y49同冻结native程序，完整验证后才决定准入。白名单、计时器或逻辑空位不是空闲/独占授权，保护GPU0其他任务和GPU1原Y48。
- Y48保持released-SPFormer/seed1999/单卡/固定30轮及冻结终态9508表达、141场景、每方法两次FIRST主结果、W2/B1比较及子组风险评价，不据早期分数调整协议或晋升。
- 仅AdaRES_project/server2 GPU0、1；后台Pipeline-AdaRES_project仍Disabled，未操作其他项目、server6、调度、新任务或嵌套执行，下一保存自动化时刻未知。
- finish明确PASS/COMPLETE且独立核实本轮租约消失才称COMPLETED；否则保留证据并abort，科学blocked与执行完成或科学有效性分开。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
