# Research Pipeline Report - AdaRES_project
快照时间：2026-10-16T06:16:42.533745+00:00；生成报告时快照年龄：1253 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-16 14:16:42
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
================================================== 2026-10-16 14:16 : AdaRES_project ==================================================

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
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 候选累计记录 102：complete=14 / rejected=75 / closed=12 / blocked=1。
- 否决范围：unclassified=72 / implementation=3；实现修订累计 3；收据文件 29（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮仅AdaRES_project，14:16:42+08快照为design；due0、未到期阻塞1、储备准备预算1。先登记稳定Y51诊断，再做定向原生源码和CPU控制；未重做Y50或重置授权重试时间。
- Y51审计公共logit平移是否造成训练CE与部署选择错配。实际native RES已使用两类softmax前景概率singleargmax；32固定fixture、1精确阳性对照、既有TRAIN8/32共65组选中掩码不变，原始前景logit规则29组排名变化。
- 最大原生选择器分值差2.220446e-16、float64对照CE差1.776357e-15、梯度差5.551115e-17，均在预注册包络内；字面原生float32损失/梯度精确相等。float64仅显式适配one-hot缓冲dtype，不冒称完整原生GPU训练。
- 首个程序get_iou误导入在数值测试前失败，原程序/日志保留；v2仅提取实际loss.py原生函数，模型、fixture与判据不变。未创建模型变体、未用GPU、未做验证集评估、未修改队列或启动训练。
- 执行状态待finish核验，科学状态Y51 rejected/hypothesis、Y50原有blocked/human_authorization未变。GPU实时占用本轮未知，不将上轮缓存当空闲证据。
- 本轮收据用量在finish前未知；统计仅begin至收据的本地响应窗口，早先读档/晚到用量/最终回复不含；缓存属于输入、推理属于输出，不重复相加、不当账单。实际本轮新诊断1、模型变体0、GPU测试0、准入0、训练启动0。

## 5. Decisions Made
- 只否定Y51所声称的公共模态错配前提，保留原生选择器，不创建无此诊断依据的数值校准或排序变体，不外推所有置信度机制无效。
- Y50尚未到18:08:33+08授权重查时间且无新授权；其既有提交阻塞原样保留，不重复审查、重测、改写证据或延后计时。
- 本轮一个新诊断预算已执行；其余已登记未完成事项仍为真实人工提交授权阻塞。按外部授权阻塞收尾，不声称路线穷尽、无信息增益或GPU不足，不硬撑deadline，不降低准入。
- 没有新的训练后指标、准入或启动，不需要重复发出上轮已经提出的人工授权通知；六段报告和原始证据完整保存。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
