# Research Pipeline Report - AdaRES_project
快照时间：2026-10-16T04:22:37.246402+00:00；生成报告时快照年龄：3460 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-16 12:22:36
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
================================================== 2026-10-16 12:22 : AdaRES_project ==================================================

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
- 研究储备 3 / 2；储备准备预算 0，不增加训练并发或启动授权。
- 候选累计记录 101：complete=14 / rejected=72 / closed=12 / validation=3。
- 否决范围：unclassified=72；实现修订累计 1；收据文件 29（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 旧12be轮因中断以INCOMPLETE保存收据并规范abort，释放已独立核实；本轮新token=f56bf800538f452ebf4b11f81ca8fd8c，仅一次begin/preflight，原截止时间未改。
- Y48原训练固定30轮终态已复核；本轮六次全量冻结评测和训练后成对成本完成。FIRST mIoU0.503403，较B1+0.1280个百分点、较W2-0.6506个百分点，总体区间跨0；小目标相对W2下降2.3335个百分点且CI上界<0，当前实现拒绝，不晋升。
- Y46修订2已实际重测CPU32、原生32缓存精确重复、released-init两步更新/完整恢复和压力测试；最大对峰值17.867GiB，但训练步中位数17.643s/父0.959s，当前高成本实现不准入并拒绝；原子重复失败/API失败和旧代码全部保留。
- Y49修订1保持描述符等价，CPU计算19.98s降至6.69s，实际原生/独立生产/两步更新/完整恢复及资源测量已做；最大对峰值14.082GiB。原生等组件质量分区区分见证仍0，科学状态validation/inconclusive，不把可运行当准入。
- 后续占用仅只读核验：[{"gpu": 0, "memory_MiB": 24186, "total_MiB": 24576, "utilization": 100, "pids": [19723]}, {"gpu": 1, "memory_MiB": 1, "total_MiB": 24576, "utilization": 0, "pids": []}]；GPU0外部作业保护，GPU1无正式训练。新增ID0、准入0、正式训练0；下轮调度时间未知。
- 报告与七段记忆已保存；此时finish/postflight仍待执行，最终执行状态以随后实际命令和收据及锁释放核验为准。收据用量窗口只覆盖begin至记录，缺失未知，缓存属于输入、推理属于输出，不重复相加，不是账单。

## 5. Decisions Made
- 拒绝Y48已训练当前监督实现，依据原冻结小目标伤害规则，不外推整个一致性机制被证伪。
- 拒绝Y46两次修订后的高成本实现；不以小误差为由放宽精确重复门槛，不添加第三次修订。
- 保留Y49同ID续做原生TRAIN信息缺口诊断；不降低准入、不凑变体、不重启或改变调度。
- 仅因真实截止时间已近而保留明确待续测试，不造假外部阻塞、不设置无证据冷却。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
