# Research Pipeline Report - AdaRES_project
快照时间：2026-09-30T03:08:22.867151+00:00；生成报告时快照年龄：3751 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-09-30 11:08:22
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
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-09-30 11:08 : AdaRES_project ==================================================

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

## 4. This Tick's Actions
- Y15 完成物理独立的 teacher/GT-only 包和配置，最终 128->129 查询、六辅助层保持128；新增329728参数、9张量。
- 四个实际训练场景两批原生反向及tools/train.py两步更新完成；SPFormer发布初始化、当前增强教师目标对齐、detach及教师不进入学生checkpoint已验证。
- 同输出teacher/GT仅最终BCE变化，其他及辅助损失完全相同；语言/slot改变新增掩码而原有K掩码不变。教师/GT梯度余弦0.99999976/0.99999994，必要性未证明。
- 固定解码器输入重复差异0；固定输出改GT预测完全一致，新实例predict不构建教师。旧端到端概率相等失败保留，不转为PASS；探针eval链式None故障已修复重跑。
- 独立进程含优化器teacher reserved5.242/7.611GiB、GT4.627/6.992GiB；teacher约慢41%-44%。两场景热推理76.5/95.1ms，非全量性能。
- 12:06实时GPU0/1均0%/1MiB，诊断租约释放；无正式30epoch训练、无入队/准入/提交。截止12:08，保留Y15 validation及具体下一测。本轮执行状态以finish收据为准。

## 5. Decisions Made
- 不把非零梯度、oracle诊断或两步优化当作训练效益；teacher弱增量、轻量化/Pareto及伤害规则待冻结，不放宽门槛、不加权调参。
- 下一轮先续做Y15大型训练场景资源/序列化恢复、teacher正确/打乱梯度特异性与稀疏重复噪声；保留GT-only物理独立对照，Y14不重开。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
