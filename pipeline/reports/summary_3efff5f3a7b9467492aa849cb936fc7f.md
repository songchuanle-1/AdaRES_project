# Research Pipeline Report - AdaRES_project
快照时间：2026-10-01T14:49:31.204188+00:00；生成报告时快照年龄：3216 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-01 22:49:31
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
================================================== 2026-10-01 22:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y15_query_completion_res] Best=0.2238 25/30 no_improve=1/30 gpu=0 ? 序列: 0.2077, 0.1777, 0.2021, 0.2238, 0.2108
  [adares_y17_target_equivalent_pairing_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [日报] 1 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-01.md

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y15 | configs_song/model_paper5/adares_y15_query_completion_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。

## 4. This Tick's Actions
- 快照时间2026-10-01 22:49:31+08，模式design；Y17经规范deathwatch3/3归档done_complete，快照活跃1/目标2/准入预算1。先创建稳定候选Y18 diagnosis，未重复preflight。
- Y18检验精确原始点测度的匹配掩码BCE/Dice；1000解析案例与逐点展开损失/梯度等价，最大误差4.44e-16/8.33e-17。64固定训练场景256表达两次原生捕获：最终层梯度方向余弦old0.778066/point0.914328，60场景改善、4下降、12反向表达；mass-only/occupancy-only/equal-size分离对照已保留，不宣称训练效力。
- 独立物理包与W2配置一致，0新增参数；释放初始化4原生批次有限梯度更新、tools.train两步及gorilla保存恢复通过。恢复前模型/优化器/调度器精确；继续一步最大权重差3.05e-5，不声称逐位/全加载器恢复。最大1055503点/7618超点压力Y18/W2峰值预留14.078125GiB，两者参数164629637；训练中位耗时1.065476/0.975367秒，额外约9.24%。32表达推理parent/repeat/标签反事实二值与选择翻转0，原始概率噪声保留。
- 源码哈希/独立导入/冻结协议六项准入审查通过。初次工具精度、缺远端压力清单及错误预测字段已真实记录并独立修复重测。首次规范分发因MSYS本地路径失败；仅修复项目sync.sh本地cygpath，保留哈希保护；提交f0a704f候选与00cf78f同步修复，无无关提交或历史回滚。
- 23:37:51规范gated dispatcher启动Y18 GPU1 PID7449；23:40:28.567759+08独立核验：Y18 Train[1/30]304→305→306/4686；Y15原PID18796 GPU0 Train[26/30]38→39→40/4686。GPU0/1显存17282/16896MiB、利用率55/41%；10个新增部署文件与Git提交哈希一致，远端标记00cf78fcbb2b18debb9754a57189249984b5045f；诊断租约已释放。短尾未见Traceback，不声称全日志无故障。
- 本轮产出一个真实准入并原生启动的新候选，以及解析、重复对照、原生训练/资源/恢复/推理/源码/故障及启动证据；更新稳定候选与七段记忆。科学状态Y15/Y18 training、Y17 efficacy inconclusive；执行状态以finish成功及锁释放为准。收据用量窗口和计数待finish后读取，此处未知，不能当账单。

## 5. Decisions Made
- 本轮目标准入1个已达到，当前两卡各一个原始独立任务；不额外设计占位变体、不启动控制复本、不改变Windows后台调度或操作其他项目。
- Y18效力未知；4个反向场景、混合占比处理稍弱于mass-only、Dice平滑尺度和训练代价增加均保留。按冻结30轮及完整ScanRefer/W2/B1、场景聚类CI、实测分组重复噪声/伤害和准确率成本前沿计划评估，不验证调参。
- 保留Y15原进程，终态后做冻结全评估；Y17此前完整评估不重复，不提升champion。下一轮核对原PID/唯一日志与训练进度；本轮未读取自动化具体下次触发时间，未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
