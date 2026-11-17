# Research Pipeline Report - AdaRES_project
快照时间：2026-10-11T13:49:58.987108+00:00；生成报告时快照年龄：2733 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-11 21:49:58
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
================================================== 2026-10-11 21:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y41_boundary_probability_res] Best=0.4697 7/30 no_improve=0/30 gpu=0 ? 序列: 0.3661, 0.4291, 0.4489, 0.4570, 0.4697
  [adares_y39_point_distribution_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [日报] 1 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-11.md

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y41 | configs_song/model_paper5/adares_y41_boundary_probability_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。

## 4. This Tick's Actions
- 本轮 design；begin 一次、紧凑视图先读。快照为2026-10-11 21:49:58+08；0–3段仍绑定该时点，不代表后续训练状态；当前线程、AdaRES归属、closure授权、冻结协议和资源白名单已核实。
- Y39终态真实核实：原PID102267已消失，last30/best23和模型/优化器/调度器保留；本轮全9508表达/141场景，Y39/W2/B1各两遍、FIRST主分析。窗口21:59:12–22:10:35+08；Y39 mIoU0.50770598/Acc25 0.60696256/Acc50 0.56121161/spmIoU0.47172302。对W2差-0.00230387，场景CI[-0.01136039,0.00633831]；对B1差+0.00548182，CI[-0.00326558,0.01451277]。
- Y39科学状态complete_inconclusive，W2保留；观测规模组区间跨0不等于无害，multi-object0/0缺数据，official unique/multiple未知；每表达重复最大差Y39=1.0。训练后配对压力成本Y39/W2为0.98774/0.94087秒、13.5137/13.4004GiB，无Pareto提升证据；原检查点未改写。
- 先注册稳定Y42诊断，再完成32例等边际/实际提取对照、TRAIN8/init4原生错误掩码/协方差/保PSD反射、两次原生更新和精确保存恢复、配对资源、两份完整模块同解释器父AST/配置/七文件独立哈希。六项可行性门通过；初始损失+0.01830、错误对照同样学习及下一步数值差均保留，不冒充效果或必要性。
- Y42资源窗口22:27:55–22:29:53+08：新增398594参数；压力训练Y42/W2中位1.123867/0.912554秒、保留14.046875/13.400391GiB，时间回退约23.16%；低于冻结24GiB上限，不宣称训练后Pareto。一个GBK构建故障和三个测试适配器绑定失败原件保留，修复未改变模型或准入标准。
- 仅七个已测试新文件提交243e0b6827073831e0d68090857755d575efbf16，未push/未改无关变更；哈希绑定六门收据和决策pipeline/decisions/20261011-223124_next_batch.md后走canonical GPU1准入派发，无重复preflight。
- Y42启动独立核验窗口2026-10-11T22:35:04.150455+08:00至2026-10-11T22:35:09.961758+08:00：PID91215/GPU1/Train1/30，步数266->274/4686，cwd/config/提交和源哈希匹配。Y41 GPU0原PID116906及源字节保持；启动时GPU1设备占用17456MiB/利用率59%，仅该窗口观测。
- 科学状态Y42 training_not_efficacy；活跃2/目标2，固定30轮原生训练继续，不等待全训练。第0–3段未知值不补零；未操作其他项目/server6/调度/新任务/后台。当前执行状态待finish；实际收据用量窗口和产出在完成后消息中补充，缺失记未知，缓存不重复相加，不当账单。

## 5. Decisions Made
- Y39证据不充分、目标未达，继续保留W2；不把四舍五入CSV、相似均值或跨0区间当科学提升/等价/无害。
- Y42完整点协方差仅获可证伪固定30轮训练准入；保留不利初始损失、错误对照学习和资源回退，无效果/必要性/新颖性/人类审批声明，Phase2未授权。
- 容量已达2，只保留原Y41 GPU0与新Y42 GPU1；终态后依冻结9508/141各两遍/FIRST/场景bootstrap/规模及对象分组/对任一基线伤害/重复噪声/训练后成本复核，不重启或微调。下一调度时刻未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
