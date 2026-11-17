# Research Pipeline Report - AdaRES_project
快照时间：2026-10-14T11:49:39.410827+00:00；生成报告时快照年龄：184 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-14 19:49:39
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
================================================== 2026-10-14 19:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y45_metric_decomposition_res] Best=0.4673 8/30 no_improve=0/30 gpu=1 ? 序列: 0.4175, 0.4484, 0.4158, 0.4620, 0.4673

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y45 | configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。

## 4. This Tick's Actions
- 仅AdaRES_project，实际任务与foreground归属匹配，closure启用，后台调度Disabled；快照为北京时间2026-10-14 19:49:39，design，原截止20:49:39未改。begin监控/目标检查各一次；0-3节保持原快照事实，本轮无新终态。
- 按ID续查两个到期候选Y46/Y48，复用既有有界观测助手。两份冻结清单和原生诊断脚本的SHA256与各自账本证据一致；非GPU对照已完成，本轮没有重跑CPU、源码、重复或置换对照，没有新增评审或变体。
- 19:51:07-19:51:13同一Y45：PID78001/GPU1、配置、目录、提交338a3d7和七个源文件哈希匹配；原生8/30轮，步数4672→4682/4686，确实继续推进，未重启。begin监控途中best0.4673，不是最终结果、目标达成或科学晋升。
- 19:51:13授权资源实测：GPU0利用率0%、18220/24576MiB、外部计算PID19723仍在；GPU1利用率68%、22750/24576MiB、Y45 PID78001。瞬时0%不等于空闲，GPU0仍有约18GiB显存和真实计算进程；没有可独占诊断卡，没有CUDA探测、占用或干预既有任务。
- 科学状态：Y45 training；Y46 blocked，native_backward/resources/independence/constraints仍pending；Y48 blocked，原生联合捕获未执行，实际非空相同掩码组混合标签数量未知，六项门槛pending。资源收据不是科学验证；不拼接不同捕获的掩码/标签，不改冻结协议或准入门槛。
- 实际产出：1份Y45连续性观测及原始输出pipeline/analyses/y45_continuation_resource_0cbacc8bb1e1496fb5a67b200db6fa55.json/.log，1份共同资源收据pipeline/analyses/candidate_resource_retry_0cbacc8bb1e1496fb5a67b200db6fa55.json；真实哈希证据补充到Y45/Y46/Y48，记忆/完整六段报告更新。新候选0、新准入0、新训练0、新原生诊断0。
- 执行状态finish前待验证；只有明确CLOSURE_POSTFLIGHT_PASS/COMPLETE并核实当前租约消失才称COMPLETED。上轮9532的FINALIZED/PASS和释放已核实，旧记忆中的待验证描述不是未释放租约。
- 用量只统计begin至收据写入的部分本地记录窗口，不含之前读取、晚到记录、最终答复；finish前数值未知。cached是input子集，reasoning是output子集，不重复相加，不作为账单；释放后只读收据和锁状态，不再写项目。

## 5. Decisions Made
- 无新科学决策；真实两卡占用支持external_resource_blocked收尾，不声称路线耗尽，不硬撑截止，不以重复测试或凑数变体填容量。目标mIoU0.515仍未达，W2保持科学比较基准，既有拒绝/不确定结论不变。
- Y46下一项仍是同一冻结TRAIN8原生完整损失/反传/配对成本分支，然后补tools.train更新、released-init与独立性门槛；Y48下一项仍是同一冻结TRAIN8联合最终掩码/criterion匹配索引捕获，检验实际混合标签及字面TP/FP/FN。没有原生结果就不声称PASS、不自动准入。
- Y46/Y48重试更新为北京时间2026-10-14 20:21:13，或授权GPU0/1实测空闲且拿到独占诊断租约时解除；稳定ID、科学状态、冻结和门槛保持不变。当前仅Y45在训，两候选不是两个已启动训练。
- 没有其他项目、server-6、调度、新任务、嵌套codex exec或后台恢复操作。已保存自动化的下一次实际触发时间未知，未读取、推算或改动。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
