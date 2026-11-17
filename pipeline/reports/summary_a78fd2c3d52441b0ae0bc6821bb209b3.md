# Research Pipeline Report - AdaRES_project
快照时间：2026-10-14T19:48:15.019500+00:00；生成报告时快照年龄：100 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-15 03:48:14
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
================================================== 2026-10-15 03:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y45_metric_decomposition_res] Best=0.5077 20/30 no_improve=3/30 gpu=1 ? 序列: 0.4973, 0.5077, 0.5035, 0.4974, 0.4996

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
- 仅AdaRES_project；实际任务与foreground归属一致，closure启用，后台Disabled。快照为北京时间2026-10-15 03:48:15，design，原截止04:48:15未改；begin监控/目标检查各一次，无新终态，0-3节沿用脚本快照。
- 按ID续查到期Y46/Y48的假设、匹配对照、门槛、阻塞、下一项与定向冻结证据；两原生诊断脚本及冻结清单SHA256匹配账本。复用已有有界观测助手，已完成非GPU对照未重跑，没有重复CPU/源码/置换测试、新评审或新变体。
- 2026-10-15 03:48:54-03:48:59同一Y45原生20/30轮，步数3797→3807/4686；PID78001/GPU1、目录、配置、提交338a3d7与七个源文件哈希匹配，训练继续推进，未重启。begin监控best0.5077/no_improve3只是途中记录，不是最终比较、科学晋升或目标达成，固定30轮不变。
- 03:48:59授权GPU0为100% /23986/24576MiB、外部计算PID19723；GPU1为66% /22754/24576MiB、Y45 PID78001。两卡有真实计算占用，没有空闲独占诊断资源，未执行占用卡上的CUDA诊断或干预任何既有任务。
- 科学状态：Y45 training；Y46 blocked，native_backward/resources/independence/constraints仍pending；Y48 blocked，原生联合掩码/匹配索引捕获未执行，实际非空同掩码组混合标签数量未知，六项门槛pending。资源收据不算科学验证，不跨次捕获拼接，不改变冻结或准入。
- 实际产出：1份连续性观测及原始输出pipeline/analyses/y45_continuation_resource_a78fd2c3d52441b0ae0bc6821bb209b3.json/.log，1份共同资源收据pipeline/analyses/candidate_resource_retry_a78fd2c3d52441b0ae0bc6821bb209b3.json；Y45/Y46/Y48新增真实哈希证据，记忆与完整六段报告更新。新候选0、新准入0、新训练0、新原生诊断0。
- 执行状态finish前待验证；明确CLOSURE_POSTFLIGHT_PASS/COMPLETE并核实当前租约消失后才称COMPLETED。上轮410e已FINALIZED/PASS并释放，旧记忆中的待验证描述不是残留租约。
- 收据用量仅begin至收据写入的部分可见本地窗口，不含更早读取、晚到记录与最终答复；finish前数值未知。cached是input子集，reasoning是output子集，不重复相加、不作为账单；finish后只读收据和释放状态，不再写项目。

## 5. Decisions Made
- 无新科学决策。依真实两卡占用按external_resource_blocked收尾，不声称路线耗尽、不硬撑截止、不重复完成的测试、不放宽准入或重写历史失败证据。目标mIoU0.515仍未达，W2保持科学比较基准，既有结论不变。
- Y46下一项仍是同一冻结TRAIN8原生完整损失/反传/配对成本，再补tools.train更新、released-init及独立性；Y48下一项仍是同一冻结TRAIN8最终掩码/criterion匹配索引联合捕获，检验实际混合标签和字面点TP/FP/FN。未执行不声称PASS、不自动准入。
- 两候选重试为北京时间2026-10-15 04:18:59，或授权GPU0/1实测空闲且取得独占诊断租约时解除；稳定ID、科学状态、冻结及门槛不变。当前仅Y45在训，两个诊断候选不是已启动训练。
- 没有其他项目、server-6、调度、新任务、嵌套codex exec或后台恢复操作。已保存自动化下一次实际触发时间未知，未读取、推算或修改调度。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
