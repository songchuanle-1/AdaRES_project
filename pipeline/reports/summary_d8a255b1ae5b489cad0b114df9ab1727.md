# Research Pipeline Report - AdaRES_project
快照时间：2026-10-14T12:50:22.001649+00:00；生成报告时快照年龄：166 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-14 20:50:21
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
================================================== 2026-10-14 20:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y45_metric_decomposition_res] Best=0.4886 10/30 no_improve=0/30 gpu=1 ? 序列: 0.4158, 0.4620, 0.4673, 0.4672, 0.4886

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
- 仅AdaRES_project；实际任务归属、最新资源授权及closure启用均核实，后台调度Disabled。快照为北京时间2026-10-14 20:50:22，design，原截止21:50:22未修改；begin监控/目标检查各一次，无新终态，0-3节沿用脚本快照。
- 按ID读取到期的Y46/Y48，只加载当前假设、门槛、下一项、阻塞条件及定向冻结/近期证据。复用已有有界观测助手，确认两原生脚本和冻结清单SHA256与账本一致；非GPU对照已完成，没有重复CPU/源码/置换测试、评审或新变体。
- 20:51:26-20:51:31同一Y45的PID78001/GPU1、工作目录、配置、提交338a3d7及七个源文件哈希一致；原生10/30轮，步数2172→2181/4686，训练继续推进，未重启。begin监控途中best0.4886，不是最终冻结比较或科学晋升。
- 20:51:31授权GPU0实测100% /24226/24576MiB、外部计算PID19723；GPU1为99% /22750/24576MiB、Y45 PID78001。两卡均有真实计算占用，没有空闲独占诊断资源；未执行占用卡上的CUDA诊断或干预已有进程。
- 科学状态：Y45 training；Y46 blocked，native_backward/resources/independence/constraints仍pending；Y48 blocked，冻结原生联合捕获未执行，实际非空同掩码组混合标签数量未知，六项门槛pending。资源观测不是科学验证，不跨次捕获拼接，不改变冻结或准入。
- 实际产出：1份连续性观测及原始输出pipeline/analyses/y45_continuation_resource_d8a255b1ae5b489cad0b114df9ab1727.json/.log，1份共同资源收据pipeline/analyses/candidate_resource_retry_d8a255b1ae5b489cad0b114df9ab1727.json；Y45/Y46/Y48新增真实哈希证据，记忆及完整六段报告更新。新候选0、新准入0、新训练0、新原生诊断0。
- 执行状态finish前待验证；只有明确CLOSURE_POSTFLIGHT_PASS和COMPLETE并核实租约消失后才称COMPLETED。上轮0cba已FINALIZED/PASS且释放；旧记忆中的待验证措辞不是残留租约。
- 收据用量窗口为begin到收据写入的部分可见本地记录，不含更早读取、晚到记录和最终答复；finish前数值未知。缓存是输入子集、推理是输出子集，不重复相加、不作为账单；finish后只读收据和释放状态，不再写项目。

## 5. Decisions Made
- 无新科学决策。按真实两卡占用以external_resource_blocked收尾，不声称路线耗尽，不硬撑截止，不制造凑数变体，不放宽准入或改写失败证据。目标mIoU0.515未达，W2仍为科学比较基准，既有结论保持不变。
- 保留同一Y46冻结TRAIN8原生损失/反传/配对成本测试，再补tools.train更新、released-init及独立性门槛；保留同一Y48冻结TRAIN8最终掩码/criterion匹配索引联合捕获，检验实际混合标签及字面点TP/FP/FN。没有原生结果不声称PASS或自动准入。
- 两候选重试更新为北京时间2026-10-14 21:21:31，或授权GPU0/1实测空闲并拿到独占诊断租约时解除；稳定ID、科学状态、冻结和门槛不变。当前仅Y45在训，两个诊断候选不计作已启动训练。
- 没有其他项目、server-6、调度、新任务、嵌套codex exec或后台恢复操作。已保存自动化下一次实际触发时刻未知，未读取、推算或修改调度。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
