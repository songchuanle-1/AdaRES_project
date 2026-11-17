# Research Pipeline Report - AdaRES_project
快照时间：2026-10-14T09:50:16.561699+00:00；生成报告时快照年龄：668 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-14 17:50:16
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
================================================== 2026-10-14 17:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y45_metric_decomposition_res] Best=0.4484 5/30 no_improve=0/30 gpu=1 ? 序列: 0.3235, 0.3570, 0.4175, 0.4484

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
- 项目仅 AdaRES_project；快照为北京时间 2026-10-14 17:50:16，模式 design，原截止时间 18:50:16 未改动。沿用当前租约；begin 的监控和目标检查没有重跑，0-3 节事实表仍对应原快照。
- 17:59:39-17:59:44 实测同一 Y45：PID78001、GPU1、工作目录、配置、提交338a3d7和七个源文件哈希匹配；原生训练6/30，步数961→972/4686。证明训练继续推进，不是重新启动或新实验；尚不能证明最终有效性。
- 17:59:44 授权卡实测：GPU0 97% /24126/24576MiB，外部PID19723；GPU1 83% /22750/24576MiB，Y45 PID78001。两卡均有真实计算进程，未尝试占用卡上的CUDA诊断，没有干预任何既有进程。
- 续查已到期的稳定候选Y48：冻结原生联合捕获脚本及冻结清单SHA256不变。新增真实资源收据和训练连续性证据，路径pipeline/analyses/y48_resource_retry_1c73_1c732ceca30e444a8c8f2d2dcffee912.json及y45_continuation_1c73_1c732ceca30e444a8c8f2d2dcffee912.json；已完成的CPU/源码/重复掩码/联合记账对照均未重跑。
- 科学状态：Y45 training；Y48 blocked，原生混合标签的非空相同掩码组数量仍未知，六项准入门槛仍pending。之前已证伪单赢家匹配前提，171个非空重复组不能代替原生匹配标签，未跨次捕获拼接。Y46尚未到18:09:42重试时间，冻结、证据、状态与重试条件保持不变。
- 当前在训1个是Y45；另有Y46和Y48两个未准入诊断候选，不能计作已开始的训练。关机后原Y43/Y44已保留优化器和调度器恢复并训练结束；Y43子组伤害拒绝、Y44证据不确定，不重启或晋升。
- 本轮实际产出为1份连续性观测及原始输出、1份资源阻塞收据、Y45/Y48新增哈希证据、记忆及报告更新；新候选0、新准入0、新训练0、新原生诊断0。执行状态在finish前为待验证；只有明确PASS/COMPLETE并核实租约释放才可称COMPLETED。
- 用量窗口仅begin至收据写入的本地记录，可能不完整，不含更早读取、后续记录与最终答复；input/output/cached/reasoning数值在finish前未知。缓存是输入子集、推理是输出子集，不重复相加，不作为账单。finish后只读取收据与释放状态，不再写项目。

## 5. Decisions Made
- 无新的科学决策或重复评审；保持原Y45训练和既有Y43/Y44/Y47结论。目标mIoU0.515未达；台账Y36的0.511不是科学晋升，W2仍是科学比较基准。
- 依真实两卡占用按external_resource_blocked收尾，不声称路线耗尽、不硬撑截止、不新增凑数变体，不改变目标、准入门槛、冻结协议、历史失败证据或调度。
- Y48下一步仍是同一冻结TRAIN8原生前向联合捕获最终掩码和criterion匹配索引，检验非空相同掩码组是否实际混合标签及字面TP/FP/FN一致；有真实证伪结果才拒绝，否则先补诊断和原生门槛，不自动准入。
- Y48重试更新为北京时间2026-10-14 18:29:44，或授权GPU0/1实际空闲且取得独占诊断租约时提前解除；Y46原重试18:09:42保持不变。下一次自动化实际触发时刻未知，未读取或修改调度。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
