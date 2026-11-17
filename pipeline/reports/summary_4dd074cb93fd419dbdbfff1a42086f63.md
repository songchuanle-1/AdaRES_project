# Research Pipeline Report - AdaRES_project
快照时间：2026-10-15T00:48:46.917224+00:00；生成报告时快照年龄：111 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-15 08:48:46
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
================================================== 2026-10-15 08:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y45_metric_decomposition_res] Best=0.5077 28/30 no_improve=11/30 gpu=1 ? 序列: 0.5064, 0.5016, 0.5026, 0.5001, 0.5023

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
- 仅对AdaRES_project执行一次begin/preflight，实际任务ID与foreground归属一致，closure启用、后台调度Disabled；资源仅server2 GPU0/1，一实验一卡。快照2026-10-15 08:48:46.917224+08，真实截止09:48:46.917224+08，monitor/goal均返回0，无新增终态。第0至3段为该快照，不表示后续实时状态。
- 08:49:14.021598至08:49:19.330274+08核实原Y45 PID78001、GPU1、配置、工作目录、commit与七源文件哈希；原生训练第28/30轮步数1054增至1064/4686。preflight best mIoU0.5077、no-improve11/30仅为中途读数，不构成失败、晋升或终态证据。未重新启动或改变模型、优化器、调度器、种子及固定30轮协议。
- 08:49:19实测server2 GPU0由外部PID19723占用，23426/24576MiB、97%；GPU1由原Y45 PID78001占用，22754/24576MiB、55%。没有空闲独占CUDA卡，所有现有进程保留。该新观测仅补充本段，不用来填造脚本快照中的未知指标。
- 续查到期稳定候选Y46/Y48，原生脚本及冻结清单哈希不变。Y46匹配对照为实际W2/Y31/Y36、恒特征等度图见证及重复/联合置换/分量字面计数；CPU接线已验证，同一TRAIN8完整原生损失/反向及配对成本仍未执行，native_backward/resources/independence/constraints待审。Y48匹配对照为实际原生indicator/matcher、重复掩码梯度与正确/错误/等值目标/置换；同一TRAIN8联合掩码和匹配捕获未执行，实际非空混合标签组数未知，六门槛pending。既有CPU和源码测试未重复，未把代理结果算作原生PASS。
- 实际新增实时观测JSON、阻塞处置JSON各1份及原始日志1份；Y45追加观测，Y46/Y48各追加观测与处置引用，重试2026-10-15 09:19:19.332502+08或授权卡真实空闲且持独占租约。新增科学诊断0、候选0、准入0、新训练启动0。观测pipeline/analyses/y45_continuation_resource_4dd074cb93fd419dbdbfff1a42086f63.json，SHA256 12d9e66b98e8b89720680596857f6b121e443dfcaa2d749f883be3fdbbfa79c7；处置pipeline/analyses/candidate_resource_retry_4dd074cb93fd419dbdbfff1a42086f63.json。
- 科学状态Y45 training、Y46/Y48 blocked，目标未达；执行状态待finish核实，不能提前称COMPLETED。收据用量和机器统计产出此时未知；统计仅限本地begin至写收据可见日志窗口，排除初始读取、迟到记录及最终回复。缓存是输入子集、推理是输出子集，不重复相加，不作账单或科学收益解释；最终须结合PASS/COMPLETE、收据及租约消失核验。

## 5. Decisions Made
- 基于本轮真实两卡占用，以external_resource_blocked收尾，保持同一冻结Y46/Y48及具体恢复条件；不等待截止、不制造等价变体、不重复无信息增益测试，不宣称路线已穷尽，不放宽准入门槛。
- 继续原Y45固定30轮，等待真实终态后按冻结方案比较，不从中途指标判方法成败。W2仍为科学比较基线，账面Y36的0.5110不等于科学晋升；已完成的Y43/Y44关机续训结论与Y47反证拒绝保持。
- 仍是1个运行实验加2个未准入、资源阻塞的稳定候选；保留外部作业，不操作其他项目/server6/调度，不创建任务或后台决策进程。下一次已保存自动化触发时间未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
