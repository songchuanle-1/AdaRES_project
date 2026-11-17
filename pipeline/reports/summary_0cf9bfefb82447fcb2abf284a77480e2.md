# Research Pipeline Report - AdaRES_project
快照时间：2026-10-14T08:48:43.687572+00:00；生成报告时快照年龄：2209 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-14 16:48:43
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
================================================== 2026-10-14 16:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y45_metric_decomposition_res] Best=0.4175 4/30 no_improve=0/30 gpu=1 ? 序列: 0.3235, 0.3570, 0.4175

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
- AdaRES快照2026-10-14 16:48:43北京时间/design，原截止17:48:43未改变；begin监控与目标检查各一次，活跃Y45一项/等待0/认领0/预算1。报告写入时尚未finish，执行状态待校验。
- 目标0.515未达，台账Y36 0.5110/差距0.004不是新科学晋升，W2比较基准不变。监控原文Y45 4/30、Best0.4175/no_improve0；脚本结构化表未采集字段仍未知，不回填后续数据。
- 先登记稳定Y48再审查实际源代码；原生match_sampled_ins已有最近查询及实例内所有查询匹配，初始单一获胜者前提被否定，原登记保留并增加源语义澄清，不声称Hungarian一对一。
- 32个实际损失/匹配/指标函数CPU控制，在构造的相同输出/不同种子归属下出现相反监督梯度；原生eos权重未变，字面Float32梯度误差0、重复及重标号检查通过。这不是已观测的原生TRAIN标签冲突。
- 程序读取旧冻结TRAIN8/32表达/128查询掩码，所有32表达存在非空完全重复组，共171组/565查询；2710空重复查询单独统计。旧捕获没有对应逐查询匹配索引，实际混合标签数量未知，未跨捕获拼接推断。
- 首版Double控制因原生函数Float32标签索引赋值类型不匹配失败，源码/日志保留；v2只遵循实际Float32类型契约，精度检查在重跑前追加记录，没有改源函数、权重、掩码阈值或准入门槛。
- 同一次原生前向联合捕获掩码/匹配索引的入口已准备，32个CPU联合统计/映射/重复检查通过；GPU分支已编译但未执行。Y48全部六项准入门槛仍待验证，科学状态部分diagnosis、原生检查blocked，不新增训练。
- 17:20:00-05核验原Y45 PID78001/cwd/config/GPU1/源码哈希一致，第5/30轮1052→1062/4686步；releasedSPFormer/seed1999/固定30轮/模型与优化器保持原样，科学状态training，尚无有效性结论。
- 17:20:05实时GPU0外部PID19723/23606MiB/89%，GPU1Y45 PID78001/22750MiB/98%，均保留。Y48原生联合捕获受真实资源阻塞；旧Y46未到期，冻结/18:09:42重试条件不改，不重复其控制。
- 收据用量与产出finish后只在任务中补充，当前未知。统计begin至收据局部日志窗口，不含初始读取/最终回复/晚到记录；缓存属于输入、推理属于输出，不重复相加、不当账单。

## 5. Decisions Made
- 20261014-171941_next_batch：同一Y48保留部分诊断，不宣称原生标签冲突或效益、不直接实现生产目标闭包。必须先联合捕获同一前向的匹配索引与硬掩码，反证则拒绝，有支持再逐门槛验证。
- 本轮以external_resource_blocked真实收尾，不称路线穷尽，不硬撑截止、不降低门槛。Y48于2026-10-14 17:50:05+08或观测到授权卡真实空闲时续做冻结原生捕获；这是研究重试条件，不是下一自动化触发时间。
- Y45保持原固定30轮；Y46旧冻结及重试不变；Y43损害拒绝/Y44不确定/Y47拒绝不重开，不使用空掩码重复作为正向诊断证明。
- 未操作其他项目/server6/外部任务，未改目标/历史证据/调度，未新建任务/嵌套执行/恢复后台。下一准确自动化时间未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
