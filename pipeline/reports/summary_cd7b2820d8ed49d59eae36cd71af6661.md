# Research Pipeline Report - AdaRES_project
快照时间：2026-10-15T01:48:25.279002+00:00；生成报告时快照年龄：89 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-15 09:48:24
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
================================================== 2026-10-15 09:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y45_metric_decomposition_res] Best=0.5077 29/30 no_improve=12/30 gpu=1 ? 序列: 0.5016, 0.5026, 0.5001, 0.5023, 0.5018

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
- 研究储备 2 / 2；储备准备预算 0，不增加训练并发或启动授权。
- 候选累计记录 100：complete=13 / rejected=72 / closed=12 / training=1 / blocked=2。
- 否决范围：unclassified=72；实现修订累计 0；收据文件 28（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮仅AdaRES_project，实际任务ID与foreground owner一致；closure启用，后台调度Disabled，资源仅server2 GPU0/1且一实验一卡。begin/preflight仅一次，monitor/goal均0。快照2026-10-15 09:48:25.279002+08，真实截止10:48:25.279002+08；无新增终态，第0至3段沿用该快照的脚本事实表。
- 已读最新储备与局部修订契约：待续Y46/Y48已计入储备2/2，储备准备预算0；训练准入预算1不是空闲卡或额外训练授权。累计100条候选、72条拒绝范围未分类、28份收据仅为历史记录口径，不是成功率。本轮没有新的负面机制测试，不据此虚构failure_scope或局部修订；未回写历史分类。
- 09:48:57.956958至09:49:03.336490+08，核实原Y45 PID78001、GPU1、工作目录、配置、commit及七源文件哈希一致；原生29/30轮步数3375增至3386/4686。preflight best mIoU0.5077、no-improve12/30仍为中途观测，不是终态成败。未重启训练或改变模型、优化器、调度器、种子、固定30轮协议。
- 09:49:03实时GPU0外部PID19723，23162/24576MiB、94%；GPU1原Y45 PID78001，22754/24576MiB、68%。没有授权空闲独占CUDA卡，两个现有作业保留。到期Y46/Y48原生脚本和冻结清单哈希不变；Y46待同一TRAIN8完整原生损失/反向与配对成本，四门槛pending；Y48待同一TRAIN8联合掩码/匹配捕获，实际非空混合标签组数未知、六门槛pending。既有实际父模型、字面计数、正确/错误对应及重复/置换等非GPU对照不重复，代理结果不作原生PASS。
- 实际产出：实时观测JSON与资源处置JSON各1、原始日志1；Y45追加观测，Y46/Y48各追加观测与处置引用，重试2026-10-15 10:19:03.338729+08或授权卡真实空闲加独占租约。新增科学诊断0、候选0、局部修订0、准入0、新训练启动0。观测pipeline/analyses/y45_continuation_resource_cd7b2820d8ed49d59eae36cd71af6661.json，SHA256 e7ad7fca7d897d331dd7366d2f5e3cea526896cd0861ed6a83e7f19c59558b4e；处置pipeline/analyses/candidate_resource_retry_cd7b2820d8ed49d59eae36cd71af6661.json。
- 科学状态Y45 training、Y46/Y48 blocked，目标未达；本轮执行待finish验证，不能提前称COMPLETED。收据用量与机器统计产出此时未知；窗口仅本地begin至写收据的可见日志，排除初始读取、迟到记录和最终回复。缓存是输入子集、推理是输出子集，不重复相加，不是账单，也不代表科学收益。完成须由PASS/COMPLETE、收据及租约消失共同核实。

## 5. Decisions Made
- 依据本轮实测两卡占用，以external_resource_blocked收尾；保留原稳定候选及冻结下一项测试，不扩充已满储备，不重复无信息增益对照，不硬等截止，不为准入制造变体或放宽门槛。
- Y45继续原固定30轮，待真实终态后执行冻结比较；训练前局部梯度冲突不等于训练后群体伤害。新局部修订政策不授权重开Y43/Y47等已拒绝工作或改写历史证据，科学比较基线仍是W2，账面Y36的0.5110不是科学晋升。
- 不跨项目、不用server6、不改调度、不建任务或后台决策进程。储备预算不改变GPU占用和训练授权；下一次已保存自动化触发时间未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
