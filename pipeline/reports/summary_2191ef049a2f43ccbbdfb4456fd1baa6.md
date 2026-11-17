# Research Pipeline Report - AdaRES_project
快照时间：2026-10-21T04:55:01.427577+00:00；生成报告时快照年龄：1457 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-21 12:55:00
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
================================================== 2026-10-21 12:54 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y56_unique_competitors_res] 16/30 尚无验证 - gpu=0
  [adares_y57_peer_second_moment_res] 18/30 尚无验证 - gpu=1

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y56 | configs_song/model_paper5/adares_y56_unique_competitors_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y57 | configs_song/model_paper5/adares_y57_peer_second_moment_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 111：complete=16 / rejected=79 / closed=12 / training=2 / blocked=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 35（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮归属已用实际CODEX_THREAD_ID核验；旧后台任务Disabled。12:55快照experiment_first启用，准入预算2；只处理AdaRES_project/server-2 GPU0/1。
- 13:11:11实核Y56 PID13738/Train16of30 step2704至2788，Y57 PID16317/Train18of30 step2528至2614；命令和cwd匹配，原生步数递增，无尾部报错。未重启、重排或同步原训练；未采集到新验证结果，不以未知代零。
- 新增Y60阶段专属置信头：保留初始头，六个精炼阶段各使用独立deepcopy头；原生YAML值、七阶段损失、point-mass竞争、mask/selector、seed1999/fixed30保持W2。总参数166222493，增1592856。
- 13:06 CPU实测原生YAML完整模型构造、发布骨干加载及两场景162738点/8条TRAIN表达collate通过。float32/64组件检查克隆logits相同、参数隔离、单头干预/梯度隔离及七头有限后向通过；不是完整稀疏任务后向或效益证明。证据pipeline/analyses/y60_cpu_2191ef049a2f43ccbbdfb4456fd1baa6.json及原始日志。
- 按项目级操作授权仅提交七个已测Y60源码/YAML，commit6ac772a0c3eb4b5e598d25ce00c0ab7f203ce1e3，raw字节/Git blob及精确提交范围核验；无push、生产同步、准入、入队。其他脏文件未提交或回退。
- Y60原生两步计划已冻结，Python3.8解析和缺GPU入口拒绝通过；计划须使用真实TRAIN loader和原生train函数，两步内七头各调用两次并有有限非零梯度，测GPU峰值。尚未执行，见pipeline/checks/y60_smoke_2191ef049a2f43ccbbdfb4456fd1baa6_plan.json。
- 实时compute和running/claiming保留检查确认无独占空卡；Y60保留同ID资源阻塞、1小时复查及queue/running提前唤醒条件。Y58/Y59原阻塞未到期，未重复等价检查。tick_attempt已绑定Y60本轮实际CPU实施检查和资源证据；不强凑第二新候选。
- 13:11:11 GPU0 89C/风扇85%/17374of24576MiB，SW Thermal Slowdown Active；GPU1 66C/18758of24576MiB。内核读取成功且未发现新Xid/fallen-off；数据盘96%且剩157271826432bytes。未擅改功耗/风扇或清理数据；先前掉卡根因仍未知。资源与健康证据pipeline/checks/y60_resource_2191ef049a2f43ccbbdfb4456fd1baa6.json。
- 首次finish被拒绝：Blocked GPU candidate lacks authorized single-task server，租约仍持有。原资源证据已明确server-2 GPU0/1，Y60台账漏required_server；仅补required_server=server-2后重试。未改证据、模型、门槛或测试结论，不计为模型修订或任务训练失败。

## 5. Decisions Made
- 保留现有两项恢复训练及全部固定协议/历史负结果。当前不是缺人工操作授权，而是原生GPU检查需等待真实独占空卡；不抢占或共跑。
- Y60工程对照仅解除跨阶段头参数共享，不宣称创新、上涨或无害；完整稀疏任务、资源及训练效果均待实测。其余旧拒绝路线不重开。
- 空卡后优先续做较早准备的Y58/Y59，再推进Y60；用新的当前租约重核精确提交/YAML/goal/资源，原生两步检查通过即同轮准入并尝试既有门控派发。
- 本轮决策记录pipeline/decisions/20261021-131322_next_batch.md；随后执行canonical finish，以真实收据为准，不预称校验通过。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
