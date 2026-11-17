# Research Pipeline Report - AdaRES_project
快照时间：2026-10-24T20:50:52.118409+00:00；生成报告时快照年龄：628 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-25 04:50:51
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
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-25 04:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y61_peer_value_gradient_res] Best=0.4984 14/30 no_improve=0/30 gpu=0 ? 序列: 0.4890, 0.4946, 0.4959, 0.4914, 0.4984

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y61 | configs_song/model_paper5/adares_y61_peer_value_gradient_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 3；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 114：complete=16 / rejected=79 / closed=12 / blocked=6 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 39（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮核验实际会话与前台owner一致、后台调度Disabled；begin完成监控、目标检查与既有任务派发检查，waiting为0。沿用最新项目指令仅server-2 GPU0、一卡一实验、280W，保护Y57暂停及现有运行。
- 上海04:51:49实核Y61原PID78290及GPU0归属，Train14/30步数4561至4585/4686递增，82至83°C、22720MiB、280W；采样未见热降频、Xid/fallen-off或尾部错误。证据pipeline/checks/ef_gpu0_runtime_10ecd3bb7a2b4b78a7382ea479609ebc.json，SHA256 087cba0ae828c57a43ee5002273d3a63c09b5cb0f6083166a13335dc13a71d07。04:56:53新版本核验live快照已显示Train15/30 step255/4686，同PID与原生产marker9bea。GPU1外部计算未干预，硬件根因仍未知。
- 待跑不足时新增稳定Y63：保留W2所有原证据、共享头和七阶段损失，增加零初始化的两通道稠密焦点查询—竞争上下文双线性残差，固定除以sqrt(d_model)。只增131072参数、总164760709，不额外消耗RNG，不扫描通道/阈值/温度/损失/seed/轮数。现有MLP可隐式表示交互，本方案检验显式结构的拟合效果，不声称发现缺失信息或预先证明收益。定向旧记录检查未发现同类历史候选；保持F1等旧拒绝边界。
- 04:55:41至04:55:58实际CPU检查通过：原生YAML、完整CPU模型、发布SPFormer、真实TRAIN两场景162738点8表达；float32/64零锚点输出及query/mask输入梯度与W2逐位一致，初始参数/RNG一致，新权重梯度有限非零；字面交叉通道公式、转置轴负对照、排列/padding/单query/空支持和完整七阶段DEC前向/后向通过。证据pipeline/analyses/y63_cpu_10ecd3bb7a2b4b78a7382ea479609ebc.json及.log，结果SHA256 df369e771b68e21fd30d68c1cddb52fc40ce3b5e0760ce091163aa028e3000b9。8项本地守卫通过；CPU结果不是原生稀疏task optimizer检查、GPU资源上界或训练效益。
- 复用项目范围授权，仅将Y63已测7文件做本地提交82fbd9fb1fc56da6afee5b145f59cfa533464707，父60466af，精确index/blob/提交范围核验；冻结其原生两步GPU检查程序和计划。版本证据pipeline/checks/y63_version_10ecd3bb7a2b4b78a7382ea479609ebc.json，SHA256 c5c310889adc87ec8ef52d1411651c6f47b0fdf2b3d1b96195d6f20712f2a29b；计划SHA256 98ec9124cbeab6225ddc10fdf4b526f4b6108fa56e615db90146da1d831feb94。实时Y61占据GPU0，Y63保持blocked，未GPU原生smoke、准入、同步或启动。
- 为保留Y62在当前HEAD下的可执行计划，实际验证Y63提交只增其7文件，Y62七文件在旧/新commit与工作区完全相同、goal和训练入口未变；保留原CPU证据、版本及计划，另存绑定新HEAD的Y62计划，检查body原样。两项各7文件验证及Y63错误ID/旧commit负对照通过，证据pipeline/checks/prepared_sibling_alignment_10ecd3bb7a2b4b78a7382ea479609ebc.json，SHA256 13f1fa80d3f0b1a579e71ce0b83e0aef6219d269c580fd6711e5911d98b0a6d2。不是Y62模型修订或重测，queue/running未变，远端仍9bea。已更新七段紧凑记忆；完整六段报告由canonical report生成，postflight以后续finish及收据为准。

## 5. Decisions Made
- 继续保护Y61在GPU0独占训练及280W，Y57继续用户暂停；begin训练best 0.4984仅是过程记录，不代表冻结后评或晋升。无新增硬件故障处置或常规审批事项。
- Y62、Y63各自的精确已测代码/YAML和原生两步计划均保留，当前共同绑定HEAD82fbd9f。GPU0独占空槽后优先推进最近可准入候选，核验当前版本/goal/280W/真实占用后执行原生TRAIN两步loss/backward/path/resource检查；真实PASS才签收据、入队并尝试既有派发，不以CPU通过代替。
- 本轮新增1个独立候选并完成可行性检查及版本保存，未为凑满额度制造第二变体；Y61仍占GPU0，安全下一步等待空槽。四项已完成实验沿用原冻结后评与成本比较，不重训。无push、生产同步、GPU1/server-6、硬件设置、跨项目或调度操作；Y61终态须其匹配核验入口，不重放Y60 finalizer。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
