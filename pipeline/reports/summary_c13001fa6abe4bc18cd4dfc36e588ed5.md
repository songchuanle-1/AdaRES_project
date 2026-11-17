# Research Pipeline Report - AdaRES_project
快照时间：2026-11-07T18:43:40.791673+00:00；生成报告时快照年龄：983 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-08 02:43:40
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
| server-1 | 0 | 未知 | 未知 | 未知 |
| server-2 | 0 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-11-08 02:43 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y90_query_peer_affinity_res] Best=0.5038 23/30 no_improve=8/30 gpu=0 ? 序列: 0.5027, 0.5004, 0.5038, 0.5024, 0.4998

--- server-2 (192.168.1.163) ---
  [adares_y91_hidden_confidence_gate_res] Best=0.4427 5/30 no_improve=0/30 gpu=0 ? 序列: 0.3348, 0.3787, 0.4297, 0.4427

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y91 | configs_song/model_paper5/adares_y91_hidden_confidence_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y90 | configs_song/model_paper5/adares_y90_query_peer_affinity_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 144：complete=16 / rejected=79 / closed=12 / blocked=35 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 68（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |
| AdaRES-Y67 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.676817+00:00 |
| AdaRES-Y70 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.677616+00:00 |
| AdaRES-Y73 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.678374+00:00 |
| AdaRES-Y75 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679091+00:00 |
| AdaRES-Y78 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679826+00:00 |
| AdaRES-Y81 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.680927+00:00 |

## 4. This Tick's Actions
- 仅处理 AdaRES_project；实际会话与 foreground owner 一致，后台 Pipeline 仍 Disabled。begin 已查进展并尝试派发现有合格队列；本轮 ready waiting 为0，原 Y90、Y91 两项训练保持。
- 实现独立 W2 变体 Y93：将原完整769维归一化竞争证据经零初始化769x256投影，残差加到原置信MLP第二隐藏状态，再交原最终线性层。原五层MLP、双norm、七级共享、query/mask/mass/overlap/context、数据/目标/损失/选择器/nativeYAML/seed1999/fixed30保持；不组合Y91/Y92，不重开旧拒绝或暂停机制，不宣称缺信息、创新或效果。
- Y93 实际原生 YAML、完整CPU模型、发布骨干、原TRAIN数据162738点/8表达加载首次通过。float32/64独立标量、查询/双通道置换、无RNG、零锚点、七级W2/权重/输入梯度、置信独占干预与精确旁路控制通过；14条真实证据/隐藏状态/最终线性消费者路径和70次原MLP层调用已记录，两个合成CPU解码器更新获得有限原/新/输入梯度。未初始化CUDA；这不是实际稀疏GPU训练、准入或效果证据。
- 仅将Y93精确测试的七个新增文件保留在当前已测试Y91的独立子提交；不包含尚未原生检查的Y92，生产HEAD/index/queue/原训练未变。10项实测证据守卫和6项运行环境守卫首次全通过、无跳过，15个CPU/native共用回调在CPU真实执行，两台授权GPU0原生检查计划已冻结。
- Y92原同字节15项绑定、Y90原25+6项终态计划、Y91原29项终态计划及全部原运行收据/冻结协议保持，仅核验引用完整性，不重放旧CPU/native/归档。本轮新增科学候选1、新CPU候选检查1、GPU原生任务0、准入0、训练启动0、终态归档0、旧CPU科学重放0。
- 按原300秒新鲜度规则，收尾脚本顺序刷新原生训练进度、两台真实物理占用和报告后执行官方finish；保留canonical resource_check真实tick_attempt，Y93实际CPU证据另行完整绑定。未新增正式FIRST、成本、效果或晋升结论。
- AdaRES-Y90 收尾实测：Train 24/30，步数 483/4686；原进程、来源、原生 YAML 和递增计数已核验。
- AdaRES-Y91 收尾实测：Train 6/30，步数 293/4686；原进程、来源、原生 YAML 和递增计数已核验。
- 本轮精确准备及实时资源记录：pipeline/checks/y93_preparation_closeout_c13001fa6abe4bc18cd4dfc36e588ed5.json

## 5. Decisions Made
- 不挤占Y90/Y91或外部进程。Y92在Y93之前；仅在匹配当前原实验自然fixed30结束、完整稳定有限last/best/model/optimizer/scheduler与TrainVal/source/YAML/receipt/freeze核验并归档，且匹配GPU0实时独占空闲后运行原生任务；真实通过才当轮准入并尝试派发。
- 当前HEADY91保持。Y92用既有bound runner，Y93原生计划尚未执行；若HEAD或当前owner先推进，则为同一科学字节另建显式测试的新绑定，不修改旧签名helper、不重放已通过CPU检查、不绕过原生门槛。
- server-2 GPU1原生检查/派发/训练/冻结后评及成本继续全部用户暂停，Y57另行实验暂停；原GPU1后评不得迁移或替代。仅server1 GPU0既有350W和server2 GPU0固定280W，各卡一个独立实验，禁止server-6。
- 原数据协议、发布骨干、完整冻结FIRST/同卡cost、所有旧负证据和专用恢复要求保留。无扫描、功率/调度/goal/门槛修改；常规项目内操作复用既有授权，无新增用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
