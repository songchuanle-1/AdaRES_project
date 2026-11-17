# Research Pipeline Report - AdaRES_project
快照时间：2026-11-07T20:45:52.859245+00:00；生成报告时快照年龄：722 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-08 04:45:52
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
================================================== 2026-11-08 04:45 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y90_query_peer_affinity_res] Best=0.5038 26/30 no_improve=11/30 gpu=0 ? 序列: 0.5024, 0.4998, 0.5031, 0.5013, 0.5007

--- server-2 (192.168.1.163) ---
  [adares_y91_hidden_confidence_gate_res] Best=0.4703 8/30 no_improve=0/30 gpu=0 ? 序列: 0.4297, 0.4427, 0.4615, 0.4657, 0.4703

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
- 候选累计记录 145：complete=16 / rejected=79 / closed=12 / blocked=36 / training=2。
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
- 仅处理 AdaRES_project；实际会话与foreground owner一致，后台Pipeline仍Disabled。begin已查进展并尝试派发现有合格队列；ready waiting为0，原Y90/Y91各占授权GPU0并继续递增。
- 优先推进已有Y92最近的执行缺口，新建server1 GPU0/350W准入、原完整冻结后评协议继承、精确七文件来源选择、后续Y91/Y92观察及canonical派发工具。只从保留的旧工具做显式AST特化；原科学模型、YAML、CPU证据、原生程序主体和现有生产HEAD/index/queue未改。
- 17项真实操作守卫首次全部通过、无跳过：同字节七新增文件、现有Y90/Y91来源链、错分支/错候选/改动文件拒绝、两套原生payload、私有运行环境及功率、完整原Y90训练/后评协议仅方法ID替换、完整终态前置、物理占用/租约拒绝和真实缺失native结果拒绝。
- server1所选payload仅在内存副本核验，不修改Y92当前required_server或原native_smoke_plan。全量实际native准入测试仍未执行，且其setup已实际确认会拒绝缺失的native结果；没有伪造native输出、准入收据、同步、入队或启动。
- Y92既有15项同字节绑定、Y93原CPU/版本/10+6项计划、Y90原25+6项及Y91原29项终态计划保持，不重放旧科学检查或终态归档。本轮新增科学候选0、CPU科学运行0、GPU原生任务0、准入0、训练启动0、归档0；只完成可复用的操作准备。
- 收尾按原300秒标准顺序刷新真实原生进度、两台物理占用、绑定审计和官方report/finish，保留canonical resource_check的真实tick_attempt，不修改科学目标字节或旧运行收据。未新增正式FIRST、成本、效果或晋升结论。
- AdaRES-Y90 收尾实测：Train 26/30，步数 1959/4686；原进程、来源、原生 YAML、功率与递增计数已核验。
- AdaRES-Y91 收尾实测：Train 8/30，步数 3936/4686；原进程、来源、原生 YAML、功率与递增计数已核验。
- 本轮操作准备及真实资源记录：pipeline/checks/y92_execution_closeout_9a94dd03b9774a71a10001ed4fd7fc5b.json

## 5. Decisions Made
- 保留原Y90和Y91；不挤占运行GPU或外部任务。Y92在Y93前，等匹配Y90自然fixed30结束、last/best/model/optimizer/scheduler稳定有限、TrainVal/source/YAML/receipt/freeze完整核验并归档，再用实时独占空闲server1 GPU0执行已冻结bound native计划。
- 真实Y92 native PASS后，须用该实测证据为仍在跑的原Y91新建并测试支持附加Y92的精确终态绑定，再选择Y92版本、执行actual-native准入和canonical派发、核验递增步数；可执行时在同轮完成。旧签名helper不修改；当前仍用y91_deployment观察Y90/Y91。
- 本轮操作准备仅覆盖server1 GPU0/350W。server2 GPU0原生备选计划原样保留，若将来选择则另补匹配操作绑定；不替换任何历史冻结后评资源。Y93科学字节和原计划保留，若HEAD/owner推进先做新的显式同字节绑定。
- server2 GPU1全部AdaRES使用继续用户暂停，Y57另行实验暂停，原GPU1冻结后评/成本继续挂起，不迁移、不替换。仅server1 GPU0既有350W与server2 GPU0固定280W，各卡一个独立实验；禁止server6、不改调度/功率/goal/门槛/数据或专用恢复要求，无新增用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
