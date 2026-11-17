# Research Pipeline Report - AdaRES_project
快照时间：2026-11-09T08:46:45.037481+00:00；生成报告时快照年龄：1619 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-09 16:46:44
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-Y91 — mIoU=0.5114 (gap to target: +0.0036)
       Epochs: ?, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0094

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5114, target=0.5150, gap=+0.0036
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
================================================== 2026-11-09 16:46 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y94_query_geometry_bias_res] Best=0.4677 7/30 no_improve=0/30 gpu=0 ? 序列: 0.3607, 0.4306, 0.4492, 0.4540, 0.4677

--- server-2 (192.168.1.163) ---
  [adares_y93_evidence_hidden_skip_res] Best=0.5050 25/30 no_improve=4/30 gpu=0 ? 序列: 0.5050, 0.5046, 0.4988, 0.5001, 0.4965

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y93 | configs_song/model_paper5/adares_y93_evidence_hidden_skip_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y94 | configs_song/model_paper5/adares_y94_query_geometry_bias_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 150：complete=16 / rejected=79 / closed=12 / blocked=41 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 7；收据文件 71（含修订或重试，不代表成功率）。
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
- 仅AdaRES_project。核验实际owner、Disabled后台调度、正式skill/experiment_first、AGENTS/goal/SERVERS/最新GPU1暂停后begin；先查询进展及派发现有合格队列，ready waiting为0。
- 运行：AdaRES-Y94 Train7/30，步4570/4686；AdaRES-Y93 Train26/30，步573/4686。原PID/source/YAML/收据/私有runtime/原数据、350W/280W与原生步数递增确认；canonical只读物理占用两个授权GPU0均busy+reserved。没有抢占、GPU1使用、live checkpoint加载、stop/restart/migration/新终态归档。
- 新准备：独立W2变体Y99，保留完整原静态掩码MLP并添加全部32维原稀疏输入的线性零初始化旁路，一次、+8192参数/no RNG、七阶段共用。全部原空间/外观输入及投影、坐标/采样/SCG、六次精炼/共享confidence/原损失和gate算法保留。与Y98外观直达、Y72循环visual、Y81 query-attended、Y92平方已投影特征不同；未做新数据/特征删减选择/阈值或验证集扫参、组合或重开旧拒绝。
- Y99首次实际CPU执行通过：原YAML/fullmodel/releasedbackbone/TRAIN162738点8表达，float32/64独立scalar/no-RNG/行及输入输出通道置换/零/符号/局部干预/双输入梯度，七阶段W2权重/RNG/输出精确锚点与输入梯度锚点，初始和最终mask干预、初始scores不变及exact bypass，完整原32维身份/两原visual投影/原batching/静态七consumer。两步合成更新有限，各387梯度张量，更新后mask-feature差0.002486199140548706。
- 仅七项已测新增文件保留在实际native通过的HEADY94父版本，使用独立index，主HEAD/index/现有训练/较早Y95-Y96-Y98原工件保持。10项实际证据和8项runtime guard首次通过；server1/server2独立原生YAML/私有环境/350W-280W/精确数据映射计划就绪。同一native callback已实际在CPU执行，GPU未执行，不冒充native PASS/准入或full-data memory。
- 本轮新科学ID1、科学CPU尝试1、GPU原生0、准入0、启动0、归档0。Y95执行绑定/server2准入前检查、Y96/Y98以及Y97原始和修订1数值失败均保持。精确备份七节记忆、刷新原训练及canonical占用、保留collector实际resource_check，再正式report/finish，不预称收尾结果。

## 5. Decisions Made
- 不强凑第二候选或伪造waiting。Y99排在Y95/Y96/Y98之后；仅当前HEADY94、匹配原Y94/server1或Y93/server2自然30轮完整有限原检查点核验/canonical归档、实时独占空闲GPU0后执行run_y99_native.py。未来HEAD或owner变化须新同字节测试绑定，不重放CPU/version/plan；实际native通过再当轮完整freeze/receipt/准入/精确source/安全派发/递增counter。
- 科学goal字节、原签名收据、原数据/私有runtime/mappings/seed1999固定30轮、禁用和专用恢复/功耗/调度保持。完整9508表达141场景/candidate+Y36-W2-B1各两次/FIRST/场景bootstrap10000-seed1999/TRAIN1487-3325/全部有害子组和指标/原设备cost不变。GPU1全部使用与原GPU1冻结后评继续暂停，Y57独立暂停；禁止server6/跨项目/外部任务干预。CPU可行性不是训练收益，正式效果待原冻结评测，无新用户操作需求。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
