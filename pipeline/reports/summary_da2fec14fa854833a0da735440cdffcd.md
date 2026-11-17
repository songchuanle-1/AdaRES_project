# Research Pipeline Report - AdaRES_project
快照时间：2026-11-09T06:43:09.825723+00:00；生成报告时快照年龄：685 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-09 14:43:09
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
================================================== 2026-11-09 14:43 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y94_query_geometry_bias_res] Best=0.4492 5/30 no_improve=0/30 gpu=0 ? 序列: 0.3315, 0.3607, 0.4306, 0.4492

--- server-2 (192.168.1.163) ---
  [adares_y93_evidence_hidden_skip_res] Best=0.5050 22/30 no_improve=1/30 gpu=0 ? 序列: 0.4971, 0.5010, 0.5041, 0.5050, 0.5046

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
- 仅AdaRES_project；实际owner、后台Disabled、正式skill/experiment_first、AGENTS/goal/SERVERS及GPU1暂停核验。先查询进展及派发现有合格队列，本轮ready waiting为0，无新终态。
- 运行：AdaRES-Y94 Train5/30，步1168/4686；AdaRES-Y93 Train23/30，步888/4686。原PID/source/YAML/receipt/私有runtime/精确数据、350W/280W及原生步数增长确认；两授权GPU0 canonical只读占用均busy+reserved。不抢占当前训练或GPU1外部任务，不加载live checkpoint，不停止或归档运行。
- 新准备：较早Y95的server2准入前边界检查17项首次通过，使用当前同字节七文件/HEADY94绑定/双原生payload、当前owner及原Y93完整冻结训练评测协议。缺失实际native 结果、Y93未完整结束归档、错误source/HEAD/goal/private runtime/GPU1/设备/280W/协议均拒绝。
- 完整原训练及9508表达141场景/candidate+Y36-W2-B1各两次/FIRST主报告/场景bootstrap10000-seed1999/TRAIN1487-3325/全部有害子组与指标/原设备cost协议绑定原哈希工件，仅内存继承测试，不注册Y95冻结计划、准入收据或原生结果fixture。
- Y95科学源/CPU结果程序/native callbacks/当前版本/双原生程序与计划/当前server2 激活不变。Y96/Y98与Y97原始及修订1数值失败保持；已完成CPU/version/binding/原生及历史终态不重放。Y93/Y94已有终态计划仅核验工件，未执行完整checkpoint核验。
- 本轮新科学ID0、科学CPU尝试0、GPU原生0、准入0、启动0、归档0；新操作测试组1计17项。精确备份七节记忆、刷新训练和canonical物理占用、保留collector实际resource_check，再正式report/finish，不预称收尾PASS。

## 5. Decisions Made
- 先推进较早Y95，不为凑额度再造或复制机制。只有HEADY94、匹配原Y93/server2 自然30轮完整有限checkpoint核验及canonical归档、实时独占空闲GPU0后执行原run_y95_bound_native_y94.py --server server-2。实际native通过后可用新只读verifier；源选择前仍须绑定实际Y95原生证据的新Y94终态加法兼容测试，继而当轮精确源选择/完整freeze/receipt/准入/canonical派发/递增counter。server1替代路径须自身匹配执行链。
- 科学goal字节、原签名收据、原数据/private runtime/mappings/seed1999固定30轮/禁用及专用恢复/功耗/调度/完整FIRST和原设备cost不变。GPU1全部使用及原GPU1 冻结后评继续暂停，Y57单独暂停；不迁移、不替代资源、不用server6、不跨项目、不干预外部任务。无需新用户操作。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
