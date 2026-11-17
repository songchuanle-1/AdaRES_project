# Research Pipeline Report - AdaRES_project
快照时间：2026-11-09T16:51:19.389346+00:00；生成报告时快照年龄：572 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-10 00:51:18
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
================================================== 2026-11-10 00:51 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y94_query_geometry_bias_res] Best=0.4982 16/30 no_improve=1/30 gpu=0 ? 序列: 0.4938, 0.4914, 0.4976, 0.4982, 0.4979

--- server-2 (192.168.1.163) ---
  [adares_y95_query_set_mean_res] Best=0.4428 5/30 no_improve=0/30 gpu=0 ? 序列: 0.3422, 0.3952, 0.4314, 0.4428

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y95 | configs_song/model_paper5/adares_y95_query_set_mean_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y94 | configs_song/model_paper5/adares_y94_query_geometry_bias_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 151：complete=16 / rejected=79 / closed=12 / blocked=42 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 7；收据文件 72（含修订或重试，不代表成功率）。
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
- 仅AdaRES_project。核验实际owner会话、Disabled后台调度、正式skill前台和experiment_first规则、AGENTS/原goal字节/SERVERS/最新GPU1暂停授权后，一次begin先查询进展和尝试派发现有合格队列；ready waiting为0，无新终态或操作故障。
- 运行：原Y94 PID218513/server-1 GPU0/350W为Train16/30 step3525/4686；原Y95 PID34484/server-2 GPU0/280W为Train5/30 step4376/4686。当前token只读观察确认原source、YAML、marker、签名收据及原生计数递增，未停止或重启。
- canonical gpu_occupation分别实际查询两服务器hostname、物理GPU及compute PIDs，并绑定本轮token/快照/候选。两个授权GPU0分别仍由原Y94/Y95占用且canonically reserved，没有可独占空闲卡；保留collector真实resource_check，未冒充原生任务检查或科学PASS。
- 已有Y96/Y98/Y99三个CPU准备候选等实际GPU原生检查，优先Y96；其当前HEADY95和Y94/server1-Y95/server2绑定、Y95原目录20261109_081912专用终态计划继续保留。未重放已完成CPU、版本、绑定、数值控制或原生计划测试，未添加占位变体。
- 本轮新科学ID0、CPU科学尝试0、GPU原生0、准入0、启动0、归档0、正式评测0。保留Y97原始及修订1负证据。精确备份七节memory并更新本轮训练和资源记录；随后官方report/finish，以实际收据和租约释放为准，不预签结果。

## 5. Decisions Made
- 本轮没有新科学决策。三个已准备候选优先且原训练仍占用全部授权GPU0，不凑额度重复设计或测试、不抢占、不伪造waiting，不因候选计时器到期取得GPU或Y57恢复权限。健康运行、资源阻塞及无行动项保持静默。
- 下轮先观察原Y94/Y95并尝试已准入工作。Y94自然fixed30结束后按finalize_y94_fixed30_y95.py完整验证精确目录20261109_111838；Y95按finalize_y95_fixed30.py验证20261109_081912。均须原进程退出、完整Train/Val1..30与稳定有限last/best/model/optimizer/scheduler后canonical归档，日志结束不代替checkpoint校验，不重放Y93或旧已归档终态。
- 仅当前HEADY95、匹配原owner完成精确终态并归档、现场独占空闲未保留GPU0后执行run_y96_bound_native_y95.py。实际native通过后再新建实际证据绑定的另一在跑任务终态兼容工具，并当轮完成设备匹配完整原freeze/receipt/准入/精确source/安全派发/原生计数递增。Y98/Y99未来须新测试绑定，不盲目使用旧HEAD/owner计划。
- 保持科学goal字节、全部旧签名收据、精确原数据/私有runtime/部署映射/seed1999固定30轮及完整9508表达141场景/FIRST重复比较/全部有害子组/原设备成本协议。全部server-2 GPU1使用及原GPU1冻结后评继续暂停，不迁移替换；Y57独立暂停原epoch27状态。server1沿用350W、server2GPU0保持280W，每卡一实验；不跨项目、不用server6、不改调度或功耗、不干预外部任务。无新增需用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
