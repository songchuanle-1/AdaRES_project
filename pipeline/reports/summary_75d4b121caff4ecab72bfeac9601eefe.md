# Research Pipeline Report - AdaRES_project
快照时间：2026-11-06T06:51:52.457268+00:00；生成报告时快照年龄：1411 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-06 14:51:51
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
================================================== 2026-11-06 14:51 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y88_mask_query_quadratic_res] Best=0.4945 11/30 no_improve=0/30 gpu=0 ? 序列: 0.4664, 0.4647, 0.4802, 0.4828, 0.4945

--- server-2 (192.168.1.163) ---
  [adares_y87_self_language_product_res] Best=0.5013 16/30 no_improve=1/30 gpu=0 ? 序列: 0.4912, 0.4918, 0.4970, 0.5013, 0.4960

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y87 | configs_song/model_paper5/adares_y87_self_language_product_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y88 | configs_song/model_paper5/adares_y88_mask_query_quadratic_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 143：complete=16 / rejected=79 / closed=12 / blocked=34 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 65（含修订或重试，不代表成功率）。
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
- 核验实际所属会话、foreground和experiment_first、Disabled后台调度、最新GPU1全用途暂停与Y57单独暂停；仅使用server1GPU0既有350W和server2GPU0既有280W，不跨项目或改变调度。
- begin已查进展、目标并尝试既有合格队列派发，无新终态或就绪waiting，无连接/部署故障；原Y82/Y83等已归档产物未重启、未重放终态helper。
- 先观察并在准备后再次确认原Y87/Y88源码、YAML、原收据、部署marker、功耗和原生步数递增；最新Y87为Train17/30、1422/4686步，Y88为Train12/30、893/4686步，均保留原PID和运行。
- 当前运行尚无终态接管助手，新增绑定原始Y87/Y88的8个专用生命周期文件及共享契约/生成器，沿用已验证的终态流程；原Y84/Y85等签名助手原字节保持，不修改模型/YAML或生产HEAD。
- 仅做只读原日志保存声明与源码/配置/marker/活进程身份定位，精确绑定Y87原目录20261105_142459、Y88原目录20261106_053553，不从日志文件名推断目录。没有加载checkpoint、远端写入、停止训练、改队列或归档。
- 两份终态接管计划各自实际通过25项CPU守卫，共50项且无skip。测试夹具不是真实终态证据；真实归档仍须原PID和同配置训练进程消失、Train/Val1至30完整、稳定有限last30/best/model/optimizer/scheduler以及原源码/YAML/收据/freeze全部核验。
- 专用归档严格使用原服务器GPU0映射与原功耗约束，保留所有checkpoint和其他队列项，拒绝GPU1或替代目录；未执行finalize_y87_fixed30.py或finalize_y88_fixed30.py。
- Y89/Y90/Y91原保存版本、CPU证据和GPU0原生计划保持，三者仍未GPU原生/未准入；canonical物理证据确认两张授权GPU0仍由Y87/Y88占用，外部GPU1任务未动。收尾前刷新当前token的新标签占用证据。
- 本轮科学新候选0、准入0、启动0、归档0、正式FIRST更新0；实质产出为两个原运行的终态接管计划和50项守卫，不将其算作新模型或训练完成。

## 5. Decisions Made
- 优先补齐已有候选的真实必要前置操作。保留Y89、Y90、Y91顺序；原属运行自然fixed30完成且完整checkpoint核验/归档、授权匹配GPU0实际独占空闲后，才执行已保存的原生两步检查并据真实结果准入/派发。
- 终态计划当前只认可已登记的Y87/Y88加法部署链。若生产HEAD先推进，须保存旧计划与原助手，增加显式加法版本绑定及新助手后再核验，不就地改写签名程序、不reset、不替换原运行或目录。
- 正式冻结FIRST仍Y36=0.510725438294，目标0.5150。训练期best、50项CPU守卫和原目录存在均不代表完整checkpoint已验证、正式效果、成本结果或晋升。
- 继续保持原29个fixed30模型全量冻结后评和同设备成本协议；GPU1后评不迁移，Y57单独暂停，Y86数值实现挂起不重放、不弱化控制。科学goal原字节、原收据、原数据/私有环境/映射/seed1999固定30轮/功耗/调度全部保持。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
