# Research Pipeline Report - AdaRES_project
快照时间：2026-10-30T22:52:26.355188+00:00；生成报告时快照年龄：1486 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-31 06:52:25
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
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-31 06:52 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y69_coarse_geometry_bias_res] Best=0.4995 20/30 no_improve=6/30 gpu=0 ? 序列: 0.4982, 0.4971, 0.4927, 0.4993, 0.4952

--- server-2 (192.168.1.163) ---
  [adares_y68_language_mask_modulation_res] Best=0.5040 30/30 no_improve=8/30 gpu=0 ? 序列: 0.4965, 0.4974, 0.4949, 0.4949, 0.4969
  [adares_y70_query_scale_fusion_res] Best=0.4997 17/30 no_improve=1/30 gpu=1 ? 序列: 0.4886, 0.4922, 0.4989, 0.4997, 0.4951

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y68 | configs_song/model_paper5/adares_y68_language_mask_modulation_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y70 | configs_song/model_paper5/adares_y70_query_scale_fusion_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y69 | configs_song/model_paper5/adares_y69_coarse_geometry_bias_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 1；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 125：complete=16 / rejected=79 / closed=12 / blocked=14 / training=3 / implementation=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 48（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮begin/design先核原三任务，Y68第30轮/Y70第17轮/Y69第20轮原生递增；保留原数据/源码/YAML/收据/功耗和Y57暂停，初始没有合格waiting。
- 继续既存Y73：双卡原生YAML/TRAIN/tools.train程序绑定原7da625版本及280W/既有390W；11项CPU证据守卫/stale goal和篡改source拒绝通过。实际CPU完整模型/发布backbone/真实TRAIN162738点8表达+合成解码器两次原optimizer更新执行冻结观察回调，六对象身份/预测消费/七indicator/387有限梯度张量与10头梯度/独立公式/首零次响应及新参数真实更新通过。Y73未GPU native/未准入，不作效益结论。
- 原Y68自然完成Train/Val1..30及PID20742退出，完整日志和稳定finite last30/best21/model/optimizer544/scheduler30或21、原source/YAML/receipt核实，12生命周期守卫通过后canonical done_complete/30/30归档。训练best mIoU .5039549469947815/epoch21，权重留原远端，完整日志拉回；正式完整冻结后评/成本尚待资源。
- 上海07:09:47—07:10:14，fresh独占idle/无预约GPU0和280W通过后Y71真实原生两步PASS：loss4.615511417388916/6.311269283294678，四不同场景16表达、134280/350887点；每步540有限梯度张量/10共享头梯度/7indicator，一次原3D/2D投影→产品残差→sum_norm真实消费/独立公式，有限非零新梯度和实际权重更新，峰6729760768B；无val选择/诊断权重保存，非效益或全量显存保证。
- 14项原生准入负守卫通过，原Y64完整训练和9508表达/141场景/FIRST/四方法两次/scene-bootstrap10000/TRAINcuts及harm/同卡成本协议深复制核等，生成Y71新receipt并同轮入队、既有安全sync/门控dispatch。生产39b→87019仅七个Y71新文件，原Y70源码/YAML/收据/PID52561与server1 Y69保持；Y71新PID53714，上海07:13真实Train1/30从69增至79/4686。
- 新增exact_deployment observer与6项身份/marker负守卫通过；部署前后原Y70实际递增，最终Train18/30 210→219/4686，原生源码与既有390W核实。Y71同observer核原生递增/源码/280W；旧39b-only Y70 observer作为历史保留，今后用新helper。
- Y68可变required_gpu1与原freeze/next_action GPU0/280W不一致，已留原前态并仅对齐为原GPU0资源等待，旧finalizer/冻结协议/receipt不改。fresh canonical occupation核53714/52561/1135633三卡忙且预约，12项原完整冻结后评及Y72/Y73 native继续等资源。
- 本轮新增候选0、准入1/派发1。记忆精确备份后整理为50行；handoff=pipeline/checks/y68_y71_handoff_fb6718a8dd684cdfbaa98d6678e7d3c5.json，Y57独立暂停/调度频率保持，无必须用户处理项。

## 5. Decisions Made
- Y68固定30和完整产物核实是训练完成，.50395仅训练日志最好值；不以该数或Y71两步loss/CPU梯度宣称正式收益或晋升，12项原全量冻结评估协议保持。
- 按本轮准入预算2使用1个名额：Y71真实native PASS后原完整后评冻结、准入、既有门控派发及原生递增核验同轮完成；继续原Y70/Y69和独立暂停Y57，不重放已完成操作。
- 当前生产87019后，Y70和Y71只能用新exact_deployment observer核精确源码/PID；保留旧hash-bound程序。未来切至Y72/Y73 retained commit前先保护/重绑存活source/observer/deployment契约，Y72/Y73实际GPU native仍是准入必要条件。
- 复用已有项目授权，维持原goal字节/原数据/seed1999/fixed30/禁用项/原收据/280W、390W、350W，不跨项目、不用server-6、不改频率或硬件，不驱逐/重启受保护运行。Y57恢复须单独明确授权。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
