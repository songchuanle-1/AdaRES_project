# Research Pipeline Report - AdaRES_project
快照时间：2026-10-19T10:48:24.562609+00:00；生成报告时快照年龄：1841 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 18:48:24
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
================================================== 2026-10-19 18:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y54_fine_support_res] Best=0.4462 5/30 no_improve=0/30 gpu=0 ? 序列: 0.3393, 0.3755, 0.4315, 0.4462
  [adares_y52_point_target_res] Best=0.3683 3/30 no_improve=0/30 gpu=1 ? 序列: 0.3354, 0.3683

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y52 | configs_song/model_paper5/adares_y52_point_target_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y54 | configs_song/model_paper5/adares_y54_fine_support_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 0、未分类阻塞 1。
- 非终态候选与排队项合计 1；总上限 8。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 108：complete=14 / rejected=79 / closed=12 / training=2 / blocked=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 33（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮18:48:24上海快照为design/reserve_only，准入预算0、新增储备预算1；规范begin已各运行一次monitor/goal，无重复preflight。0-3节保留这一快照，后续训练核实只写本节。
- 先登记唯一新稳定候选Y57 diagnosis并冻结源码/缓存/严格W2权重、焦点0、Householder方向、32表达与两dtype算术界，再于19:02:40-19:02:46执行单线程CPU检验；同一候选推进为design储备，不新增模型、队列、准入或训练。
- 实际产出为8个固定TRAIN场景的32表达在float32/64下64组匹配对照和2个同peer阴性：64/64保留焦点、mask、点数、竞争权重及每query均值/方差，在唯一query128保持时二阶矩变化可分；原生焦点logit误差最大9.536743e-7/1.776357e-15，重复0、排列最大1.907349e-6，均符合预注册算术界。64组不是64个独立场景或训练实验。
- 科学结论仅为peer均值的表示容量见证；反射未证明decoder可达，更未证明任务伤害。mask和GT不变时真实IoU也不变，故置信度不变不能作预测错误。旧捕获缺少采集时权重blob SHA的来源限制保留；本轮严格头回放不能追认旧缓存权重。
- 19:05:44-19:05:50只读双采样：原Y52 PID69064/GPU1第4/30轮912到921/4686，原Y54 PID47332/GPU0第5/30轮2223到2232/4686；cwd/config精确、各卡有各自compute PID，22716/22718MiB，最近有界tail无异常。空闲授权卡0；没有GPU诊断或训练启动。
- 新助手通过错误token拒绝及台账前后哈希不变测试；Python3.8语法/compile通过。本轮未改共享流水线/运行模型，未重做旧125项回归。
- 收据尚未生成时用量为未知；最终收据窗口仅begin.started_at至receipt.recorded_at内可见响应，排除早读/晚到/最终答复，缓存属于input、推理属于output，不重复加总或当账单。实际新增ID1、模型修订0、准入0、训练启动0；最终变化计数以收据为准。

## 5. Decisions Made
- 本轮有界储备准备完成：仅新增并实际检验Y57；保留任务相关性未知，不把表示容量当效果/新颖性或准入。下一项先在源码/CPU核实同8场景point-GT和几何/点数绑定，再冻结真实TRAIN任务相关及可达干预测试；不制造GPU阻塞掩盖这项CPU工作。
- Y56不重复旧CPU检验，不改旧负证据、七层batch2冻结程序或资源条件；Y52/Y54保持当前fixed30。只操作AdaRES，不改目标、准入门槛、授权、其他项目或调度，不设全路线耗尽/无信息增益冷却，不硬等deadline。
- 完整六段审计与七段记忆在当前租约内留档；本报告不预称postflight通过。只有规范finish成功并独立核实租约/pending释放后才称本轮完成；否则保留证据并abort核实释放。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
