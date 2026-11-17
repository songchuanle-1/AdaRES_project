# Research Pipeline Report - AdaRES_project
快照时间：2026-11-01T22:52:18.757583+00:00；生成报告时快照年龄：777 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-02 06:52:18
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
================================================== 2026-11-02 06:52 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y76_stage_confidence_conditioning_res] Best=0.4948 12/30 no_improve=0/30 gpu=0 ? 序列: 0.4692, 0.4781, 0.4890, 0.4928, 0.4948

--- server-2 (192.168.1.163) ---
  [adares_y75_language_query_init_res] Best=0.5044 22/30 no_improve=4/30 gpu=1 ? 序列: 0.5044, 0.4968, 0.5024, 0.4999, 0.5019
  [adares_y74_language_update_gate_res] 进程未见 (deathwatch 1/3) — 暂不处置

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y74 | configs_song/model_paper5/adares_y74_language_update_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y75 | configs_song/model_paper5/adares_y75_language_query_init_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y76 | configs_song/model_paper5/adares_y76_stage_confidence_conditioning_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 130：complete=16 / rejected=79 / closed=12 / blocked=20 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 54（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- Y74自然完成固定30轮，完整Train/Val1..30、稳定finite last30/best/model/optimizer544/scheduler30与23、原源码/YAML/receipt/freeze及20生命周期守卫核验后canonical显式GPU0归档。原checkpoint留远端；训练best0.505369604在epoch23，不是正式FIRST。
- 释放的server2GPU0 fresh核独占idle/no compute/no reservation，保持280W，Y77精确原生YAML/tools.train真实TRAIN两步loss4.61551142/6.31049061、原geometry/mass/sample身份/每步12SCG位置消费/初始prediction/七共享头、独立公式首零次响应/有限原新梯度和参数更新通过。
- 16准入/资源/原数据/精确源版本证据拒绝守卫通过；原Y64完整training/evaluation深复制核等。主master仅选择七测试A文件，server2-only安全additive同步，已门控启动PID111334并确认Train1步数递增至300；原Y75/Y76 PID/源/YAML/receipt和训练保持。
- 新增Y75专用GPU1固定30轮终态准备，20生命周期和精确current498e源契约守卫、实际只读原六源码/YAML/receipt/freeze/声明目录20261101_020954核验通过。未加载checkpoint，不确认尚未结束的30轮，不归档或停止原训练。
- 三卡现均busy+reserved，Y78原准备与18项完整冻结FIRST/cost后评等待匹配独占卡。保存handoff和七段记忆，Y57暂停保持，无正式FIRST/成本更新。

## 5. Decisions Made
- 本轮实际native PASS后即完成Y77原完整freeze/准入/门控派发及真实进度确认；准入1/启动1/完成1在snapshot预算2内。
- Y74只归档一次不重训；两批native验证不代替效益/全量显存或完整后评协议。Y75专用收尾准备不是完成；Y78无GPU任务PASS不能入队。
- 保留科学goal字节/原数据/seed1999/fixed30/禁用/既有功耗和Y57专用暂停；后续observer用新y77精确契约，不改旧hash-bound helper，不碰server6/其他项目或调度频率。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
