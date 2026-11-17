# Research Pipeline Report - AdaRES_project
快照时间：2026-10-27T18:43:26.360591+00:00；生成报告时快照年龄：90 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-28 02:43:25
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
================================================== 2026-10-28 02:43 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y63_peer_bilinear_res] Best=0.5074 29/30 no_improve=7/30 gpu=0 ? 序列: 0.5000, 0.5003, 0.5026, 0.5026, 0.5012

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y63 | configs_song/model_paper5/adares_y63_peer_bilinear_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 3；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 117：complete=16 / rejected=79 / closed=12 / blocked=9 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 41（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮核验实际会话01a07e32-78bf-7670-925b-370911ea48db与项目owner一致；旧Windows Pipeline-AdaRES_project Disabled；前67b0收据FINALIZED/PASS/context_committed=true/lease_release=released已核。canonical begin为design，无新终态或操作故障，running1/waiting0/ready0。
- 上海10月28日02:43:34至02:43:39通过既有观察器核实Y63在server-1 GPU0由原PID3071037训练，第29/30轮3572到3579/4686递增；精确cwd/config、六模型源码、原生YAML确定性部署映射、marker82f、独占及原准入通过。证据pipeline/checks/y63_server1_progress_dab0f03a4b2848bcbcf795c731d974e9.json，SHAcea6ab90f947a45bb092f5c9ea25328bf08b40e0c759004dcab1450a523c692a。
- 实际canonical资源复查：server-2 GPU0仍是外来PID13813、97%/24138MiB；GPU1外来PID13814只读不操作。证据pipeline/checks/gpu_occupation_server-2_dab0f03a4b2848bcbcf795c731d974e9.json，SHA5bd4ff0930a420f8ded06b3def4b3fd85a05e46d66611b7475a621dc687378e3，绑定Y56/Y58/Y59/Y60/Y61/Y64/Y65。
- server-1 GPU0仍由Y63独占，47%/22756MiB；canonical证据pipeline/checks/gpu_occupation_server-1_dab0f03a4b2848bcbcf795c731d974e9.json，SHA95c4a0ca6d838741b58c579cd1cf66607a8f56606831d8a69b43329da9c7f905，绑定Y62冻结后评资源挂起。两台授权GPU无空槽；另次只读server-2 hostname dl01-MS-7C60/84%/24138MiB/84°C/280W，硬件未改。
- 保留前轮Y64/Y65已测试CPU版本与未执行原生两步计划、Y64未激活server-1方案。资源collector记录本轮resource_check/blocked；本轮无新候选/源码/commit/push/生产sync/准入/队列或running变更/启动。更新七标题memory；报告等待canonical finish，不预称通过。

## 5. Decisions Made
- 继续Y63固定30轮原训练，不以正常进度增长作为训练完成或新科学结论；当前不提前归档、不抢GPU执行原生测试、不迁移或重训。
- 实际授权空槽出现后优先Y64：server-1须先证实Y63完整epoch30训练/验证和稳定完整检查点并canonical归档，再激活既有替代计划/重核物理空闲；server-2按原生计划保持280W。实际native两步PASS后冻结后评/准入/入队并当轮派发，再推进Y65。
- 已有两项独立候选的版本及实际可执行原生检查计划已准备，下一步均需GPU。无重复CPU守卫/旧helper重放，不为额度制造重复变体，不把CPU准备当成准入或训练指标。
- 保留Y57/GPU1明确暂停和专用恢复、原数据/seed1999/fixed30/后评协议与bans；不使用server-6、不干扰外来任务、不跨项目、不改调度。当前仅健康运行与未变资源阻塞，无新增用户处理事项，通知静默。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
