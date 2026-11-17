# Research Pipeline Report - AdaRES_project
快照时间：2026-10-24T04:50:02.699286+00:00；生成报告时快照年龄：473 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-24 12:50:02
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
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-24 12:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y60_stage_specific_confidence_res] Best=0.5075 23/30 no_improve=6/30 gpu=0 ? 序列: 0.4960, 0.5027, 0.5041, 0.5025, 0.5046

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y60 | configs_song/model_paper5/adares_y60_stage_specific_confidence_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 3；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 113：complete=16 / rejected=79 / closed=12 / blocked=5 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 38（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 上海2026-10-24 12:50:02取得design快照，核实实际CODEX_THREAD_ID与AdaRES owner一致、后台调度Disabled；experiment_first启用，running1/waiting0，无新增终态或操作故障。仅server-2 GPU0/280W，保留Y57暂停及外部GPU1任务，不跨项目或改调度。
- 12:50:52实核Y60原PID17186/命令/cwd/starttime/GPU0归属，Train23/30步1401至1424增长；GPU0为83至84C/280W/22746MiB，SW Thermal Slowdown仍Active，无本次训练尾部错误或新kernel Xid/fallen-off，不证明散热根因解决。证据pipeline/checks/ef_gpu0_runtime_e660e09f8a5d441aa2cc337f14b65e4e.json。
- 12:51:03生命周期v3实查唯一授权GPU0仍由Y60占用；Y56/Y58/Y59既有终态及不可变后评冻结证据一致，Y61七文件/9bea提交/原生计划/current goal一致。tick_attempt为Y56/resource_check/blocked，证据同时归入对应候选；未干预GPU1。证据pipeline/checks/gpu0_lifecycle_v3_e660e09f8a5d441aa2cc337f14b65e4e.json。
- 现有手动归档入口绑定旧Y59身份，新增scripts/finalize_y60_fixed30.py及test_y60_terminal_handoff.py；不改旧hash绑定工具。11项CPU schema/负控制及Python3.8语法通过，覆盖运行中、未满30轮、空optimizer/重置scheduler、错误ID/GPU1/暂停变化、错误检查点目录或非有限指标时拒绝相应归档。合成fixture仅是回归，不作训练完成证据。
- 仅实际执行新入口check，复核Y60原准入/6ac源码/冻结计划并从原生保存公告核实目录20261023_085324，保存原PID17186/config/log和目录的哈希绑定交接计划。没有加载检查点、停止训练或执行finalize，queue/running哈希及HEAD9bea不变。未来原进程确已消失才允许完整last30/best/model/optimizer/scheduler/终态日志核验后走canonical finalize_task，已canonical完成不重放。证据pipeline/checks/y60_terminal_handoff_e660e09f8a5d441aa2cc337f14b65e4e.json。
- 无新候选、模型/YAML修改、Git提交、原生CPU/GPU测试、生产同步、准入、入队或训练启动，无新增科学效益结论。Y60原训练继续、Y57暂停不动，Y61保留既有原生计划/准入派发入口。更新七段记忆后按canonical report/finish收尾，结果以实际收据为准，不预称成功。

## 5. Decisions Made
- 趁Y60仍在训练，准备它自己的守卫归档入口与目录绑定计划，不重放Y59 finalizer，不提前归档或伪造deathwatch。只有原进程确已消失、完整30轮日志和模型/optimizer/scheduler检查通过后才调用canonical归档；本轮只准备，不标记训练完成。
- 继续保护GPU0-only/280W、Y60固定30轮/seed1999及Y57暂停。空卡后按原冻结计划后评，Y61真实原生两步PASS后才准入派发；不改调度、资源授权、训练协议或历史结论。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
