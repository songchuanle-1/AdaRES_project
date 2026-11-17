# Research Pipeline Report - AdaRES_project
快照时间：2026-10-25T22:51:17.395123+00:00；生成报告时快照年龄：59 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-26 06:51:16
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
================================================== 2026-10-26 06:51 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y62_overlap_density_res] Best=0.4884 11/30 no_improve=1/30 gpu=0 ? 序列: 0.4667, 0.4834, 0.4779, 0.4884, 0.4875

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
  server-1: running=1 waiting=1
    待推: [AdaRES-Y63] configs_song/model_paper5/adares_y63_peer_bilinear_res.yaml gpu=0
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y62 | configs_song/model_paper5/adares_y62_overlap_density_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 1 | AdaRES-Y63 | configs_song/model_paper5/adares_y63_peer_bilinear_res.yaml | waiting | - | server-1 |
- 活跃 2（运行 1 / 认领 0 / 等待 1）；目标 4；准入预算 2。
- 研究储备 1 / 3；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 1；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 116：complete=16 / rejected=79 / closed=12 / blocked=7 / training=1 / admission=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 41（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 快照2026-10-25T22:51:17.395123+00:00：核验实际会话属于AdaRES_project，后台Pipeline-AdaRES_project Disabled，canonical begin已查询进展和检查既有派发；running1/waiting1/ready1，无新增终态或运行故障。前cd048轮收据已核FINALIZED/PASS/committed/released。
- 上海10月26日06:51:36至06:51:41核Y62同PID2540376/server-1 GPU0，第11/30轮596到605/4686增长，六文件/原生YAML/源码marker一致，bounded尾部无错误。证据pipeline/checks/server1_training_progress_bdc67d0eebfc46338a3bc6124b92379a.json，SHA256 150a3af32bdf3b2badfecbf26934185f12dc4df972f3ea1141907d9142f30d42。
- 于2026-10-25T22:51:34.806066+00:00实测server-2 GPU0被外来PID13813占用，100%利用率、24138/24576MiB，无授权空槽。canonical证据pipeline/checks/gpu_occupation_server-2_bdc67d0eebfc46338a3bc6124b92379a.json，SHA256 d13fb4a389beadaa968e5b19d4892c0588b2159014151dc5d1f71469fb330396；原始日志绑定Y56/Y58/Y59/Y60/Y61/Y64，真实resource_check/blocked已保存。
- 核Y64既有CPU源manifest的7文件哈希逐项一致，HEAD仍82fbd9fb1fc56da6afee5b145f59cfa533464707。更新七段紧凑记忆，将前轮Y64 CPU准备明确标为历史，不重复执行其prepare/record或CPU测试。

## 5. Decisions Made
- Y62现有训练继续，Y63已准入并等候server-1 GPU0；Y64保持CPU准备完成、原生GPU两步检查待跑，无新增候选/提交/生产同步/准入/启动。保留Y63版本，空槽后再推进Y64限定版本/原生计划及真实检查。
- 五项fixed30完成实验的全量冻结后评/成本比较继续等待授权独占资源；server-2 GPU0保持280W，Y57/GPU1用户暂停，禁止server-6及干扰外来任务。不把CPU后向、正常训练步数或静态Unknown描述当作训练效益证据。
- 当前是正常训练推进与原有资源等待，未出现需新增用户操作的实质变化；本轮心跳静默。保存完整六段报告，finish后核FINALIZED/PASS/committed/released和锁/pending缺失，释放后不补写。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
