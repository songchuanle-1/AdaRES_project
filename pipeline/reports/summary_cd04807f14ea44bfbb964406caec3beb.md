# Research Pipeline Report - AdaRES_project
快照时间：2026-10-25T20:46:31.006317+00:00；生成报告时快照年龄：326 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-26 04:46:30
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
================================================== 2026-10-26 04:46 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y62_overlap_density_res] Best=0.4834 8/30 no_improve=0/30 gpu=0 ? 序列: 0.4317, 0.4553, 0.4604, 0.4667, 0.4834

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
- 候选累计记录 115：complete=16 / rejected=79 / closed=12 / blocked=6 / training=1 / admission=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 41（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 快照2026-10-25T20:46:31.006317+00:00，核实际会话所属与Disabled后台后canonical begin先查询进展/派发既有队列，无新增终态或运行故障；前轮4c8595收据已核FINALIZED/PASS/committed/released。
- 上海04:46:57至04:47:02核Y62同PID2540376/server-1 GPU0，第8/30轮3401至3409/4686增长，六文件/原生YAML/marker一致，bounded尾部无错误。证据pipeline/checks/server1_training_progress_cd04807f14ea44bfbb964406caec3beb.json，SHAb3a5bb6cbf85d007869aed800e1263d0290748edf078385aee01e7c5d4c8586e。
- 为待跑缺口实现Y64有向包含置信度：I_ij/A_j-I_ij/A_i聚合peer query，零初始化256x2投影，新增512参数，原生YAML逐字节与W2相同、独立七文件/唯一basename。靶向历史机制核查保留所有拒绝路线；本轮HEAD不变，未提交或生产同步。
- 上海04:49:52至04:50:13完成Y64真实CPU程序：完整模型/发布SPFormer/原生YAML/真实TRAIN collate（162738点、8表达）；float32/64精确锚点/RNG/输入梯度、字面公式与方向负对照、非均匀点质量、置换/空/零质量、质量缩放/超点拆分以及七阶段合成DEC后向均通过。证据pipeline/analyses/y64_cpu_cd04807f14ea44bfbb964406caec3beb.json，SHA42ad108f530737999ee728429fa895a140c571333039e377eb3ea46cadaedb9e；精确源manifest checks/y64_cpu_version_cd04807f14ea44bfbb964406caec3beb.json，SHAe768af8f5567fb86cdd561d5224f4b201e8d7cb23d3ede98d319eac11f168b1b。
- 于2026-10-25T20:50:52.797214+00:00用新label y64_finalize实测server-2 GPU0仍被外来PID13813占用，100%/24136MiB，无授权空槽。证据pipeline/checks/gpu_occupation_server-2_y64_finalize_cd04807f14ea44bfbb964406caec3beb.json，SHA326b213cfefa9770449ca526b39502ff488f8320daf28c8be5e255932b77b562，原始日志与六项候选绑定；旧采集证据保留。

## 5. Decisions Made
- Y62现有训练继续，Y63同卡waiting；Y64 CPU准备通过后挂起等待原生GPU检查。保留Y63准入版本和HEAD，不将CPU结果代签真实稀疏task两步loss/backward/optimizer或训练效果，无新准入/启动。
- Y64未使用validation，也没有调整阈值、温度、损失、特征子集、seed或fixed30；q可能已隐式编码尺寸，结构能否改善置信度和性能仍未知。其方向边不同于Y50高阶覆盖或Y62密度，不重开历史拒绝路线。
- 资源空闲后核精确源manifest，保留Y63版本再保存限定提交/冻结原生计划，执行至少两步原生YAML/TRAIN及每步七次分支/有限非零梯度检查，并冻结与Y62/Y63一致的完整后评协议，通过才准入及门控派发。server-2 GPU0维持280W，Y57/GPU1暂停，不用server-6或干扰外来任务。
- 本轮有一项实际新准备成果，向用户简报Y62运行、Y64 CPU检查通过及GPU检查等待；没有需要新增用户确认的事项。生成完整六段报告后finish并核收据/释放，释放后不写项目状态。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
