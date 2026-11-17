# Research Pipeline Report - AdaRES_project
快照时间：2026-10-27T02:44:07.213240+00:00；生成报告时快照年龄：432 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-27 10:44:06
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
================================================== 2026-10-27 10:44 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y63_peer_bilinear_res] Best=0.4891 12/30 no_improve=1/30 gpu=0 ? 序列: 0.4569, 0.4678, 0.4790, 0.4891, 0.4889

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
- 核验AdaRES_project实际会话归属及当前AGENTS/goal/SERVERS/execution_mode，Pipeline-AdaRES_project保持Disabled。canonical begin为design，running1/waiting0/ready0，无新terminal或连接/部署故障。
- 上海2026-10-27 10:44:41至10:44:46实测Y63在server-1 GPU0同PID3071037，第12/30轮2818至2827/4686步增长；精确cwd/config、六模型源码、原生YAML确定性路径适配、marker82f与GPU0独占compute均通过。证据pipeline/checks/y63_server1_progress_975bfddfe5334e868527ad637cba467f.json。
- 优先推进既有Y64，新增原生两步检查脚本probe_y64_native_smoke.py：使用原生YAML/真实TRAIN/tools.train至少两步loss/backward/optimizer，逐步核验七次头调用、11头参数梯度、查询/掩码梯度、有向context公式、首步零锚点及第二步非零参数和实际logit响应、峰显存；强制授权GPU0、server-2正确hostname和280W。脚本尚未在GPU运行。
- 执行7项CPU检查脚本守卫测试，拒绝缺少/额外阶段、缺第二步、非零初始锚点、未执行响应、零权重、NaN/Inf/负值/伪布尔和未授权GPU，全部通过。真实native loss/backward仍未执行，守卫通过不代替原生GPU检查。测试原始日志pipeline/checks/y64_native_plan_guards_975bfddfe5334e868527ad637cba467f.log。
- 在项目锁下确认无用户已暂存内容、无waiting/claiming队列后，仅保存CPU测试过且逐字节相同的Y64七文件为提交e27895f45b6f172454c50d40d1c0184c6aa8dfcb，父提交82fbd9f。核验提交范围仅七文件、与原CPU manifest/当前goal/原训练入口一致，旧goal与篡改源码负对照拒绝通过。已冻结FROZEN program/native plan/version，plan为pipeline/checks/y64_native_975bfddfe5334e868527ad637cba467f_plan.json。没有push、生产同步、GPU执行、准入或入队。
- 保存Y64版本后再次校验Y63原准入有效，queue/running哈希未变，生产Y63 marker保持82f。本轮两服canonical occupation：server-1 GPU0独占Y63；server-2 GPU0外来PID13813、95%/24138MiB，无授权空槽。单次直读server-2 GPU0上限280W/87°C。Y57暂停与外来任务保留，六项冻结后评及Y64/Y65实际GPU检查继续等待资源。

## 5. Decisions Made
- 本轮推进已有Y64从CPU准备到有精确提交和可执行原生检查方案；没有新增变体或训练效果结论。Y65继续保留既有CPU准备，不为填额度增加重复候选。
- GPU0空闲后先validate_frozen核当前goal/七文件/提交/原训练入口，再以新token及明确--gpu --gpu-id 0执行Y64冻结plan。至少两步真实native loss/backward/optimizer、七次路径调用和新梯度/实际响应通过后，冻结原全量后评协议，才可准入并当轮尝试门控派发。
- 本地HEAD仅增加Y64七文件；Y63仍使用原82f版本，原准入及生产文件保持，运行进程不重启。Y57/server-2 GPU1继续用户暂停，server-2 GPU0保持280W，禁止server-6、驱逐任务、跨项目和调度更改；保持seed1999/fixed30、原数据及全部禁用项。
- 七节记忆已更新，接下来canonical report/finish并核收据/锁/pending；当前记录不提前宣称postflight通过，释放后不补写。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
