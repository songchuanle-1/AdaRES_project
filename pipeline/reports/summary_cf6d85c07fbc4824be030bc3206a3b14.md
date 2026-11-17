# Research Pipeline Report - AdaRES_project
快照时间：2026-10-26T18:48:20.453295+00:00；生成报告时快照年龄：300 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-27 02:48:19
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
================================================== 2026-10-27 02:48 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y63_peer_bilinear_res] Best=0.3594 3/30 no_improve=0/30 gpu=0 ? 序列: 0.3290, 0.3594
  [adares_y62_overlap_density_res] 进程未见 (deathwatch 2/3) — 暂不处置

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
  server-1: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y62 | configs_song/model_paper5/adares_y62_overlap_density_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y63 | configs_song/model_paper5/adares_y63_peer_bilinear_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 1；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 116：complete=16 / rejected=79 / closed=12 / blocked=7 / training=1 / admission=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 41（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮核验实际CODEX_THREAD_ID与AdaRES_project前台owner一致，后台Pipeline-AdaRES_project为Disabled。前轮4e61在Y63派发后中断，deadline已过；确认无本地python/scp/helper写进程后使用原token canonical abort，收据INCOMPLETE、context_committed=false、lease_release=released。随后新begin获得本轮快照，不重建或延长旧快照。
- 上海2026-10-27 02:49:57至02:50:02实测Y63在server-1 GPU0、原PID3071037、第3/30轮1171至1179/4686步增长；精确cwd/config、六模型源码、原生YAML确定性路径适配、生产marker与GPU0独占compute PID核验通过。单次GPU69%、22750MiB、68°C、350W。证据pipeline/checks/y63_server1_progress_cf6d85c07fbc4824be030bc3206a3b14.json。Y63为前轮已派发，本轮没有再次启动。
- 对Y62读取完整原生日志和CPU检查点，核实原PID2540376缺席、Train30/30及Val30/30完成。last为epoch30/scheduler30，best为epoch21/scheduler21，两者optimizer状态544、模型及优化器张量有限，文件稳定且保存SHA256。训练best mIoU=0.5082247853279114，冻结全量后评及成本比较尚未执行。证据pipeline/checks/y62_server1_fixed30_cf6d85c07fbc4824be030bc3206a3b14.json。
- 在项目写锁内调用canonical finalize_task，明确gpu=0，将Y62归档为done_complete/30/30并拉取原始日志，检查点留原服务器。仅移除Y62残留队列登记，Y63队列及running映射、Y57暂停与所有外来任务保留。Y62进入冻结后评资源挂起。新收尾工具与既有终态保护共16项CPU测试通过，归档绑定记录在pipeline/checks/y62_server1_archive_binding_cf6d85c07fbc4824be030bc3206a3b14.json。
- 使用canonical gpu_occupation.py采集两服本轮token/snapshot绑定的原始GPU/compute证据：server-1 GPU0独占Y63；server-2 GPU0外来PID13813、100%/24138MiB，未有授权空槽；另次只读核验server-2 GPU0功耗上限280W、85°C。Y56/Y58/Y59/Y60/Y61/Y62后评和Y64原生检查保留等待资源，未接触暂停GPU1或外来进程。

## 5. Decisions Made
- 完成归档与科学有效性分开记录。Y62的0.5082是训练阶段best，不替代冻结全量评估、不改变冠军或宣称方法有效；保持固定30轮、seed1999、原数据、原损失、全量后评协议和全部禁用项。
- Y64已有真实CPU准备；没有空闲授权GPU时，不重复CPU测试冒充原生检查、不挤占Y63或外来任务、不为凑额度新增变体。空槽后核对七文件SHA，完成同原生YAML真实TRAIN至少两步criterion/backward/optimizer及新分支证据，再按既定版本/准入/派发流程推进。
- 一次只读使用旧snapshot helper因其硬编码server-2主机名而失败，不代表server-1故障；保留失败结果，使用新项目内server-1专用核验工具完成观察与终态归档，不修改旧哈希绑定证据。
- 无本轮新增候选、源码提交、生产同步、准入或训练重启。Y57继续用户暂停，server-2 GPU0保持280W，禁止server-6、跨项目和调度更改。已更新七节记忆；接下来按本轮真实结果执行report/finish并核收据及释放状态，当前记录不提前宣称postflight通过。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
