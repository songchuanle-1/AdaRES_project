# Research Pipeline Report - AdaRES_project
快照时间：2026-10-22T02:48:36.905534+00:00；生成报告时快照年龄：898 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-22 10:48:36
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
================================================== 2026-10-22 10:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y58_fixed_backbone_statistics_res] Best=0.4993 14/30 no_improve=0/30 gpu=0 ? 序列: 0.4837, 0.4881, 0.4912, 0.4878, 0.4993

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1 waiting=1
    待推: [AdaRES-Y59] configs_song/model_paper5/adares_y59_frozen_text_mode_res.yaml gpu=0
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y58 | configs_song/model_paper5/adares_y58_fixed_backbone_statistics_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 1 | AdaRES-Y59 | configs_song/model_paper5/adares_y59_frozen_text_mode_res.yaml | waiting | - | server-2 |
- 活跃 2（运行 1 / 认领 0 / 等待 1）；目标 3；准入预算 1。
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 1；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 112：complete=16 / rejected=79 / closed=12 / blocked=3 / training=1 / deployment=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 37（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮上海10:48:36获取实验优先design快照，实际会话归属匹配，后台Disabled；容量1、运行1、就绪排队1，新增预算1。只用server-2 GPU0，Y57用户暂停及GPU0临时280W保留。
- 10:50:19实核Y58原PID77094/命令/cwd/starttime/GPU0归属，Train14/30 step3733至3757递增、尾部无报错。GPU0=280W/85至86C，本次热降频Active；内核无新Xid/fallen-off。GPU1仍有外部负载，不干预，不宣称散热根因已修复。证据pipeline/checks/ef_gpu0_runtime_3c92a71dd21447688bbadff8c0c10db5.json。
- 执行Y60到期资源复查，精确源码/YAML/冻结payload/commit/goal匹配；唯一获准GPU0仍忙，Y60没有执行native GPU smoke。Y58当次Train14 step3718至3797递增，前13轮日志best mIoU=0.4992775022983551，只是中途指标。资源证据pipeline/checks/prepared_resources_3c92a71dd21447688bbadff8c0c10db5.json；额外实核Y58/Y59各7文件和现有准入收据均通过，Y59保持waiting/gpu0。
- 新增独立Y61 peer-value gradient对照及唯一原生YAML，仅竞争peer values使用相同可训练LayerNorm(query.detach())；focal输入/掩码重叠梯度/共享头/全部损失/selector/seed1999/fixed30不变，参数增量0。与旧Y20掩码分支截断和Y60阶段分头不同，不重开旧拒绝，不宣称训练收益。
- 10:56:59 Y61真实CPU测试通过：原生YAML完整模型/发布预训练/真实TRAIN两场景162738点及8表达；float32和float64中前向、focal查询梯度及mask梯度逐位等W2，仅peer值输入梯度从0.00060267/0.00064859变0，peer mask梯度仍非零；仿射/头参数梯度、排列、重复及padding控制通过。证据pipeline/analyses/y61_cpu_3c92a71dd21447688bbadff8c0c10db5.json及.log。该测试不含完整稀疏骨干/任务criterion/optimizer两步，不是native smoke PASS或效益。
- Y61精确7文件/不可变payload/CPU结果/资源已留档，候选按实际GPU阻塞保留；没有Git提交、生产同步、准入、入队、启动或新完成项。当前tick_attempt为Y61 implementation_test/passed，仅CPU范围。已更新七段记忆和next_batch决策，尚未预称本轮收尾校验成功。

## 5. Decisions Made
- 保护Y58固定30轮，GPU0释放后由既有门控按FIFO派发Y59；不共卡、不抢占其他任务、不恢复Y57或扩大GPU授权。热降频如实报告，继续保持已授权280W，不改风扇、功耗或调度。
- 派发器要求已准入任务code_commit等于当前HEAD；因此Y61目前保存精确测试包但不移动HEAD，避免使Y59失配。先保护现有Y59派发及已提交Y60，再保存Y61测试版本、绑定并执行原生YAML真实TRAIN两步检查；只有实际通过后才能准入/入队/尝试启动。资源等待不是缺用户授权，不重复索取常规许可。决策pipeline/decisions/20261022-110104_next_batch.md。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
