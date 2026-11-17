# Research Pipeline Report - AdaRES_project
快照时间：2026-10-20T14:31:11.688236+00:00；生成报告时快照年龄：3563 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-20 22:31:11
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
================================================== 2026-10-20 22:31 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2:
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| 无 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 队列无登记运行；远端进程须另核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 2；准入预算 2。
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 2；总上限 8。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 109：complete=16 / rejected=79 / closed=12 / design=1 / blocked=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 33（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 22:31上海接管AdaRES_project，队列0/容量2；仅server-2 GPU0/1，owner实核、后台Disabled。Y56独立唯一竞争者实现，经同ID两次预注册性能修订，CPU/18边界梯度逐位等价/5结构/10TRAIN40表达all7原生后向全部通过，原高开销与驱动失败保留。
- Y56实际复制初始层confidence差0.000218302、任务方向导数-6.35568e-8，最终层0；整体loss差未越float32分辨率，不是收益。零新增参数，平衡顺序时延median比1.02037、最慢复制scene1.42296；固定大输入backward约0.429秒。
- 用户明确授权Y56七文件本地提交，200486ee12bcf3d7e82393d009d06214dfe83e38；未push/无其他index改动。规范准入后23:13在GPU0/PID83638确认原生step0到5增长、实际源码字节相符。
- Y57保留原4/4控制失败；同ID预注册一次初始完整256二阶矩跨7头连接修订，干预与独立物理四批均过同一原界；CPU/5结构/all7头及anchor梯度/资源复测通过，参数增量66048/stress峰7618953216bytes。不是训练效益证明。
- 用户另行明确授权Y57七文件本地提交，05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea；未push。规范准入后已确认GPU1/PID95947原生step0到3增长，实际部署字节相符，Y56原PID83638保持运行。两卡各一实验、容量2/2、无新增等待，两个准入目标已达。

## 5. Decisions Made
- 目标0.5150、历史CSV冠军Y36 0.5110/差0.0040不变，上轮冻结比较未重跑，Y54大型目标伤害保留。
- 两个直接人类回答各仅覆盖各自七文件，不相互替代。训练只按既有门禁/实时资源核对，保持Y56原PID；不跨项目、不改调度、不使用诊断trainedW2权重正式初始化。
- 保持releasedSPFormer/seed1999/fixed30/Q128/单argmax，以及完整全量双次FIRST/scene聚类/原伤害规则；效益、新颖性及新模型mIoU未知。收尾以真实dispatch和canonical finish为准，不预称校验成功。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
