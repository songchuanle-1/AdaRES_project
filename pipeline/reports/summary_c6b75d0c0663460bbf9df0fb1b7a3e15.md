# Research Pipeline Report - AdaRES_project
快照时间：2026-10-29T10:48:09.402159+00:00；生成报告时快照年龄：860 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-29 18:48:08
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
================================================== 2026-10-29 18:48 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y66_peer_state_feedback_res] Best=0.4944 10/30 no_improve=0/30 gpu=0 ? 序列: 0.4497, 0.4648, 0.4699, 0.4826, 0.4944

--- server-2 (192.168.1.163) ---
  [adares_y65_confidence_feedback_res] Best=0.5022 17/30 no_improve=1/30 gpu=0 ? 序列: 0.5010, 0.5005, 0.4999, 0.5022, 0.4995
  [adares_y67_stage_readout_norm_res] Best=0.3215 2/30 no_improve=0/30 gpu=1 ?

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y65 | configs_song/model_paper5/adares_y65_confidence_feedback_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y67 | configs_song/model_paper5/adares_y67_stage_readout_norm_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y66 | configs_song/model_paper5/adares_y66_peer_state_feedback_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 119：complete=16 / rejected=79 / closed=12 / blocked=9 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 45（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 核本会话所属、新GPU1资源授权/科学goal冻结和Disabled后台调度；前6fb1轮收据完整通过/释放。三项原训练观察：Y65 Train17从4490至4499、Y66 Train10从405至412、Y67 Train2从2367至2376，实际源码/YAML/准入/marker/独占与原280W/350W/390W通过。
- 新增独立Y68直接有效语言条件mask-feature调制路径：零初始化W，无额外RNG，+65536参数。保留原共享confidence头、targets/损失/Boolean gate/selector/Q128/seed1999/fixed30和所有通道，不宣称新信息或效益。
- 上海18:53:54至18:54:25实际CPU原生YAML/完整模型/发布SPFormer/真实TRAIN collate162738点8表达通过；float32/64独立有效词公式、零输出/输入梯度/RNG、padding/置换/复制/错误语言和极值控制，以及完整合成DEC七阶段锚点/一次调制七预测消费/有限非零新W梯度/干预与旁路反控通过。未做真实稀疏criterion/optimizer或GPU检查。
- 只提交Y68七文件至9af，保存GPU0/280W和GPU1/390W两个同科学body的原生计划；10项native evidence负对照及goal/source反控通过。三原运行收据/队列/映射未改，无push/生产sync/新准入。
- 三卡canonical物理证据和运行预留显示全部忙；Y68真实原生两步与八项冻结后评等待。Y57单独暂停仍保留；一次只读GPU0 performance显示HW/SW thermal slowdown均Not Active，不据单次采样判原硬件根因已修复。

## 5. Decisions Made
- 优先保持三项原训练；按快照准备1项可区分结构候选，不凑额度或挤占GPU。Y68组件/合成完整DEC的通过不等于真实稀疏任务PASS，也不替代训练效益。
- 新路径改变所有阶段mask特征形成，原query已有语言交互，不复用历史最终confidence回读/泄漏前提、soft gate或norm重连；不进行验证调参、重开旧拒绝或feature subset扫描。
- 空闲资源出现后canonical终态归档/预约清理并核对应GPU绑定计划；真实原生YAML/TRAIN两步通过后冻结原完整评估、准入并当轮派发。保留原全量val/对照/重复/FIRST/scene bootstrap/伤害与成本协议、Y57恢复要求和既有调度频率。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
