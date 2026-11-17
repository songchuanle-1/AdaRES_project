# Research Pipeline Report - AdaRES_project
快照时间：2026-11-01T08:52:26.423726+00:00；生成报告时快照年龄：1875 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-01 16:52:25
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
================================================== 2026-11-01 16:52 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y72_visual_mask_refinement_res] Best=0.5004 28/30 no_improve=13/30 gpu=0 ? 序列: 0.4912, 0.4938, 0.4938, 0.4902, 0.4878

--- server-2 (192.168.1.163) ---
  [adares_y74_language_update_gate_res] Best=0.4985 13/30 no_improve=1/30 gpu=0 ? 序列: 0.4821, 0.4893, 0.4898, 0.4985, 0.4898
  [adares_y75_language_query_init_res] Best=0.3771 3/30 no_improve=0/30 gpu=1 ? 序列: 0.3000, 0.3771

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
| AdaRES-Y72 | configs_song/model_paper5/adares_y72_visual_mask_refinement_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 128：complete=16 / rejected=79 / closed=12 / blocked=18 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 53（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 核实际owner/foreground与Disabled旧后台，先查三原运行和waiting0。Y72/Y74/Y75原PID/source/YAML/receipt/功耗及独占核验，native计数持续增长；快照Y73终态只是前轮已归档反映，未再次归档、加载checkpoint或重放finalizer。
- 新增Y77：原初始预测前加入原采样query相对全部原有效点数加权scene质心的位置投影，零W[3,256]/无RNG，新增768参数；保留全部原数据/通道/已有2D/采样/SCG/硬门控/7共享头/targets/losses/selector/seed1999/fixed30/native YAML，不宣称缺失信息、创新或效益。
- 实际CPU原生完整模型/发布backbone/TRAIN collate162738点8表达与float32/64独立逐标量质心投影/平移置换/zero/RNG/原输入梯度/七阶段W2锚点、原query/geometry/mass/SCG与真实初始prediction消费、新梯度/bypass控制通过；仅七测试A文件在隔离index保留，主HEAD9bf/index/queue/running未改。
- 原生双卡GPU0/280W和GPU1/390W检查方案、11证据拒绝守卫、Py3.8 AST/current goal/七源码YAML/retained commit与stale-goal/tampered-source拒绝通过；冻结native回调在CPU两合成decoder optimizer步骤实际通过，含每步12个原SCG位置消费、初始prediction和7原共享头/有限新梯度及真实更新。未执行native sparse GPU任务或准入。
- Y76原三native方案/CPU callback/server1预检、Y72原完整终态helper依赖及receipt/freeze仍有效。实际两台canonical occupation确认三授权卡busy且预约，Y76/Y77真实GPU任务和16项原完整后评/cost等待独占资源；保存完整交接、旧记忆字节备份和七段记忆。

## 5. Decisions Made
- 本轮新增候选1、准入0、派发0、归档0，在快照budget内；本轮真实implementation_test passed仅表示CPU实现与原生观察程序可执行。无空闲授权GPU，不能将CPU/静态/合成证据作为native GPU PASS直接入队。
- Y76先于Y77；下一较近Y72须真实自然fixed30退出、原完整证据核验和canonical归档，再fresh server1GPU0 idle/no compute/no reservation350W后才Y76现成方案activate/run。其CPU已过不重做；actual native PASS后当轮原完整freeze/准入/新明确source契约/安全派发。
- 保留科学goal字节、原数据/验证边界/全部禁用项/seed1999/fixed30/原完整后评协议和现有运行；一授权卡一实验及350/280/390W不变，Y57仅明确单独续训请求可恢复，禁server6/跨项目/改调度。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
