# Research Pipeline Report - AdaRES_project
快照时间：2026-10-16T07:18:07.472330+00:00；生成报告时快照年龄：3326 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-16 15:18:07
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
================================================== 2026-10-16 15:18 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y50_joint_incidence_res] 1/30 尚无验证 - gpu=1

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y50 | configs_song/model_paper5/adares_y50_joint_incidence_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 候选累计记录 103：complete=14 / rejected=76 / closed=12 / training=1。
- 否决范围：unclassified=72 / implementation=3 / hypothesis=1；实现修订累计 3；收据文件 30（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 科学状态：Y50 training；Y52完成CPU准备，真实blocked/exclusive_gpu_slot。执行状态：INCOMPLETE；finish返回1并保留租约，尚待abort释放。
- Y52先登记diagnosis再做源审计与35CPU控制；实际indicator是匹配CE，不是质量回归；仅辅助score存在等超点与点级IoU口径差异。
- 冻结8TRAIN场景/32表达/每表达128掩码重放，47/4096现有0.5筛选边界不同；匹配后频率未知，非性能提升。独立七文件未提交原型/新增参数0，32精确计数与5表达映射、完整Criterion最终及辅助层组件对照已执行。
- 保留v1逐位梯度断言失败；v2最大2.27e-13与原样重复一致，辅助score梯度精确；其余损失不变。不能替代全模型原生反向、资源及恢复门禁。
- 客户端16:06:33.71至16:06:34.23+08复核：GPU0外来PID19723/100%/22322MiB，GPU1本项目Y50 PID31539/45%/22694MiB。Y50 Train[2/30]3615→3616/4686；早期验证mIoU约.312、Acc@.25约.408/Acc@.5约.335、spmIoU约.284，精确未舍入值未知。
- 本轮新增研究候选1、独立本地原型1、准入0、启动0；Y50提交授权已解决不再索要。收据用量窗口及计数须finish后读取；当前未知，不将缓存重复相加或将用量视为账单。
- 真实收尾故障：CLOSURE_INCOMPLETE: Blocked GPU candidate has a queue opening。检查器只按本项目active=1/容量2认定有空位，未反映GPU0外来PID19723的实时占用。保存检查器源码哈希、具体条件及实时占用证据，不改门槛、不造排队项，不重试相同finish。

## 5. Decisions Made
- 真实下一步是同一Y52的原生匹配/反向与资源测试；CPU准备完成后两张授权卡均被占用，保留明确retry/unblock，不造假填卡或重开旧路线。
- GPU0外来占用不算本项目第二个实验；不动现有Y50、外来进程、目标、门禁、调度或其他项目。Y52提交许可未获，当前尚未达到需提交阶段。
- 上一轮7548已INCOMPLETE并abort释放；本轮按真实blocked候选验证收尾，finish成功才称完成，否则保留证据并abort核实释放。下一调度时间未知。
- 调用规范abort，仅中止本轮tick并核实释放；保留Y52证据和原有Y50训练，不将本轮称为完成。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
