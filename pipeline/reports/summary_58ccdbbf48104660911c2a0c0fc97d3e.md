# Research Pipeline Report - AdaRES_project
快照时间：2026-10-01T23:48:36.785511+00:00；生成报告时快照年龄：2035 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-02 07:48:36
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-W2 — mIoU=0.5100 (gap to target: +0.0050)
       Epochs: 30/30, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0080

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5100, target=0.5150, gap=+0.0050
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
================================================== 2026-10-02 07:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y18_point_measure_res] Best=0.4980 12/30 no_improve=0/30 gpu=1 ? 序列: 0.4670, 0.4882, 0.4874, 0.4895, 0.4980
  [adares_y15_query_completion_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [日报] 1 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-02.md

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y18 | configs_song/model_paper5/adares_y18_point_measure_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。

## 4. This Tick's Actions
- 本轮 design，快照北京时间2026-10-02 07:48:36；monitor/goal只运行一次，Y15经deathwatch3/3确认done_complete，保留其科学拒绝及全部历史失败。
- 实际新增并测试Y19/Y20/Y21三个稳定诊断候选；Y19执行AST提取的G1原生loss与独立点展开参考，等质量loss/gradient差0/0，不等质量仅改变标定，拒绝重命名旧目标。
- Y20执行train64独立扫描/256表达原生训练梯度及重复/断梯度对照：余弦均值+0.0277698，43正/21负，重复无符号变化，拒绝全局对抗梯度假设。另执行train8缓存相同indicator输入审计，logits差0，断路梯度归零且head仍有梯度；原始端到端logits差8.66407保留，原因未知。
- Y21执行train64/1938表达（916含冻结方向词）的原生重复/180度旋转/平移对照，身份检查一致；方向词相对其他表达的旋转退化差-0.00962754，24正/40负，重复IoU差0/选择翻转0，拒绝方向特异损伤假设。平移敏感性及相机相对语义仍未知。
- 08:11:12至08:11:15原Y18 PID7449/唯一日志验证Train12步骤3214→3217/4686；08:15:37 GPU0 1MiB/0%、GPU1 22720MiB/75%，GPU0诊断flock可获得并释放。未新增训练、准入、独立生产包或配置，未重复完整验证集评测。
- 科学状态：Y19/Y20/Y21 rejected、Y18 training、目标未达；持久化当前早停周期、七段memory与哈希证据，旧周期原样保留。本报告生成时postflight尚待finish，不提前称完成；收据用量和最终执行状态将在可见报告补充。

## 5. Decisions Made
- 三条实际诊断路线均有本轮反证；同类标签标定、detach及yaw/方向词子集重复探针不再有支持的信息增益，按no_information_gain短期冷却至2026-10-02 10:30+08，不宣称所有独立路线穷尽，也不伪造资源阻塞。
- 新原生机制证据、Y18终态/证据、目标或队列变化可提前唤醒；到期应检验真正独立且预注册的假设，不重命名已拒绝路线、不调整验证集阈值/权重/温度/epoch。
- 保留Y18原固定30epoch训练及冻结终态评估方案；只使用server-2授权卡，未操作其他项目或调度。下次自动化准确时间未读取，未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
