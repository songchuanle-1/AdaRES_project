# Research Pipeline Report - AdaRES_project
快照时间：2026-10-20T12:26:25.438415+00:00；生成报告时快照年龄：3511 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-20 20:26:24
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
================================================== 2026-10-20 20:26 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y52_point_target_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [adares_y54_fine_support_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [日报] 2 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-20.md

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
- 研究储备 2 / 2；储备准备预算 0，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 0、数值诊断待解 未知、未分类阻塞 2。
- 非终态候选与排队项合计 2；总上限 8。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 109：complete=14 / rejected=79 / closed=12 / training=2 / blocked=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 33（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮20:26规范快照design，Y52/Y54已done_complete且队列0；不重复preflight，不修改调度。
- Y56原8场景224记录无精确复制；修正语言chunk采样单位后新增32不同场景896记录，发现一个场景初始层3复制记录。原失败日志保留。
- Y56实际3案例固定focal0/唯一集/GT复制转移，原头置信度变化.00171-.00444，逆复制参考<=1.20e-7；未证明任务损失/最终分割伤害。
- Y57置信度条件信号在全输入标量对照消失；完整256向量有错位对照分离，但对全输入区间跨0、两个下降场景，仍不确定。
- Y52/Y54/W2/B1四方法各新执行9508表达/141场景完整双次FIRST冻结评估；同一原best断点，全SHA/完整配置值/运行源码绑定通过，原始字段和序列化检查失败均留档。Y36因runner剩余时间保护启动前拒绝，未执行且未创建日志，不虚报第五方法。
- 本轮FIRST mIoU：Y52 0.509708428366；Y54 0.506342952986；W2 0.510107995189；B1 0.502221764356；不选择较好重复。
- 预注册任何子组对任一参考CI上界<0触发数：Y52 0，Y54 0；逐组及完整overall区间均保留，不只看均值。
- 完整配对scene-bootstrap10000/1999、预注册子组伤害和成本已写审查JSON；6项聚焦回归通过。

## 5. Decisions Made
- Y52/Y54固定训练和自身全量评估完成，当前W2/B1比较已做，仍待同冻结Y36双次及配对审查；不由rounded监控或重启分段best替代规范评估，不自动晋升、重跑或调参。
- 只续做原Y56/Y57稳定ID；下一项分别为实际任务损失可达性、完整逐通道二阶矩独立原型与控制/后向/资源。未新增模型、Git提交、准入或训练。
- 完整六段报告0-3保留20:26 preflight事实，4-5记载本轮后续研究；finish前不预称校验成功。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
