# Research Pipeline Report - AdaRES_project
快照时间：2026-10-14T05:53:03.712784+00:00；生成报告时快照年龄：3673 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-14 13:53:03
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
================================================== 2026-10-14 13:53 : AdaRES_project ==================================================

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
| 无 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 快照内无运行任务 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 2；准入预算 2。

## 4. This Tick's Actions
- 快照2026-10-14 13:53:03北京时间/design；本轮延续同一租约，14:48新触发未重复begin/preflight。报告写入时尚未finish，执行状态待校验，不能预称完成。
- 关机中断的原Y43、Y44均已保留模型/优化器/调度器状态续训到30轮；Y43按冻结小目标损害规则拒绝，Y44完整评估不确定，不重开或晋升。
- Y45原生梯度归因TRAIN8中7/8旧指标与分类似然梯度相反，released-init4中0/4；缓存重复一致，解释局部目标冲突而非总体效益。六项可执行性门槛审查PASS。
- Y45精确七文件提交338a3d7a5b21fa463d116e566b482f0733fde9c2及哈希绑定准入；14:32:18-24核验PID78001/cwd/config/GPU1/源文件一致，原生第1/30轮270→280/4686步；科学状态training，尚无效益或Pareto结论。
- Y46完成原生TRAIN8/32表达/128查询捕获：1242预测多分量、29/32真目标多分量。CPU32实际Y36及CPU32实际W2/Y31/Y36对照在等局部/归一化形状输出下有不同全局组件，不宣称普遍盲区或新颖性。
- Y46全组件CPU原型完整测试32条捕获表达：8个有限非零梯度，缓存重复一致、重标号误差1.94e-16；原型平方损失梯度不冒充原生训练反传。CPU302.85秒不是配对部署延迟。
- Y46 CPUv1结果保存误读<stdin>失败，原源码/日志保留；v2只修附带源码字段后完整重跑，没有改模型、测试或容差。
- 14:50:44实时GPU0外部PID19723/23946MiB/91%，GPU1Y45PID78001/17444MiB/47%，均保留；Y46下一项原生反传/资源测试真实受阻，已冻结证据及下一步，未准入或生成生产包。
- 收据用量及产出finish后只在任务中补充，当前未知。用量为begin至收据的局部日志窗口，不含初始读取、最终回复或晚到记录；缓存属于输入、推理属于输出，不重复相加、不当账单。

## 5. Decisions Made
- 20261014-142914_next_batch：准入同一Y45，releasedSPFormer/seed1999/固定30轮/单GPU及原冻结评估不变；准入不等于有效，未提交无关改动或推送origin。
- 同一Y46保留全部前景/背景分量，不过滤最大分量或假定目标必须连通。原生损失活动及成本决定是否构建生产独立包，当前待测。
- external_resource_blocked基于真实占用，不声称路线穷尽；Y46于2026-10-14 15:50:44+08或授权卡真实空闲时续做冻结原生测试，不重复已无新信息的缓存对照、不占外部任务。
- 下次自动化准确触发时间未知，以保存调度为准；不改调度、不新建任务、不恢复后台研究。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
