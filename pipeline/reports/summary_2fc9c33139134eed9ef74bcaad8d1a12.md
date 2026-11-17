# Research Pipeline Report - AdaRES_project
快照时间：2026-10-02T15:47:49.785728+00:00；生成报告时快照年龄：514 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-02 23:47:49
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
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-02 23:47 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y18_point_measure_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [日报] 1 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-02.md

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| 无 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 快照内无运行任务 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 1；准入预算 1。

## 4. This Tick's Actions
- 本轮23:47:49+08 design快照：Y18已由正常monitor在deathwatch3/3确认done_complete；到期候选Y23的恢复依赖解除。未重复preflight，也未重跑已完成的Y18科学评估。
- 按用户授权恢复同一Y23：实时独占GPU1空闲、原PID消失、源码和原epoch4检查点哈希均通过守卫。保留原恢复脚本及其哈希，用GPU1-only适配限定计算进程查询；GPU0未查询、未使用。
- 23:49恢复验证通过：server-2 GPU1，新PID129584，原生--resume原last.pth，model/AdamW/PolyLR从epoch4恢复至epoch5，固定总30轮和seed1999不变。原epoch5未保存的256步会重放，不声称全加载器/RNG或逐位轨迹一致。
- 23:51:12至23:51:17两次原生采样：Train5/30 steps167->177/4686，实际命令/config/GPU1/--resume一致；限定24KB尾部未见Traceback/OOM。GPU1显存16444MiB、利用率89%是设备占用，不是候选隔离峰值。
- 保存恢复守卫和递增步数JSON及原始日志；SHA256分别6f6dc132fdc8ff585d671801b54ce6ddb96991747c7e3f03617cc6cc6ea450c5、2c5741f7d2fe192ee4746e7b13fe7bfcf54a006c2e5dcc840b70df60b4ff307f。Y23由blocked转training，更新七节记忆并保留所有旧失败和暂停证据。
- 科学状态：Y23 training、最终疗效未知；Y18已完成固定30轮和冻结评估，mIoU约0.507606、目标未达、相对W2/B1场景聚类区间均跨零。本轮执行结果须以finish和锁释放检查为准；收据用量窗口及产出计数生成前未知，生成后在可见报告补充，不把缓存重复相加或视为账单。

## 5. Decisions Made
- 解除已满足的Y23用户恢复依赖；恢复原候选而非从头训练或重新准入，不更改源码、门槛、冻结设计和历史证据。
- 恢复后活跃1/目标1，停止新增候选；继续同一PID固定30轮，到终态执行exp_design_Y23_d20c71e4.md的冻结全量评估。不为填槽产生变体，不重复启动。
- GPU0继续保留给项目外用途，未经用户明确归还不查询或使用；不操作其他项目、后台或调度。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
