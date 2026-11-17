# Research Pipeline Report - AdaRES_project
快照时间：2026-10-19T07:50:18.995065+00:00；生成报告时快照年龄：3633 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 15:50:18
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
================================================== 2026-10-19 15:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y52_point_target_res] 进程未见 (deathwatch 2/3) — 暂不处置

  >>> 队列推送: [AdaRES-Y54 fixed30] on server-2 gpu0
  >>> 已推送，队列已更新
  [日报] 1 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-19.md

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y52 | configs_song/model_paper5/adares_y52_point_target_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y54 | configs_song/model_paper5/adares_y54_fine_support_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 3、未分类阻塞 0。
- 非终态候选与排队项合计 3；总上限 8。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 108：complete=14 / rejected=79 / closed=12 / blocked=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 33（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮用户明确要求“挂载了 提交训练吧”，原/bak_home=/dev/sda1和source/data/work/marker恢复；仅继续原已准入Y52/Y54，不新增候选、准入或修改固定30轮。0-3保留上海15:50:18预检快照，以下为后验实测；旧后台Disabled，无跨项目/调度/额外monitor或goal预检。
- Y54由begin规范启动GPU0/PID47332；Y52原PID15934消失、旧log止于第1/30轮535/4686且所有原运行目录无.pth。保留日志SHA、原task/experiment/deathwatch和attempt_history，按直接用户请求通过现有派发器在真正空闲GPU1新开同一fixed30 attempt，不伪称断点续训或optimizer恢复。
- Y52派发先后被源码观察器兼容/格式问题拦截，保留失败证据后修复Python3.8路径API、完整YAML结构及Python CRLF/LF等价；实际代码/字段/二进制/缺失/越界仍拒绝，未改远端源码或训练参数。最终源码正反16项、闭环/前台/上下文109项共125项回归全过；中间4项生成转义失败修复重测，失败摘要保留。
- 上海16:45:02至16:45:08原PID双采样：Y52 GPU1/PID69064 Train[1/30]348至356/4686；Y54 GPU0/PID47332 Train[2/30]1646至1656/4686。cwd/config匹配、bounded tail无异常，两个launch_verified=true，waiting0。完整证据pipeline/analyses/mounted_training_verified_097ff69b175b49c59fbcf414fb0bd4f5.json。
- Y56旧数据盘阻塞已归档，回到diagnosis待续实际TRAIN重复度/trained-parent控制；不把CPU随机头合成效应当收益或准入。当前训练best未在最后核验抽取，标未知；尚无固定30轮终态全量比较。实际新增候选0/模型修订0/准入0，本轮完成两项原生训练确认、原中断保留及125项工程回归，不是科学成功率。
- 用量窗口仅本轮owner.started_at至receipt本地可见记录，具体值待收据/未知，排除早读、晚到记录和最终回复；缓存含在input、推理含在output，不重复相加、不把用量当账单。此报告尚待finish，不预写PASS。

## 5. Decisions Made
- 按用户本次明确请求提交既有实验：Y54原PID保持，Y52原无断点partial attempt保留后新开fixed30，不冒充续训；两卡均有实测原生增长。
- 所有原科学门禁、训练/评估冻结和目标资源不变；源码兼容及格式等价修复有125项测试和真实完整性派发验证，不操作其他项目或调度。下一步继续同一PID训练至固定协议终点后按冻结规则评估。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
