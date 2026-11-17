# Research Pipeline Report - AdaRES_project
快照时间：2026-10-23T12:47:21.642307+00:00；生成报告时快照年龄：798 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-23 20:47:21
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
================================================== 2026-10-23 20:47 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y59_frozen_text_mode_res] 进程未见 (deathwatch 1/3) — 暂不处置

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y59 | configs_song/model_paper5/adares_y59_frozen_text_mode_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 3；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 113：complete=16 / rejected=79 / closed=12 / blocked=5 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 37（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮begin预检仍登记Y59 running，是完成后旧登记，不是实际仍训练。随后实查原PID28712及同配置训练均不存在，日志Train30/4686与Val30/363及最后保存公告完整；CPU完整读取并哈希last epoch30/model/optimizer543项/scheduler30与best epoch13/.4956555963。按canonical finalize_task归档Y59=done_complete并拉取完整日志，保留远端检查点/原后评，不伪造deathwatch次数、不重复训练或宣布科学晋升。
- 归档后唯一授权GPU0实际空闲：0%/1MiB/36C，280W保持；GPU1外部任务未干预。资源v2包装器在旧采集器清除retry_after后报KeyError，发生在真实空卡结果已保存之后。原脚本及prepared_resources原始输出/失败记录保留；这是操作记录缺陷，不是Y60模型失败或修订，未阻断后续真实检查。
- 20:50:37至20:51:12上海，在实时独占GPU0与当前租约守卫下，原Y60冻结程序/同commit6ac/原生YAML/真实TRAIN/tools.train两步optimizer实测通过。两loss7.8867197/8.2183971，599个梯度张量全有限，各七头每步10个梯度张量且末层非零、各调用2次；两批4场景/16表达，峰6515851264bytes。只是初始化最小检查，不是完整数据资源上界或效益。
- 仅准入Y60一项，保存experiment_first授权/四门收据和训练后冻结比较计划，当前源/YAML/commit/goal一致；按既有安全同步与门控队列在GPU0启动，PID17186。20:54:25确认命令/cwd/starttime/GPU归属和Train1/30 step20至44；后续v3资源记录step239至340增长。不是只凭PID声称启动。无模型/YAML修改或新Git提交，未使用GPU1/server-6、未改调度。
- Y60启动期GPU0=280W/56至59C，后续74C；短采样未见热降频、新Xid或训练报错，不证明历史硬件问题解决。Y57保持用户暂停及原epoch27/model/optimizer/scheduler检查点，不恢复不迁移。Y56/Y58/Y59后评待独占资源，Y61精确七文件/原CPU payload/decoder goal一致，原生GPU两步仍未执行。
- 新增生命周期v3按实际stage与可选retry识别待办，排除在训已准入Y60，分别处理三项完成后评与Y61版本准备。8项CPU回归通过，真实忙卡资源复查通过；旧v1/v2/错误证据不改。当前唯一运行Y60、无waiting；tick_attempt=Y60/native_smoke/passed，所有引用同时归入候选。

## 5. Decisions Made
- 保护Y60固定30轮/seed1999原训练和280W限制，效益待完成训练后的原冻结全量对照，不按Y59验证轨迹调参或选择新机制。Y59训练完成只是执行完成，非科学晋升；保留best13/last30及原负结果。
- 后续观察使用capture_ef_gpu0_runtime_v2.py --id AdaRES-Y60；资源复查改用recheck_gpu0_lifecycle_v3.py --token CURRENT_TOKEN。不要对已完成Y59使用运行observer，不重放Y59/Y58 finalizer，也不再运行依赖Y60必须待检的旧v2入口。v3为当前已完成/在训/待版本阶段分别留待办，不把空卡或CPU测试当native PASS。
- 唯一授权GPU0被已验证Y60占用，Y56/Y58/Y59冻结全量后评与成本比较继续等待；Y61后续保存精确版本、冻结并运行真实原生两步后才准入。当前不新增候选或更改HEAD6ac，不为额度强行占资源。保持GPU0-only/Y57暂停/数据协议/禁用项/调度不变，无重复常规授权需求。
- 完整六段报告和七标题记忆留档后执行canonical finish，核验receipt/context提交/锁和pending释放；当前尚不预称postflight通过，释放后不补写。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
