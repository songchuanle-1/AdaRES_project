# Research Pipeline Report - AdaRES_project
快照时间：2026-11-13T15:46:25.493759+00:00；生成报告时快照年龄：2085 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：4
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-13 23:46:24
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✓] Champion: AdaRES-Y100 — mIoU=0.5164 (target met, exceeded by +0.0014)
       Epochs: ?, State: done_complete
  Phase 1: PHASE1_REACHED ✓

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0144

## Phase 2: Story Verification
  (champion 已达标，检查论文故事是否闭环)

  [✓] Motivation (为什么需要这个方法): covered
  [✓] Method (我们的方法是什么): covered
  [?] Gap (champion vs baseline 提升): covered (γ=+0.0144)
  [✗] Ablation (核心设计是否必要): missing
  [✗] Stability (多seed稳定性): missing
  [✗] Generalization (跨setting泛化): missing
  [✗] Limitation (失败案例分析): missing

  Story: INCOMPLETE ✗ (2/7 elements covered)
  Phase 2: 需要补充消融/seed实验

  Suggested next steps:
    - 设计消融实验验证 champion 核心设计选择
    - 跑 champion 的多 seed 稳定性验证

## Overall
  PHASE1_REACHED — champion 达标，Phase 2 进行中
================================================================================
```

## 1. GPU Overview
- 仅列授权卡；未采集利用率、显存或进程时标为未知，不推断空闲。
| Server | GPU | Util | Memory | Process |
|---|---|---|---|---|
| server-1 | 0 | 未知 | 未知 | 未知 |
| server-2 | 0 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-11-13 23:46 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  无训练进程

--- server-2 (192.168.1.163) ---
  [adares_y102_geometry_mask_skip_res] 1/30 尚无验证 - gpu=0

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1:
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y102 | configs_song/model_paper5/adares_y102_geometry_mask_skip_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 4；准入预算 0。
- 研究储备 1 / 0；储备准备预算 0，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 1；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 155：complete=16 / rejected=79 / closed=12 / blocked=46 / validation=1 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 7；收据文件 79（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |
| AdaRES-Y67 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.676817+00:00 |
| AdaRES-Y70 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.677616+00:00 |
| AdaRES-Y73 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.678374+00:00 |
| AdaRES-Y75 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679091+00:00 |
| AdaRES-Y78 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679826+00:00 |
| AdaRES-Y81 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.680927+00:00 |

## 4. This Tick's Actions
- 完成：Y101原固定四组八遍全9508/141结果、原修正重复噪声和配对10000/1999分析、原TRAIN8+largest-pair串行/原生内存压力成本及源码/检查点后审计均已核验；原训练/预测/终态归档没有重跑。正式FIRST mIoU0.5062989703，低于同设备Y36 0.5107673921和W2 0.5100091053，未达0.515。
- 科学处置：整体配对区间均跨零，预设尺寸子组未触发伤害判据；记录complete_inconclusive，不宣称增益、等效、机制被证伪或晋升。参数+65536；冻结串行/压力成本完成，原权重和源码字节不变。
- 运行：释放原GPU0后，已完成归档的Y99按自己原冻结server-1 GPU0/350W协议启动首次四组全量后评并实测计数递增。原模型commit与当前加法部署marker分开绑定，26文件和原配置/私有环境+11项CPU身份边界检查通过；未新建训练、入队准入或迁移GPU1任务。
- Y102仍为原PID4747/server-2 GPU0/280W，{'phase': 'Train', 'epoch': 1, 'total': 30, 'step': 7, 'count': 4686}，current_native_progress_confirmed=False；保留原现场和全部失败，不重启或重复索取管理员许可。收集器一次摘要函数导入错误发生在远端操作前，已更正为原冻结函数并保留负记录。
- 收尾兼容：Y62/Y63/Y64旧资源挂起记录没有required_gpu字段，初次审计因此KeyError。原失败和原脚本保留；仅从各自原签名冻结资源中读取GPU0/350W，并再次要求当前白名单[0]和相同候选/服务器身份。未改旧记录、收据、资源权限或官方检查器，刷新使用新不可变标签。

## 5. Decisions Made
- 保持本轮零新实验/准入/训练预算；只推进原已授权、已训练归档且原设备匹配的冻结后评。Y101完整结果留档不自动替换原基线，Y99后续必须完成原四组全量比较、原成本/后审计再科学处置。
- 25项到期任务：Y101完成、Y99实际运行；其余原GPU0完整后评/原生检查等待当前物理占用，原GPU1六项仍用户暂停，Y86原数值失败不重跑。Y57单独暂停；不跨项目、不用server6、不改goal/原收据/数据/功耗/调度。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
