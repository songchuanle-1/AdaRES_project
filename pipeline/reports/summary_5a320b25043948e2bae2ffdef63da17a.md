# Research Pipeline Report - AdaRES_project
快照时间：2026-11-14T05:48:18.264132+00:00；生成报告时快照年龄：1007 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：4
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-14 13:48:17
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
================================================== 2026-11-14 13:48 : AdaRES_project ==================================================

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
- 研究储备 0 / 0；储备准备预算 0，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 155：complete=19 / rejected=81 / closed=12 / blocked=42 / training=1。
- 否决范围：unclassified=72 / implementation=6 / hypothesis=3；实现修订累计 7；收据文件 79（含修订或重试，不代表成功率）。
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
- 完成：原server1GPU0实际释放后，核验旧拒绝恢复目录/收据不存在、旧成本进程已退出，使用新不可变标签启动原科学核心与工作程序正文均字节不变的Y90成本恢复；24块测量、原源码/检查点后审计通过，107次资源采样均无外部GPU0进程，原首轮受干扰成本仍不接受。
- 完成：Y90正式FIRST0.5038137204；原小目标相对W2差值-0.0231434791、95%CI[-0.0399042583,-0.0066074527]触发原冻结伤害判据。全八遍后评/原配对与噪声分析/有效成本/后审计齐备，归档rejected_material_subgroup_harm，仅拒绝该fixed30/best14/seed1999实现，不推广到整个机制。
- 运行：复用Y88原26文件/YAML/私有环境和11边界准备，仅替换当前任务标签/目录，启动首个原server1GPU0/350W四方法各两遍冻结后评。工作PID3891354，子PID3891626，原生进度实际增加至{'pass': 0, 'chunks': 240, 'total_chunks': 363}；结果仍待完整后评和成本。
- 运行核验：Y102原PID4747/server2GPU0/280W仍Train1/30step7；原进程、6源码、原YAML/环境及现场保留，不重启、不猜检查点目录、不重复被拒绝的栈附加。

## 5. Decisions Made
- 本轮没有新增候选/准入/训练/原生smoke，保持goal_rc4和零准备/训练额度，不把训练best当正式FIRST。十三到期项中完成Y90、启动Y88，其他十一项按本轮真实GPU0占用等待。Y103原生仍未执行/无准入/阶段门保持；Y100后评保持原server2GPU0，禁止迁移。
- 首次冗余占用采集因已无阻塞候选而在远端创建前失败，保留实测错误并改用新鲜只读空闲/350W/旧目录不存在核验，后续才创建新意图与进程；没有审批拒绝、科学重试、证据覆盖或保护放宽。
- 原goal/收据/数据/源码/检查点/权限/功耗/调度不变；GPU1六项后评和Y57单独用户暂停、Y86/Y97原数值负证据与专用恢复要求均保留。不跨项目、不用server6、不干预其他任务、不自动重开Y90/Y92/Y94。后续只观察现有Y88及Y102，Y88完整科学结论仍待测。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
