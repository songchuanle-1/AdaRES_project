# Research Pipeline Report - AdaRES_project
快照时间：2026-10-17T03:48:38.740674+00:00；生成报告时快照年龄：3339 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-17 11:48:38
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
================================================== 2026-10-17 11:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y50_joint_incidence_res] 进程未见 (deathwatch 1/3) — 暂不处置

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
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 候选累计记录 104：complete=14 / rejected=76 / closed=12 / training=1 / blocked=1。
- 否决范围：unclassified=72 / implementation=3 / hypothesis=1；实现修订累计 3；收据文件 30（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮begin快照2026-10-17 11:48:38上海，模式design；只续同一Y52并处理Y50实际结束，不重复preflight、不改变context/pending哈希及12:48:38不可延长deadline。0-3表为预检事实，以下为之后实测，不将登记running当实际训练。
- 独立原log与检查点证明Y50 Train[30/30]4686/4686、last epoch30正常结束；best epoch21元数据mIoU0.5031588077545166。canonical deathwatch仍1of3，保持原队列/CSV，待后续正常preflight自然核对。
- 已完成Y50/W2/B1全9508表达/141场景各两次，共6完整passes；FIRST主结果依次0.5031587787728745/0.5101051604388791/0.5021216786662852。复用原评估器及scene-bootstrap10000/seed1999、固定TRAIN尺寸切点1487/3325，无验证选择。
- Y50-B1均值+.00103710010658928，场景95%区间[-.00817041093,.01023463737]；Y50-W2均值-.00694638166600463，区间[-.01522865954,.00124739109]。总体均跨0，不宣称稳健收益。中等目标相对W2-.01809714871870814，区间[-.0324303211972209,-.0030292705680468]触发预设上界<0伤害审查；当前实现拒绝且不提升最佳。
- Y52实际8TRAIN场景/32表达、生产batch2稀疏前向/FPS/原生匹配，224层表达记录、1050匹配query；185记录目标差异、最大.1540667266，展开点计数参考误差0。released-init筛选与score梯度均0，保留限制；固定已训练W2诊断态4批共享query梯度均与parent分离，最大.00012113276170566678，repeat噪声0，不等于训练Y52效果。
- 真实AdamW/PolyLR两步及539非空优化器状态，model/optimizer/scheduler和Python/NumPy/Torch/CUDA RNG全部精确恢复，峰值reserved7832862720bytes。原生推理部署batch1，压力仍用原4个batch2中最大379425点批次，有限反向且reserved<24GiB，新增参数0；不是全数据最坏情况或Pareto收益。
- 保留raw源换行绑定、旧非持久BERT buffer、JSON scalar及父模型batch2 predict失败；新预注册兼容程序仅修诊断绑定/加载/输出/batch，原模型七文件hash与所有旧失败不改，模型修订0。12:39:47至12:40:41固定W2权重推理32表达二值mask/argmax/IoU完全一致，概率最大差约2.325e-6，repeat也有微小差异，不声称概率逐位一致。
- 本轮没有新候选、模型修改、Git提交、准入或训练派发；实际产出为原生捕获/反向/完整恢复、两组资源与推理控制、6完整评估及冻结统计。Y50证据54/Y52证据113是累计条目，不是本轮实验数或成功率。收据用量窗口为begin到receipt，当前计数待收据/未知，缓存已含input、reasoning已含output，不重复相加也不当账单；六段留档后按契约finish，未成功前执行状态为待验证。

## 5. Decisions Made
- Y50本次固定30实现因预设中目标伤害规则拒绝，failure_scope=implementation；总体区间跨0，不泛化为所有高阶共同覆盖假设被推翻，不自动重跑、调阈值或调轮数。历史CSV最佳Y36 .5110，目标.515差.0040不变；不把此次W2/B1重测与历史Y36混成同轮比较。
- Y52同一冻结原型已补齐原生有限范围证据，真实下一步为仅七个已测文件的限定提交许可。原件明确用户授权仅Y50，不能自授或扩展到Y52；当前blocked/human_authorization而非GPU不足，不把尚未提交/准入写成已训练。待明确许可或精确commit后重验当前goal/六门禁和真实commit回执，再走canonical dispatcher，不部署诊断权重。
- 外来GPU0进程保持不动，GPU1仅在真实空闲与独占租约下做诊断；不碰server6、其他项目、调度或后台，不新建任务、不嵌套codex exec。原型之外脏文件不动。
- 更新七段记忆与真实design-cycle外部授权依赖，尝试正常finish；不复用旧GPU阻塞checker错误，不改门槛迎合校验。若失败保留确切失败并规范abort、独立核实释放，收尾不成功绝不称本轮完成；释放后不写项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
