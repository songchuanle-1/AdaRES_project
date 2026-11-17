# Research Pipeline Report - AdaRES_project
快照时间：2026-11-08T00:50:08.441338+00:00；生成报告时快照年龄：1416 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-08 08:50:07
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
| server-1 | 0 | 未知 | 未知 | 未知 |
| server-2 | 0 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-11-08 08:50 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y90_query_peer_affinity_res] Best=0.5038 30/30 no_improve=15/30 gpu=0 ? 序列: 0.5007, 0.5017, 0.4978, 0.5014, 0.4966

--- server-2 (192.168.1.163) ---
  [adares_y91_hidden_confidence_gate_res] Best=0.5021 14/30 no_improve=0/30 gpu=0 ? 序列: 0.4886, 0.4942, 0.4956, 0.4953, 0.5021

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y91 | configs_song/model_paper5/adares_y91_hidden_confidence_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y90 | configs_song/model_paper5/adares_y90_query_peer_affinity_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 145：complete=16 / rejected=79 / closed=12 / blocked=36 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 68（含修订或重试，不代表成功率）。
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
- 仅处理 AdaRES_project；实际会话归属、后台 Pipeline Disabled、当前资源及实验优先授权已核验。begin 已查进展并尝试派发合格队列，快照 ready waiting 为0；不跨项目、不改变调度或科学目标字节。
- Y90 原PID2442962及同配置训练进程已自然退出。使用原25项生命周期守卫及已绑定来源执行完整只读终态核验：原20261107_071428、Train/Val1至30、稳定有限last30/best14、model/optimizer544/scheduler30与14及source/YAML/receipt/freeze全部一致，显式server1 GPU0归档完成。原checkpoint留在远端，训练best 0.5038180947303772不是正式FIRST，不重启、不重放归档。
- Y92 在新鲜独占空闲server1 GPU0/既有350W上执行保留的真实原生YAML/TRAIN两步检查。loss为4.615511417388916和6.318481922149658，原及新参数、输入梯度/更新有限，新增视觉特征投影及原14个mask/score/head消费者与12个原粗门路径实测通过，peak reserved为6717177856字节；两批不是效果或全量显存上界。
- 选择Y92前，为仍在跑的Y91新建显式精确终态绑定，29项原生命周期及8项真实附加来源/原生证据/runtime守卫首次通过，无跳过；原PID83827、目录20261107_102124、receipt/freeze保持，未加载或停止Y91。旧签名helper保持，当前未来终态入口为finalize_y91_fixed30_y92.py。
- Y92 的22项实际原生准入守卫通过，无跳过；完整原Y90训练和后评协议仅方法ID替换后冻结，生成收据并当轮入队。仅选择既有已测试七新增文件，保留无关index/worktree；仅server1安全同步和canonical派发，实际PID1305468与递增原生步数已核验，不把claiming或PID单独当启动成功。
- Y93 新建同字节七文件/相同原生程序主体的附加提交及当前Y92/server1、Y91/server2运行绑定。16项实际来源、payload、runtime、负例、当前占用及真实父原生/receipt/freeze守卫首次通过，无跳过；仅commit元数据变化，生产HEAD/index/queue和CPU科学证据保持。Y93仍GPU未执行、未准入，不能写native PASS或效果。
- 收尾真实观测：Y91 Train14/30、3260/4686，Y92 Train1/30、779/4686；原生来源/YAML/receipt/runtime/功率及递增计数已核验。两台canonical物理占用均显示授权GPU0忙且由对应原PID保留，外部GPU1/PID3701未触碰；按原300秒证据时效收尾，真实resource_check/tick_attempt保持。
- 本轮新增科学候选0、CPU科学重放0，真实GPU原生检查1、准入入队1、训练启动1、完整精确终态归档1。Y93初次生成工具在执行前发现元数据命名及旧父提交文字问题，原v1保留未执行，仅新增元数据修正版v2，原科学/原生主体和守卫不改。一次已存在progress标签被不可变性守卫在查询前拒绝，改用新postlaunch/final标签；初次只读SSH行内引用失败后改stdin查询成功，未改变训练。完整失败轨迹及原证据保持。
- 七节记忆已原样备份后更新；所有历史负结果、专用恢复、科学goal和原运行收据保持。未新增全量正式FIRST、成本、科学收益或晋升结论；本报告不预先宣称尚未执行的官方finish结果。

## 5. Decisions Made
- 继续保留Y91和Y92原独立训练，每卡一个实验，不抢占、不迁移、不重启。Y90原完整冻结server1 GPU0/350W后评继续等合法空槽，不用训练best替代正式FIRST。
- Y93只在HEADY92、对应当前占用实验自然fixed30且完整精确checkpoint核验归档、实时GPU0独占空闲及无预约后，使用run_y93_bound_native_v2.py；原run_y93_native.py及未执行v1不可作为当前入口。下一次HEAD/占用实验变化仍须新显式同字节绑定，不修改旧签名helper。
- Y91未来终态使用新29+8精确绑定；后续若要选择Y93并推进HEAD，须先从实际保存声明只读确认Y92原目录，再以真实原生证据建立仍在跑Y92的精确终态/观察绑定。不得从本地启动时间猜checkpoint目录，不把日志结束当完整checkpoint校验。
- server2 GPU1全部AdaRES使用及其原冻结后评/成本继续用户暂停，Y57另行实验暂停；不迁移或替换历史资源。仅server1 GPU0既有350W、server2 GPU0固定280W，禁止server6，不改数据、epoch、seed、协议、门槛、功率或调度。无需新增用户处理。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
