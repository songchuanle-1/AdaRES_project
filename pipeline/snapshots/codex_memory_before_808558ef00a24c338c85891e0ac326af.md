# AdaRES_project Research Memory
## Goal And Phase
- 当前token e5ca4e47b09f466ea2a1dc215c0acf3c，快照2026-10-31T08:53:54.363604+00:00，deadline2026-10-31T09:53:54.363604+00:00，不得延长。本轮完成状态只认官方同token收据，不预签postflight。
- 实际CODEX_THREAD_ID=配置owner=01a07e32-78bf-7670-925b-370911ea48db；Pipeline-AdaRES_project Disabled。前6260a6507f484bf890891bff6c6f1de9已核FINALIZED/PASS/context_committed/released，2026-10-31T06:48:32.506353+00:00释放。
- experiment_first=true，本轮快照candidate_budget2/creation2；实际新增准入Y72/Y73共2项，均原生PASS并门控启动/步数递增，预算已达。无新候选ID，无正式指标或科学效益结论；不再凑本轮额度。
- 授权server-1 GPU0既有350W、server-2 GPU0保持280W/GPU1既有390W，每卡一独立实验；Y57实验本身继续用户暂停。禁server-6/跨项目/改调度频率/改功耗硬件/驱逐/重启迁移受保护运行。
- 科学goal字节SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a不变；最新AGENTS/SERVERS及grant=authorizations/server2_gpu1_20261029.json SHA6b24941bb4e278650fb8964af29657bf1d9f71a0bdd6b58c0f241564cec2d838覆盖旧GPU1资源句，原运行收据不重签。
- PROJECT_ROOT=D:/project，PYTHONDONTWRITEBYTECODE=1，PYTHONUTF8=1，使用python -B；密钥SSH已可用，未读密码文件，不推定sudo。
- 旧完整58行机制/准备/失败/专用恢复记忆精确保留：pipeline/snapshots/codex_memory_before_e5ca4e47b09f466ea2a1dc215c0acf3c.md SHAdedf0a13e5ccfad82dfcc4644915fff2fd40ef65975f76d1608d3186d9b6c61a；更早110行记忆见pipeline/snapshots/codex_memory_before_0f90937f76bd42829822d9816707553f.md SHA8e892332cc5128818d9d8a3c7be5bd4768e2fc19f29d56ad4a5d9910970d6009。
## Champion And Baseline
- 目标mIoU .5150；历史CSV Y36 .5110，差.0040。冻结FIRST：Y36约.510725438294、W2 .510107995189、B1 .502221764356、Y52 .509708428366、Y54 .506342952986。训练best不可替代FIRST或晋升；Y36对W2收益仍不确定。
- 原完整后评保留：9508表达/141场景、候选+Y36/W2/B1各两次、FIRST主结果、scene bootstrap10000/seed1999、TRAIN cuts1487/3325、原subgroup harm/mIoU/Acc25/50/activequeries/latency/memory与同卡cost。Y54大目标对Y36 CI[-.02678985,-.00070265]伤害与Y52总体跨0保留，单seed不量化seed方差。
## Active Hypotheses
- Y71视觉乘积融合：原3D/既有2D投影s+a前加(s*a)@W后原sum_norm；零W256x256/no RNG、+65536、总164695173。原全部输入、loss/七共享confidence、hardmask/sampling/SCG/seed1999/fixed30保持。原代码87019、原PID53714不变；本轮生产server2已安全追加Y72/Y73至7da。
- Y72视觉mask细化：六后阶段用原静态mask_feats+clean(当前原ila_ffn视觉)@W，无跨阶段累积、stage0保持。零W65536/总164695173；原共享头/损失/数据/seed1999/fixed30/native YAML不改。保留代码c60c903941d6cd7ffcc343d7c13610f2ba416caa。
- Y72本轮真实server1 GPU0/350W两步TRAIN PASS，loss4.615511417388916/6.384006500244141，四场景16表达/两批134280和350887点；每步540有限梯度张量/10头梯度/7头调用、六current visual/static mask身份与真实预测消费/独立逐标量公式、首零次响应/新梯度和真实更新通过；峰6975127552B。不是效益或全量显存保证；未用val选择/未保存诊断模型。raw=pipeline/analyses/y72_server1_smoke_e5ca4e47b09f466ea2a1dc215c0acf3c.json SHA44d862b6b3d91f812a030bb3a2d3997c6755710f4ceb9a9d8fa011cc42f49f2f。
- Y73初始query锚点：六原qqa_ffn后q+(q_initial-q)@W，同一原stage0未detach锚点；零W65536/总164695173，原static mask/位置/sampling/全部输入/损失/seed1999/fixed30/YAML保持。保留代码7da625ab750447eaad76e82309c89d887df93994(parentY72/c60)。
- Y73本轮真实server2 GPU1/390W两步TRAIN PASS，loss4.615511417388916/6.345763206481934，四原TRAIN场景16表达、每步六current/initial未detach锚点身份及实际预测消费/七头调用/独立公式/首零次响应/新梯度真实更新通过，峰6721372160B；无效益或全量显存保证。raw=pipeline/analyses/y73_gpu1_smoke_e5ca4e47b09f466ea2a1dc215c0acf3c.json SHA3659fde87beaa37a91df04c8fb50ce4a3d0ad32c6a3c0e95940c2f09cff66f67。
- 两项实际原生结果通过scripts/advance_y72_y73.py的版本/goal/YAML/TRAIN/六路径/梯度/数据/资源验证及12项准入负守卫；原Y64完整后评协议深复制核等，仅改方法/commit/授权资源，原training和evaluation内容保持。Y72 receipt=pipeline/admissions/AdaRES-Y72.json SHAe8722b6580739855cf638714de51ecd73bfb0f624bbc7ef6d7cce3db1f09445e；Y73 receipt=pipeline/admissions/AdaRES-Y73.json SHA929e1b3b563dec7ce3dfa52b9e3371d238ca9656f7d341344aee845e7784d761。
- 两份freeze分别pipeline/checks/AdaRES-Y72_e5ca4e47b09f466ea2a1dc215c0acf3c_post_training_freeze.json、AdaRES-Y73_e5ca4e47b09f466ea2a1dc215c0acf3c_post_training_freeze.json；收据引用exact hash。原CPU/native准备、旧失败夹具与计划全部保留，已运行的activate/native/admit/dispatch不重放。
- 本轮先核原Y69/Y70终态：完整Train/Val1..30、原source/YAML/receipt/freeze、稳定finite last30/best/model/optimizer544/scheduler30或20，18/20项生命周期守卫及canonical done_complete/30/30归档；原远端checkpoint保留。Y69训练best .5014769434928894/epoch20；Y70 .5029489994049072/epoch20；都不是正式FIRST。
- Y69终态proof=pipeline/checks/y69_server1_fixed30_e5ca4e47b09f466ea2a1dc215c0acf3c.json SHAc095b779b960c6cd652c7697440f6177841e669914e08756128a8cdcc6389487；原目录/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y69_coarse_geometry_bias_res/20261030_131803，last SHA959bf1c890f2f9e953ead6e8e72f4d9eaed3d804fcbde777ac5dc2a6827e5acd，best SHA83d6e5b72549cfd69b3e3420bc4f38e6739bb9952b950e0162f8442de9e1ff71。
- Y70终态proof=pipeline/checks/y70_server2_fixed30_e5ca4e47b09f466ea2a1dc215c0acf3c.json SHAead75cfa081f12868e54454e287bf458beb510e42638e6a0a10998a994845c80；原目录/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y70_query_scale_fusion_res/20261030_064601，last SHAd32731d68b6dccde4ff6ebe7e4abd5ee9c3ef4c9db800918ef89ca01f3c37b86，best SHA3fab1c259df4eae3ddff1c86fc6fb326b99d8ff54ddb7a5f5667c52d96e08e8b。
- 原Y69/Y70完整日志已分别拉取Checkpoints/server-1或server-2/<原basename>/e5ca4e47b09f466ea2a1dc215c0acf3c/fixed30.log；不重训/不重放finalizer。两者完整冻结复评/cost按原server1GPU0/350W和server2GPU1/390W等待独占资源。
## Closed Or Failed Directions
- Y53/Y55/Y50/Y51/Y49rev1/Y46/Y48等旧拒绝关闭和goal bans不变；F1否决mask-entropy，禁val扫描entropy/margin/occupancy/threshold/temperature/loss/feature subsets/epoch数；不改名重开。
- 本轮初次只读terminal-readiness构造将CR/LF展开进regex而SyntaxError；原.py/.log保留，独立attempt2改raw string并先AST核对后只读成功。repair=pipeline/checks/terminal_readiness_repair_e5ca4e47b09f466ea2a1dc215c0acf3c.json，未影响训练或源码。
- Y70上轮准备初版误import不存在server2传输模块，原三失败源码和失败JSON保留；独立attempt2复用已有通用reconcile_y62_y63_server1传输并20守卫通过，本轮仅执行成功版finalizer。旧Y69 CPU reference/Y67 inplace/旧CPU wrapper与Y72 kwargs夹具等负结果/独立修复见备份及candidate；不把实现错误或梯度视效益。
- server1私有torch4字节栈修复/CUDA11.3库/默认350W与精确路径适配有原证据；不改系统/其他环境/硬件。已归档Y56-Y70等旧产物不重放，旧Y22 rejected字节snapshot/旧Y23专用resume依赖照原AGENTS保留。
## Running And Queue
- 上海17:13左右原生观察：AdaRES-Y71 Train14/30 506→515/4686；AdaRES-Y72 Train1/30 375→383/4686；AdaRES-Y73 Train1/30 40→50/4686。对应PID53714/311445/62829、GPU及功耗如上；均确认原生递增，waiting0。
- 三运行证据：AdaRES-Y71 pipeline/checks/production_adares-y71_e5ca4e47b09f466ea2a1dc215c0acf3c_after_y73_deployment.json SHAeeff998938902879262516c16640ea88b90f7bcde7534f9972aea39fceea0c36；AdaRES-Y72 pipeline/checks/production_adares-y72_e5ca4e47b09f466ea2a1dc215c0acf3c_final.json SHA15d851b5fdf6d8bb145618d1dde296bc35b27a3db1459296f9b2b23bd76e674d；AdaRES-Y73 pipeline/checks/production_adares-y73_e5ca4e47b09f466ea2a1dc215c0acf3c_after_dispatch.json SHA96312f822ec3dde5d044e440f8ba3bdaa34f366afd06fa56c571ffd3cb0bcec5。本行步数若口述不符，以不可变JSON为准，不补写旧证据。
- 本地master由87019→c60→7da，仅将两次各七已保留且实际tested的A状态文件写入精确index，再compare-and-swap更新分支；其他index条目/工作文件不变，旧index原字节已存source_selection*.index。server1由9acd→c60仅additive，server2由87019→7da仅14 additive，原Y71 source/YAML/receipt/PID保持。
- 必须改用新scripts/y72_y73_deployment.py observe --id AdaRES-Y71|AdaRES-Y72|AdaRES-Y73 --token NEW_TOKEN --label progress；只接受明确87019/c60/7da版本边和精确新增文件、原源码/YAML/PID/功耗、实际增长。旧observe_y70_y71_exact_deployment.py现在HEAD/marker不兼容，仅保留历史勿再调用。
- 14项固定30已完成Y56/Y58/Y59/Y60/Y61/Y62/Y63/Y64/Y65/Y66/Y67/Y68/Y69/Y70等待原完整FIRST后评/cost；目前三卡满，无idle。当前occupation：server-2 pipeline/checks/gpu_occupation_server-2_final_evaluations_e5ca4e47b09f466ea2a1dc215c0acf3c.json SHA7629c099eb92823341bf809e9286c87f824df431c5999d492a5b518ef1a2f7f9；server-1 pipeline/checks/gpu_occupation_server-1_final_evaluations_e5ca4e47b09f466ea2a1dc215c0acf3c.json SHA5a7cad7b5ffedb4208553c1d79a02483562e7ce4462e2577c6f2b6ac0703705f；finish须<300秒，超时新label保留原证据。
- server2=192.168.1.163/dl01-MS-7C60，root=/bak_home/songcl/Project/AdaRES_project，code=.../AdaRES，python=/home/songcl/.conda/envs/adares_project/bin/python，原data=/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/；生产7da。
- server1=192.168.1.175/dl02-MS-7B73，root=/data1/songcl/Project/AdaRES_project，code=.../AdaRES，private python=.../runtime/adares_project/bin/python，sitepackages=.../lib/python3.8/site-packages，pointnet=.../runtime/pointnet2-0.0.0-py3.8-linux-x86_64.egg，LD_LIBRARY_PATH=.../runtime/cuda113lib64，data=.../data/，work=.../AdaRES_checkpoints，生产c60。原data proof checks/server1_data_verified_63b27dd8efe04a0bab6dc0291cefdbf0.json SHA8c590fa828fbc6c0df840c21ba88b1802cc0d2323eb0a124a57aa328b0c2361f；只允许SERVERS两批准path mapping。
- Y57 paused/dispatch_hold/resume_requires_user_authorization，原20261021_004739/last.pth epoch27/model/optimizer543/scheduler27/best .5028846859931946保持；仅明确Y57继续授权才从28续同模型/优化器/调度器至fixed30，不迁移/不从pretrain重启。
## Open Questions And Risks
- 尚无本轮正式全量后评/成本或科学胜出结论；official unique-multiple/seed方差及server2掉卡/散热/Xid79根因仍未知。实际原生增长不是硬件修复或效益证明，保留原恢复要求。
- 后继版本部署前须重绑存活source/observer/deployment精确契约；新observer只支持明确87019/c60/7da，禁止容忍任意marker或改旧hash-bound文件。原Y72code c60与当前local7da不同但七源码/YAML逐字相同，不修改原receipt/task commit。
- canonical occupation必须当前token/snapshot/server/hostname/raw哈希、<300秒；未知/失败/缺设备/idle不能证明exclusive hold，不为finish弱化checker。Y69/Y70已按原freeze设置明确required GPU0/1并blocked/validation/frozen_evaluation，当前资源证据已绑定。
- heartbeat仍原owner与原2小时频率，不改通知/调度，不建新线程/自动化或嵌套codex exec；没有真正新授权缺口，不重复索权。
## Last Tick
- 本e5ca4e47b09f466ea2a1dc215c0acf3c preflight先观察，实证Y69/Y70自然终态后完整核验并canonical归档，释放两原槽；Y71原训练全过程保持增长。
- 同轮Y72/server1 GPU0和Y73/server2 GPU1分别真实native PASS、原完整后评freeze/准入、保留版本安全选择、canonical additive sync/dispatch、实际Train1增长；共2准入已达本轮预算。效益未知，不把native或training best作正式指标。
- 当前三运行/14项冻结后评资源等待，三卡真实busy+reserved；handoff=pipeline/checks/y69_y70_y72_y73_handoff_e5ca4e47b09f466ea2a1dc215c0acf3c.json SHA436034f1394c47efd8c5fa0ae5f4d7e57f25692f7ad9c044d7bd49ffdc3bb177。原goal字节/数据/禁用项/运行收据/Y57暂停/调度保持；负的只读脚本尝试和修复原样留档。
- 下一轮只用新observer先核三PID增长及终态；按新快照准备后继独立变体，合格原生检查需真实授权空槽，不挤占当前任务或重放已完成native/准入/归档/启动。
- report/finish后只读核官方FINALIZED/PASS/context_committed/released及锁/pending/archive_pending；释放后不写。本文件不预签收尾。
