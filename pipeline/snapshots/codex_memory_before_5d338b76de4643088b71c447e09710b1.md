# AdaRES_project Research Memory
## Goal And Phase
- 当前token cdd559a609d54988857e341abae24d98，snapshot2026-11-03T00:48:48.315017+00:00，deadline2026-11-03T01:48:48.315017+00:00不延长；postflight只认同token官方收据。
- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled；experiment_first enabled，admission2/creation2。本轮新候选0/准入0/启动0/归档0；Y78专用终态准备22守卫与只读原目录检查通过。无正式FIRST/cost。
- PROJECT_ROOT=D:/project，PYTHONDONTWRITEBYTECODE=1，PYTHONUTF8=1，python -B；SSH密钥有效，未读密码文件，不推定sudo。
- 仅server1GPU0既有350W、server2GPU0保持280W/GPU1既有390W；一GPU一独立实验；Y57单独暂停，禁server6/跨项目/改调度/硬件功耗/驱逐/重启迁移受保护运行。
- goal科学字节SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a保持；最新AGENTS/SERVERS与server2_gpu1_20261029覆盖旧GPU1资源句，原receipt不改。
- 前轮完整七节记忆精确备份：pipeline/snapshots/codex_memory_before_cdd559a609d54988857e341abae24d98.md SHAc52fc96b5fcda929225cd28606a6c13c83a3ac9f9206dc4e715e8488d3e97ac4；历史机制/禁用/专用恢复与全部原证据沿备份保持。
## Champion And Baseline
- 目标mIoU .5150；历史CSV Y36 .5110。冻结FIRST：Y36 .510725438294、W2 .510107995189、B1 .502221764356、Y52 .509708428366、Y54 .506342952986。本轮无正式全量FIRST/成本更新；训练best不替代FIRST或晋升。
- 原后评协议：9508表达/141场景、候选+Y36/W2/B1各两次、FIRST主结果、scene bootstrap10000/seed1999、TRAIN cuts1487/3325、原subgroup harm/mIoU/Acc25/50/activequeries/latency/memory和同卡cost。Y54大目标对Y36伤害CI[-.02678985,-.00070265]与Y52总体跨0保留；单seed不估seed方差。
## Active Hypotheses
- Y78专用终态准备：pipeline/checks/y78_terminal_preparation_v2_cdd559a609d54988857e341abae24d98.json SHAb2ae7f3273d802fda5ff0acec3d668eb5338ceafc528f1883ae07c24e66aa4e5；22生命周期拒绝守卫、六原源码/YAML/receipt/freeze/生产marker和声明目录20261102_000333只读核验通过；原PID22192/server2GPU1/390W/源5eee与当前HEAD4e68精确Y78→Y79→Y80 A边绑定。未加载checkpoint/未full terminal验证/未停止或归档。未来自然fixed30退出后先核plan.dependencies及原receipt/freeze，再专用finalize_y78_fixed30_v2.py；不要重放prepare。
- Y80独立W2六次query-set→visual-FFN attention残差，零W256x256/no RNG/+65536，原static mask与六qia/五next-lia/七共享confidence-query_norm保持；在bd938轮server2GPU0/280W真实原native YAML/TRAIN两步PASS，loss4.615510940551758/6.330480575561523，峰reserved7004487680B；仅两批，不是效益或全量显存保证。
- Y80原生记录：pipeline/checks/AdaRES-Y80_bd93870b90dd48b7b5366e46173cc01e_native_smoke.json SHA68af8d4cd961721ec5bfbbf3abd4061e1d70e50f2605fc0089094dd7855ae05c；原始实际结果pipeline/analyses/y80_gpu0_smoke_bd93870b90dd48b7b5366e46173cc01e.json SHA7d7c5ca15d1c40e6a2d122f9760cd2e5de97de37284901b1d9fff24031f5ecdd；17准入/源链/资源拒绝守卫通过。
- Y80 receipt：pipeline/admissions/AdaRES-Y80.json SHA9ff6b809a01d04aa59be9d3d3ec300fffac6169195d50c085057532a764e7d3c；完整后评freeze：pipeline/checks/AdaRES-Y80_bd93870b90dd48b7b5366e46173cc01e_post_training_freeze.json SHA41ce8a10306c4426a75b44e2db2c8f70107d16c76909dc73c9a9f7d4fe30e42a；原Y64 evaluation/training深复制核等，不缩减后评。
- Y77在bd938轮专用finalize_y77_fixed30.py核20守卫、原source/YAML/receipt/freeze、完整Train/Val1..30及稳定finite last30/best21/model/optimizer544/scheduler30和21后，canonical显式GPU0 done_complete归档；原目录20261101_175820/全部产物保留。训练best .5108122825622559不是FIRST。
- Y77 terminal：pipeline/checks/y77_server2_fixed30_bd93870b90dd48b7b5366e46173cc01e.json SHAfc13e7e847de5946762f7afe3df68c9cf33acf6dd2e651308633f0e0edbb6dca；archive：pipeline/checks/y77_server2_fixed30_bd93870b90dd48b7b5366e46173cc01e_archive_binding.json SHA6297e821c7d4aecca76f1e2889d900659a1f6ac34e7513d84bb874c51bba537d；后评仍原server2GPU0/280W，现由Y80占用。不要重放terminal/prepare或重训。
- Y81=adares_y81_query_mask_refinement；独立W2六次mask_base+query-set attention残差，零W65536/no RNG，原static base逐次重用不累积。既有完整CPU/model/data/标量/七阶段锚点/两优化步/11native计划守卫/13冻结回调CPU实际执行已通过；无GPU PASS。commit4157937d0c3cf344a44394ce15c95349aa94d50f；精确机制/全部原证据见前轮备份。
- Y81 version：pipeline/checks/y81_cpu_version_b5d4f7646f114f71b5f59d796e2b1858.json SHA006cf2d7c383c512639ccd7d324e411344e48423408585e2c219c2d1c1abcb2f；native preparation：pipeline/checks/y81_native_b5d4f7646f114f71b5f59d796e2b1858.json SHA98d46ba0100809e2f515d607600d39ee2445529fcde440e849292d21fa70ed3a；不重放已通过CPU/prepare。
- Y82=adares_y82_scale_product；独立W2六次(coarse+fine)/2+(coarse*fine)@W，共享零W256x256/no RNG/+65536/总164695173。两原branch全channel/128query未detach，初始prediction/static masks/Boolean gate算法/SCG/k/gamma/sampling/geometry/原全输入/2D/backbone/七共享头targets-loss-selector/seed1999/fixed30/native YAML保持。
- Y82不同于Y70的query条件凸混合与Y71的3D/2D输入乘积位置，不组合或重开旧路线；不宣称新信息/创新/效益，无val扫描。原model/backbone/TRAIN162738点8表达、float32/64独立逐标量乘积投影、zero/RNG/全输入梯度/置换/branch swap/同均值不同乘积/错配/非零-bypass、七阶段W2与输入梯度锚点、两合成CPU优化步和原coarse/fine/qla/shared head/query_norm/hard gates检查通过。
- Y82实际CPU：pipeline/analyses/y82_cpu_bd93870b90dd48b7b5366e46173cc01e.json SHA2837d258176dcd7e2a3bd744237eee38b501343e09ba0b77904f0bdb6cf39fb7；version：pipeline/checks/y82_cpu_version_bd93870b90dd48b7b5366e46173cc01e.json SHAab5ffaea3e6c89d9a5b159d17156ff1993bb2a5c4f2cc77e5e54935340f194bb；commit5773768ffb87d7fea7432e85fd0da561556aa2fc，parentY81/415793，branch refs/heads/adares-y82-scale-product。仅七测试A文件隔离index保留，主HEAD/index/生产运行保持。
- Y82双原native GPU计划：pipeline/checks/y82_native_bd93870b90dd48b7b5366e46173cc01e.json SHA1a450adf2e85912333500ce302f1053401bcd2c4a95aaa10e447eb781cd316e0；11拒绝守卫/陈旧goal-源码篡改拒绝/Python3.8 AST/12原native回调与实际CPU回调AST一致通过。无CUDA/native sparse TRAIN任务/GPU PASS/准入。
- Y81优先Y82；二者均blocked/validation/exclusive_gpu_slot/server2GPU0，GPU1原计划备选。先canonical匹配owner完整归档，再fresh独占idle/no compute/no reservation/原功耗与匹配冻结plan的native两步PASS，才能原完整后评freeze/receipt/精确source契约/安全派发并核计数；无server1计划。
## Closed Or Failed Directions
- Y80实际CPU及原native回调首次执行通过；元数据核验helper把15函数数目的宽文本替换误及162738点为152738，初次assert失败；attempt2仅修元数据预期，原模型/计划/实测结果不变，失败helper/repair证据保留于CPU verification。
- 保留Y53/Y55/Y50/Y51/Y49rev1/Y46/Y48等拒绝及goal bans；F1否决mask-entropy，禁val扫描entropy/margin/occupancy/threshold/temperature/loss/feature subsets/epoch数，不改名重开。旧Y22/Y23与全部专用恢复见AGENTS和精确备份。
- Y78原两CPU wrapper失败.py/.log保持：float64置换逐位比较1.11e-16舍入在原128eps范围；decoder-only缺原text_encoder优化器group，attempt3完整model绑定通过。模型与YAML未改，revision0；不是机制否决或GPU PASS。Y79在08e82轮CPU/model/原生回调检查首次即通过。
- 所有已完成fixed30模型/原checkpoint/日志/freeze/receipt保留，不重训、不重放终态helper。原server1私有torch/CUDA11.3库、server2掉卡与专用恢复要求均不变。
- 收尾wrapper初次误假设canonical occupation不会改tick_attempt，实际末次server1资源证据绑定Y62；attempt2保留该真实资源记录并重新绑定本轮已执行Y79 CPU结果。原失败helper及repair证据保留，模型/科学控制/队列未改。repair：pipeline/checks/closeout_wrapper_repair_08e82df71e3743dcb86c0eec4d39460c.json SHA364e230b88f573c54a5e62fdfece02640555d884a0ca09d93a1106b3a99d4d09。
- Y81在b5d4轮实际CPU和完整原native回调CPU检查首次通过；未出现科学失败或模型修订。旧helper/失败程序/原日志保持不可变；不得修改旧hash-bound helper来支持新HEAD。
- bd938轮Y80实际GPU与17准入守卫、Y82实际CPU/11native计划守卫首次通过；旧hash-bound helper/失败probe/原日志均保持，不为新HEAD修改旧helper。
- Y78归档准备v1复查发现两个遗留GPU0字段（归档后断言/后评required_gpu），尚未执行归档；v1全部源码和原plan保留。v2从已核GPU1 task派生字段并新增两项终态后资源守卫，22检查通过；当前terminal_handoff_plan绑定v2及supersedes链，不改变科学模型。
## Running And Queue
- AdaRES-Y80 server2GPU0/280W PID42226 原source/YAML/receipt/marker/计算归属/功耗和native递增已核：Train3/30 step1826/4686；pipeline/checks/production_adares-y80_cdd559a609d54988857e341abae24d98_progress.json SHAda0592a123800d87d9ac4a0cbf4e8a85a93c0371498f20673f8c4bb1b1737ba5。
- AdaRES-Y78 server2GPU1/390W PID22192 原source/YAML/receipt/marker/计算归属/功耗和native递增已核：Train28/30 step344/4686；pipeline/checks/production_adares-y78_cdd559a609d54988857e341abae24d98_progress.json SHAa4c446d7ba9fdf96be92b821376c1d495a8a334db872248f5cb0edc8a4c71181。
- AdaRES-Y79 server1GPU0/350W PID2218231 原source/YAML/receipt/marker/计算归属/功耗和native递增已核：Train12/30 step3928/4686；pipeline/checks/production_adares-y79_cdd559a609d54988857e341abae24d98_progress.json SHA6bd69fe04d87df0b4f5d6c816cf0def6e3daed323a9fe9a2c997b9578cab8866。
- 主master/server2 marker4e68cd84d56d890222adc2bec8ab59cb0c8715b6；server1 marker仍e0c7dba199fcf1a009941450d04aeb799418a7e6。仅选择七已测试Y80 A文件；server2从5eee安全ff追加Y79/Y80共14文件，原Y78源码保持。现用scripts/y80_deployment.py observe --id AdaRES-Y78|AdaRES-Y79|AdaRES-Y80 --token NEW_TOKEN --label progress；旧y79及更早observer不支持新HEAD。
- 21项Y56/Y58..Y77 fixed30已完成，等待各自原冻结完整FIRST/cost；保留原required server/GPU/功耗，不重训、不用训练best代替FIRST。
- fresh occupation server2：pipeline/checks/gpu_occupation_server-2_finish_cdd559a609d54988857e341abae24d98.json SHAd0b3c35bb9692e289bb388a8e73110a2891a1db6e2f12497b6c751f39bb5fdce；server1：pipeline/checks/gpu_occupation_server-1_finish_cdd559a609d54988857e341abae24d98.json SHAbb9409634713c21f45ab8888ee48cba255ecca74f42329f3e79d7abb610fce09；三卡busy+reserved。finish证据须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。
- server1=songcl@192.168.1.175/dl02-MS-7B73，root=/data1/songcl/Project/AdaRES_project，private python=.../runtime/adares_project/bin/python，sitepackages=.../lib/python3.8/site-packages，pointnet=.../runtime/pointnet2-0.0.0-py3.8-linux-x86_64.egg，LD_LIBRARY_PATH=.../runtime/cuda113lib64，原精确data=.../data/，work=.../AdaRES_checkpoints；仅SERVERS两授权path mappings，原数据证明保持。
- server2=songcl@192.168.1.163/dl01-MS-7C60，root=/bak_home/songcl/Project/AdaRES_project，code=.../AdaRES，python=/home/songcl/.conda/envs/adares_project/bin/python，原data=/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/；原环境/data保持。
- Y57单独paused/dispatch_hold/resume_requires_user_authorization；原20261021_004739/last.pth epoch27/model/optimizer543/scheduler27/best .5028846859931946保留，只有单独明确Y57继续授权才能从28续至fixed30，不迁移/不从pretrain重启。
## Open Questions And Risks
- 原完整9508表达/141场景、候选+Y36/W2/B1各两次/FIRST/bootstrap10000/seed1999/TRAIN cuts1487-3325/原subgroup harm与同卡成本协议保持；本轮无正式科学胜出/效益或全量成本。
- Y81/Y82未做实际GPU任务，不得准入；已有完成模型后评等待原资源。单seed方差与server2历史掉卡/Xid79根因仍未知，训练递增不证明硬件修复。
- 无新增项目常规操作授权缺口，不重复索权；Y57原epoch27/model/optimizer543/scheduler27单独暂停，需其单独明确继续授权。
- 下轮先y80_deployment核三原运行并派发现有任务；Y78自然fixed30终态先核其terminal_handoff_plan全部依赖/原receipt/freeze，再python -B scripts/finalize_y78_fixed30_v2.py --token NEW_TOKEN完整原目录/Train-Val1..30/稳定finite last30-best/model-optimizer-scheduler与精确source链核验，canonical显式GPU1归档；不可套Y77 helper。随后Y81原GPU1/390W native计划在fresh空槽激活，Y82随后。
- 不调用旧record_prepared_y80_y81_wait.py或事件型record_y80_y82_closeout.py；当前record_prepared_y81_y82_wait.py支持HEAD4e68与Y78/Y79/Y80运行、Y81/Y82等待、Y78可选新终态准备。未来安静复核使用新token且不带--new-terminal-preparation，不重放prepare/CPU。heartbeat原owner/频率/通知保持，不建线程/新自动化/嵌套codex exec。
## Last Tick
- cdd559a609d54988857e341abae24d98 begin/design先查并尝试派发waiting0；快照new_terminal:Y77是bd938轮既有归档，不重放。Y78/Y79/Y80原native进度递增，无新完成/故障，fresh三卡busy+reserved。
- 本轮新候选0/准入0/启动0/归档0；Y78独立专用终态准备/22守卫及原source/YAML/receipt/freeze/marker/声明目录只读核验首次通过；未加载checkpoint，仍原训练。Y81/Y82已准备，仍待真实native GPU任务；无FIRST/cost。
- handoff：pipeline/checks/prepared_y81_y82_resource_cdd559a609d54988857e341abae24d98.json SHA0821c1ad1e3baeeb92a6596d7a2ba68288cc81722f1ca843c88045d723a861e9；tick_attempt绑定AdaRES-Y78本轮实际终态准备。主HEAD/server2 marker4e68，server1仍e0c，当前observer为y80_deployment。
- Y78真实自然fixed30退出后原专用完整核验/canonical显式GPU1归档，再fresh独占idle/no compute/no reservation/390W后Y81原native两步PASS→原完整后评freeze/receipt/精确source契约/安全派发及native递增；Y82随后，Y57单独暂停。
- 官方report/finish后仅只读核FINALIZED/PASS/context_committed/released及无lock/pending/archive_pending；此记忆不预签结果，释放后不写。
