# AdaRES_project Research Memory
## Goal And Phase
- 当前token 5d338b76de4643088b71c447e09710b1，snapshot2026-11-03T02:51:37.547167+00:00，deadline2026-11-03T03:51:37.547167+00:00不延长；postflight只认同token官方收据。
- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled；experiment_first enabled，admission2/creation2。本轮新候选0/准入1/启动1/归档1，无正式FIRST/cost。
- PROJECT_ROOT=D:/project，PYTHONDONTWRITEBYTECODE=1，PYTHONUTF8=1，python -B；SSH密钥有效，未读密码文件，不推定sudo。
- 仅server1GPU0既有350W、server2GPU0保持280W/GPU1既有390W；一GPU一独立实验；Y57单独暂停，禁server6/跨项目/改调度/硬件功耗/驱逐/重启迁移受保护运行。
- goal科学字节SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a保持；最新AGENTS/SERVERS与server2_gpu1_20261029覆盖旧GPU1资源句，原receipt不改。
- 前轮完整七节记忆精确备份：pipeline/snapshots/codex_memory_before_5d338b76de4643088b71c447e09710b1.md SHAd284371bc6d780184eccb3b84908ef3b94fea6d5e63335ad2e1b99268230489b；历史科学/禁用/专用恢复与全部原证据沿备份保持。
## Champion And Baseline
- 目标mIoU .5150；历史CSV Y36 .5110。冻结FIRST：Y36 .510725438294、W2 .510107995189、B1 .502221764356、Y52 .509708428366、Y54 .506342952986。本轮无正式全量FIRST/成本更新；训练best不替代FIRST或晋升。
- 原后评协议：9508表达/141场景、候选+Y36/W2/B1各两次、FIRST主结果、scene bootstrap10000/seed1999、TRAIN cuts1487/3325、原subgroup harm/mIoU/Acc25/50/activequeries/latency/memory和同卡cost。Y54大目标对Y36伤害CI[-.02678985,-.00070265]与Y52总体跨0保留；单seed不估seed方差。
## Active Hypotheses
- Y78本轮专用finalize_y78_fixed30_v2.py核22生命周期守卫、原source/YAML/receipt/freeze/完整Train-Val1..30和稳定finite last30/best21/model/optimizer544/scheduler30-21，canonical显式GPU1 done_complete归档。原目录20261102_000333产物保留；训练best .5039719343185425不是FIRST。terminal：pipeline/checks/y78_server2_fixed30_5d338b76de4643088b71c447e09710b1.json SHA7105bf1c928444d2c3743b89b3b2df6b31cc70200428dd331bf7e7cfeaa5209f；archive：pipeline/checks/y78_server2_fixed30_5d338b76de4643088b71c447e09710b1_archive_binding.json SHA388448a3efb4f5fd35305b27ed6f6c1518f1b1165dfd6763c9c271945eb612b7。不要重放prepare/终态或重训，完整后评仍原GPU1/390W。
- Y80独立W2六次query-set→visual-FFN attention残差，零W256x256/no RNG/+65536，原static mask与六qia/五next-lia/七共享confidence-query_norm保持；在bd938轮server2GPU0/280W真实原native YAML/TRAIN两步PASS，loss4.615510940551758/6.330480575561523，峰reserved7004487680B；仅两批，不是效益或全量显存保证。
- Y80原生记录：pipeline/checks/AdaRES-Y80_bd93870b90dd48b7b5366e46173cc01e_native_smoke.json SHA68af8d4cd961721ec5bfbbf3abd4061e1d70e50f2605fc0089094dd7855ae05c；原始实际结果pipeline/analyses/y80_gpu0_smoke_bd93870b90dd48b7b5366e46173cc01e.json SHA7d7c5ca15d1c40e6a2d122f9760cd2e5de97de37284901b1d9fff24031f5ecdd；17准入/源链/资源拒绝守卫通过。
- Y80 receipt：pipeline/admissions/AdaRES-Y80.json SHA9ff6b809a01d04aa59be9d3d3ec300fffac6169195d50c085057532a764e7d3c；完整后评freeze：pipeline/checks/AdaRES-Y80_bd93870b90dd48b7b5366e46173cc01e_post_training_freeze.json SHA41ce8a10306c4426a75b44e2db2c8f70107d16c76909dc73c9a9f7d4fe30e42a；原Y64 evaluation/training深复制核等，不缩减后评。
- Y77在bd938轮专用finalize_y77_fixed30.py核20守卫、原source/YAML/receipt/freeze、完整Train/Val1..30及稳定finite last30/best21/model/optimizer544/scheduler30和21后，canonical显式GPU0 done_complete归档；原目录20261101_175820/全部产物保留。训练best .5108122825622559不是FIRST。
- Y77 terminal：pipeline/checks/y77_server2_fixed30_bd93870b90dd48b7b5366e46173cc01e.json SHAfc13e7e847de5946762f7afe3df68c9cf33acf6dd2e651308633f0e0edbb6dca；archive：pipeline/checks/y77_server2_fixed30_bd93870b90dd48b7b5366e46173cc01e_archive_binding.json SHA6297e821c7d4aecca76f1e2889d900659a1f6ac34e7513d84bb874c51bba537d；后评仍原server2GPU0/280W，现由Y80占用。不要重放terminal/prepare或重训。
- Y81=adares_y81_query_mask_refinement；独立W2六次mask_base+current refined-query-set attention残差，零W65536/no RNG/总164695173；原static base逐次重用不累积，初始prediction/全输入/2D/backbone/原hard gates-SCG-k-gamma/sampling/七共享confidence头targets-loss-selector/seed1999/fixed30/native YAML保持。此前CPU准备不重放，本轮实际原native GPU1/390W两步PASS。
- Y81原native实测loss4.615511417388916/6.37349796295166、四原场景16表达、12refinement/prediction/original coarse-gate消费者与14shared-head/query_norm、float64参考/首零次响应/有限新-原-输入梯度与真实更新通过，峰reserved7067402240B。7预准入资源/source-chain守卫与17实际native准入守卫通过；两批不证明效益或全量显存上界。
- Y82=adares_y82_scale_product；独立W2六次(coarse+fine)/2+(coarse*fine)@W，共享零W256x256/no RNG/+65536/总164695173。两原branch全channel/128query未detach，初始prediction/static masks/Boolean gate算法/SCG/k/gamma/sampling/geometry/原全输入/2D/backbone/七共享头targets-loss-selector/seed1999/fixed30/native YAML保持。
- Y82不同于Y70的query条件凸混合与Y71的3D/2D输入乘积位置，不组合或重开旧路线；不宣称新信息/创新/效益，无val扫描。原model/backbone/TRAIN162738点8表达、float32/64独立逐标量乘积投影、zero/RNG/全输入梯度/置换/branch swap/同均值不同乘积/错配/非零-bypass、七阶段W2与输入梯度锚点、两合成CPU优化步和原coarse/fine/qla/shared head/query_norm/hard gates检查通过。
- Y82实际CPU：pipeline/analyses/y82_cpu_bd93870b90dd48b7b5366e46173cc01e.json SHA2837d258176dcd7e2a3bd744237eee38b501343e09ba0b77904f0bdb6cf39fb7；version：pipeline/checks/y82_cpu_version_bd93870b90dd48b7b5366e46173cc01e.json SHAab5ffaea3e6c89d9a5b159d17156ff1993bb2a5c4f2cc77e5e54935340f194bb；commit5773768ffb87d7fea7432e85fd0da561556aa2fc，parentY81/415793，branch refs/heads/adares-y82-scale-product。仅七测试A文件隔离index保留，主HEAD/index/生产运行保持。
- Y82双原native GPU计划：pipeline/checks/y82_native_bd93870b90dd48b7b5366e46173cc01e.json SHA1a450adf2e85912333500ce302f1053401bcd2c4a95aaa10e447eb781cd316e0；11拒绝守卫/陈旧goal-源码篡改拒绝/Python3.8 AST/12原native回调与实际CPU回调AST一致通过。无CUDA/native sparse TRAIN任务/GPU PASS/准入。
- Y81已在GPU1独立训练；剩Y82已CPU/model/data/version/native-plan/callback准备、blocked/validation/exclusive_gpu_slot/server2GPU0，GPU1原计划备选，无server1计划。无需重放CPU，fresh匹配独占idle/no compute/no reservation/原功耗后原native两步PASS才能原完整后评freeze/receipt/精确source契约/安全派发和步数递增。
- Y81本轮receipt：pipeline/admissions/AdaRES-Y81.json SHA76f21c9d98991ae8bc7f41cb6e3ad41bfc96301f2caf4be222161bbe7c8f2648；native smoke：pipeline/checks/AdaRES-Y81_5d338b76de4643088b71c447e09710b1_native_smoke.json SHA993c1610fc11c6bb48fb042e2613fdbeb11b8a5065792410139710fdd3c72d8e；原完整后评freeze：pipeline/checks/AdaRES-Y81_5d338b76de4643088b71c447e09710b1_post_training_freeze.json SHAb7e6407383e14330fff75b3fcc66f553510df9b479794084dde6cffde1c31ed4。原Y64完整evaluation/training深复制核等；commit4157937d0c3cf344a44394ce15c95349aa94d50f精确七A文件选择，旧receipt和科学goal字节不改。
## Closed Or Failed Directions
- Y80实际CPU及原native回调首次执行通过；元数据核验helper把15函数数目的宽文本替换误及162738点为152738，初次assert失败；attempt2仅修元数据预期，原模型/计划/实测结果不变，失败helper/repair证据保留于CPU verification。
- 保留Y53/Y55/Y50/Y51/Y49rev1/Y46/Y48等拒绝及goal bans；F1否决mask-entropy，禁val扫描entropy/margin/occupancy/threshold/temperature/loss/feature subsets/epoch数，不改名重开。旧Y22/Y23与全部专用恢复见AGENTS和精确备份。
- Y78原两CPU wrapper失败.py/.log保持：float64置换逐位比较1.11e-16舍入在原128eps范围；decoder-only缺原text_encoder优化器group，attempt3完整model绑定通过。模型与YAML未改，revision0；不是机制否决或GPU PASS。Y79在08e82轮CPU/model/原生回调检查首次即通过。
- 所有已完成fixed30模型/原checkpoint/日志/freeze/receipt保留，不重训、不重放终态helper。原server1私有torch/CUDA11.3库、server2掉卡与专用恢复要求均不变。
- 收尾wrapper初次误假设canonical occupation不会改tick_attempt，实际末次server1资源证据绑定Y62；attempt2保留该真实资源记录并重新绑定本轮已执行Y79 CPU结果。原失败helper及repair证据保留，模型/科学控制/队列未改。repair：pipeline/checks/closeout_wrapper_repair_08e82df71e3743dcb86c0eec4d39460c.json SHA364e230b88f573c54a5e62fdfece02640555d884a0ca09d93a1106b3a99d4d09。
- Y81在b5d4轮实际CPU和完整原native回调CPU检查首次通过；未出现科学失败或模型修订。旧helper/失败程序/原日志保持不可变；不得修改旧hash-bound helper来支持新HEAD。
- bd938轮Y80实际GPU与17准入守卫、Y82实际CPU/11native计划守卫首次通过；旧hash-bound helper/失败probe/原日志均保持，不为新HEAD修改旧helper。
- Y78归档准备v1复查发现两个遗留GPU0字段（归档后断言/后评required_gpu），当时尚未执行归档；v1全部源码和原plan保留。v2从已核GPU1 task派生字段并新增两项终态后资源守卫，22检查通过；当前terminal_handoff_plan绑定v2及supersedes链，不改变科学模型。 本轮仅v2完整真实归档成功，原v1未运行归档。
## Running And Queue
- AdaRES-Y80 server-2 GPU0 PID42226 source/YAML/receipt/marker/计算归属/原功耗和native递增核验通过：Train6/30 step2234/4686；pipeline/checks/production_adares-y80_5d338b76de4643088b71c447e09710b1_launch.json SHAda17efa6597697ee638bfd37ebbbd828d3f0464f19e8c5ae1b9038949bcf3f63。
- AdaRES-Y81 server-2 GPU1 PID120624 source/YAML/receipt/marker/计算归属/原功耗和native递增核验通过：Train1/30 step368/4686；pipeline/checks/production_adares-y81_5d338b76de4643088b71c447e09710b1_launch.json SHA91e82429914fbadd529bec220b7cc627bba8171407c0909f126c6986f7d082ec。
- AdaRES-Y79 server-1 GPU0 PID2218231 source/YAML/receipt/marker/计算归属/原功耗和native递增核验通过：Train15/30 step2979/4686；pipeline/checks/production_adares-y79_5d338b76de4643088b71c447e09710b1_launch.json SHAd9dea27937871d12f72d21569e1283cbd956309e8cacd600566aec4e9457cc8c。
- 主master/server2 marker4157937d0c3cf344a44394ce15c95349aa94d50f，server1仍e0c7dba199fcf1a009941450d04aeb799418a7e6；仅选择七已测试Y81 A文件，server2从4e68安全ff追加七文件，原Y79/Y80源码保持。现用scripts/y81_deployment.py observe --id AdaRES-Y79|AdaRES-Y80|AdaRES-Y81 --token NEW_TOKEN --label progress；旧y80及更早observer不支持新HEAD。
- 22项Y56/Y58..Y78 fixed30已完成，等待各自原冻结完整FIRST/cost；保留原required server/GPU/功耗，不重训、不用训练best代替FIRST。
- fresh occupation server2：pipeline/checks/gpu_occupation_server-2_closeout_5d338b76de4643088b71c447e09710b1.json SHAfc11c57b2c217a163217a943ea109620391760dbe49d076c9066fde1055a455d；server1：pipeline/checks/gpu_occupation_server-1_closeout_5d338b76de4643088b71c447e09710b1.json SHA6b0c9359bc82564fc16c435fd0ee9dc5f4adbd471a4e246c97b59bf09e6bf685；三卡busy+reserved。finish证据须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。
- server1=songcl@192.168.1.175/dl02-MS-7B73，root=/data1/songcl/Project/AdaRES_project，private python=.../runtime/adares_project/bin/python，sitepackages=.../lib/python3.8/site-packages，pointnet=.../runtime/pointnet2-0.0.0-py3.8-linux-x86_64.egg，LD_LIBRARY_PATH=.../runtime/cuda113lib64，原精确data=.../data/，work=.../AdaRES_checkpoints；仅SERVERS两授权path mappings，原数据证明保持。
- server2=songcl@192.168.1.163/dl01-MS-7C60，root=/bak_home/songcl/Project/AdaRES_project，code=.../AdaRES，python=/home/songcl/.conda/envs/adares_project/bin/python，原data=/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/；原环境/data保持。
- Y57单独paused/dispatch_hold/resume_requires_user_authorization；原20261021_004739/last.pth epoch27/model/optimizer543/scheduler27/best .5028846859931946保留，只有单独明确Y57继续授权才能从28续至fixed30，不迁移/不从pretrain重启。
## Open Questions And Risks
- 原完整9508表达/141场景、候选+Y36/W2/B1各两次/FIRST/bootstrap10000/seed1999/TRAIN cuts1487-3325/原subgroup harm与同卡成本协议保持；本轮无正式科学胜出/效益或全量成本。
- Y82未实际GPU任务，不得准入；22个完成模型完整后评仍需原冻结资源。单seed方差/server2历史掉卡Xid79根因仍未知，训练递增不证明硬件修复。
- 无新增项目常规操作授权缺口，不重复索权；Y57原epoch27/model/optimizer543/scheduler27单独暂停，需其单独明确继续授权。
- 下轮先y81_deployment核三原运行并派发现有任务；Y82最近原native检查等待fresh授权独占匹配空槽。Y79后续自然fixed30终态须建立其原PID/source/YAML/receipt/freeze/原目录/完整Train-Val和稳定finite检查的专用契约再显式server1GPU0归档，不直接套其他ID helper。
- record_prepared_y81_y82_wait.py假设Y78仍训练/旧HEAD，本轮后不再适用；record_y81_launch_closeout.py为当前事件收尾，不重放。下一轮另以当前token保存真实资源/进度复核。保留当前heartbeat owner/频率/通知，不建线程/新自动化/嵌套codex exec。
## Last Tick
- 5d338b76de4643088b71c447e09710b1 begin/design先查并派发waiting0；Y78原PID完成后绑定v2/22守卫及原freeze-dependencies完成完整稳定fixed30核验和canonical显式GPU1归档，未影响其他任务/Y57暂停。
- Y81本轮7资源/source生命周期守卫和实际原native GPU1/390W两步/17准入守卫通过；精确测试版本选择/原完整后评freeze/receipt/入队/安全派发，并确认native递增。新候选0/准入1/启动1/归档1；剩Y82原准备完整，未重放CPU，无正式FIRST/cost。
- handoff：pipeline/checks/y81_launch_y82_wait_5d338b76de4643088b71c447e09710b1.json SHA02740fe777f160ae3b5975bc73e03def2e77ee6f4cab776abd862e52403fb70e；当前HEAD/server2 marker415793、server1仍e0c。Y79/Y80原训练保持，Y81接原GPU1独立训练，Y78全部原产物保留。
- 后续先核三原运行/现有队列；空槽时Y82原native PASS后完整冻结后评/准入/派发。Y57继续单独暂停，所有原数据/mapping/私有环境/禁用/seed1999/fixed30与350W-280W-390W保持。
- 官方report/finish后仅只读核FINALIZED/PASS/context_committed/released及无lock/pending/archive_pending；此记忆不预签postflight，释放后不写。
