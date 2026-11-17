# AdaRES 迁移交接文档

交接日期：2026-11-17，Asia/Shanghai。项目为 `AdaRES_project`，不是
`AdaQ3D_project`。用户本次要求上传代码、统计实验并整理交接，**尚未指定或授权新服务器运行**。

## 1. 先读结论

- 项目任务：ScanRefer 3D referring expression segmentation。沿用发布的 SPFormer
  sparse-UNet backbone，研究 query、decoder 和 confidence/readout 结构。
- 科学目标：mIoU >= 0.515，同时保留准确率、活跃 query、时间与显存报告协议。
- 当前最好的已登记训练指标是 Y100 的 0.5164485573768616；正式 FIRST、配对比较、
  成本与后审尚待完成。**训练 best 超过阈值不是正式科学达标或论文结论**。
- Y36 原始完整评测是 `complete_inconclusive`；相对 W2 的总体置信区间穿零，
  未证明优于 W2。不得把流水线的 rounded champion 自动解释为科学晋升。
- Y90/Y92/Y94 完整后评拒绝其具体实现，原因是预注册子组伤害；不是机制全族证伪。
  Y96/Y99/Y101 的完整后评为 `complete_inconclusive`，不晋升。
- 当前阻塞：Y102 原进程仍在，但原生日志停在 Train1/30 step7/4686。
  本次只读查询没有发信号、重启、恢复训练或加载 checkpoint。

## 2. 代码备份及身份

GitHub 私有仓库：`songchuanle-1/AdaRES_project`。
`main` 为本次完整交接树，`history/*` 保留原始分支及原提交身份。
本地生产仓库原路径 `D:/project/AdaRES_project`，活动分支 `master`，
原 HEAD `0c6c844719c8466ddd8888d054fe3b7ee658e643`。

本次发布保留本地未提交源码的实际字节，不重置工作树、不切换活动分支，
不改原暂存区、不覆盖原运行源提交，不把历史收据重签成新的交接提交。
Y103 等仅在其他本地分支存在的源码也纳入交接树，并在发布清单标注原分支/提交。
所有原分支另以 `history/原分支名` 保存，可按原 SHA 找回冻结版本。

本地原代码只有模型、配置与部分工具，缺少共享运行模块。本次从已核验的
server-2 原部署目录**只读取得源码**，补齐 dataset、utils、定制 `ipdn.torch`、
PointGroup/PointNet2 C++/CUDA 源码及构建文件。已有本地文件优先保留；
若与 server-2 不同，原服务器版本另存 `vendor/server2_source/`，
差异和来源 SHA256 见 `publication_manifest.json`。

交接快照不是任一已测实验的统一冻结源树。复现单个实验应查该 ID 的准入收据、
冻结 source manifest、原提交和部署路径适配，不用快照 HEAD 冒充历史 tested commit。
控制工具位于 `vendor/songcl_research_pipeline/`，仅为可追溯归档；
原脚本中的 Windows 路径与 owner 绑定没有被擅自改写。

## 3. 实验统计口径

机器可读明细见 `experiment_inventory.csv/json`，可读表见 `EXPERIMENTS.md`。

| 维度 | 数量/状态 |
|---|---|
| 有训练记录的唯一稳定 ID | 85 |
| 注册为 done_complete | 81 |
| done_earlystop / closed_low_metric | 各 1 |
| 用户暂停 / 登记运行中 | 各 1：Y57 / Y102 |
| 候选记录 | 155 |
| 候选 complete / rejected / closed | 19 / 82 / 12 |
| 候选 blocked / training | 41 / 1 |
| 合并台账涉及唯一 ID | 159 |
| 原 CSV 行数 / 唯一 ID | 87 / 83 |
| 独立实验注册 JSON | 85 |
| 准入收据文件 | 79，包含历史/修订，不是训练次数 |

原 CSV 混合了两种列顺序，本次按实际 schema 解析，并按稳定 ID 去重。
B2/B4 的 E30 是延长原训练，不是新的独立从头训练。H5 的原登记为
done_earlystop，即使 epochs 字段为30/30也不擅自改写。
候选 blocked 中包含已完成训练、等待冻结后评的实验，不能理解为全部未训练。
done_complete 是训练终态，不自动证明 checkpoint 完整或科学有效。

早期路线覆盖 B0-B5 基线/indicator、C1/C2 排序损失、D1/F4/G1/H5、
W2 point-mass competition，以及 Y 系列的 mask、token、query、geometry、
归一化、残差和交互读出结构。逐项假设、YAML、负结果和 next_action 均在
完整明细及原 `pipeline/research_work.json`，不根据编号推断未执行实验。

## 4. 正式指标与待办

Y36 原始完整后评 `pipeline/analyses/AdaRES-Y36_frozen_review_1fb792ef.md` 的
FIRST：Y36 0.510716624292、W2 0.510009849923、B1 0.502112417447。
后续各实验有各自的同设备对照和重复预测，其数值不完全相同；
不能混用不同设备、不同 pass 的 baseline，不能挑更有利的重复作为 FIRST。

| 实验 | 当前关键结果 | 接手约束 |
|---|---|---|
| Y90 | FIRST 0.5038137203772663，子组伤害，拒绝具体实现 | 完整预测/成本/后审已完成，原有受争用成本尝试仍无效；不重跑 |
| Y92 | FIRST 0.5093426233066561，子组伤害 | 不重开、不自动晋升 |
| Y94 | FIRST 0.5075611062753286，子组伤害 | 不重开、不自动晋升 |
| Y96 | FIRST 0.5082609879270078，完整后评不确定 | 不重放已完成预测/成本 |
| Y99 | FIRST 0.5109262480516882，完整后评不确定 | 点估计略高不等于证实改进 |
| Y101 | FIRST 0.5062989703031004，完整后评不确定 | 不晋升、不重跑 |
| Y100 | fixed30，last30/best23，训练 best 0.5164485573768616 | 等原 server-2 GPU0/280W，在 Y102 后；不换卡代评 |
| Y88 | fixed30，last30/best21，8 次原生预测均已保存 | 完整正式比较/成本/后审待办；成本从未启动；server-1 暂停 |
| Y102 | 原 PID4747 存活，Train1/30 step7/4686、日志1892字节 | 停滞，不是健康运行；无已确认原生保存目录，不猜测 checkpoint |
| Y103 | 代码与 CPU 准备存在，尚无 native 检查/准入/训练 | phase gate 与 server-1 用户暂停仍生效 |
| Y57 | 用户单独暂停，原 epoch27 模型/优化器/调度器须保留 | 必须单独明确授权恢复，不因资源授权自动恢复 |
| Y86/Y97 | 原生/CPU 数值实现失败及原修订失败保留 | 不放松 guard/tolerance，不改名重跑，不等于机制证伪 |

Y88 已完成的原生预测不能再次执行旧的 remaining-baseline launcher。
其归档入口：
`pipeline/analyses/y88_user_pause_ce356360782a431ab9adc3e015f7d5aa/user_pause_archive.json`，
SHA256 `8c1ed94b12c62def5b153d74bcc45035d8003cf15dcd2f5785f325c5df43d37f`。

Y102 最近本次观察 UTC 2026-11-17T10:54:32.366082+00:00：
原解释器、命令行和源 marker 仍对应原训练；log SHA256
`5698413aeb13e0a24db416fc21bca124f4255c4c5ca7496879e58bd30174d339`。
之前 ptrace/stack 附加被拒绝，管理员帮助请求已存在；
不要重复附加、假设 sudo 或据此重启。后续诊断和迁移原进程均须单独处理。

## 5. 最新资源权限

按最新用户指令和以下文件解释资源，不能只看冻结 goal 中的历史资源句子：

- `AGENTS.md`、`SERVERS.yaml`。
- `pipeline/authorizations/server1_pause_20261115_ce356360782a431ab9adc3e015f7d5aa.json`。
- `pipeline/authorizations/server2_gpu1_pause_20261104.json`。

当前仅 server-2 GPU0/280W 可用，一卡一个独立实验。server-1 全部 GPU
训练、派发、native 检查、冻结评估与成本暂停，保留其350W原硬件设置。
server-2 GPU1 单独暂停；Y57 又是独立的实验暂停。**禁止 server-6**。
原绑定 server-1 或 GPU1 的冻结后评/成本不能迁到其他设备绕过暂停。
本次整理不改变功耗、硬件、原进程、队列、授权、owner 或调度频率。

已有 owner 为 `01a07e32-78bf-7670-925b-370911ea48db`。
旧 `Pipeline-AdaRES_project` 后台调度按既有规则保持禁用，不另建并行 scheduler。
新服务器需要用户明确给出服务器/GPU、数据路径与授权，再另行登记运行绑定。

科学目标文件不可重写，原 SHA256：
`4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a`。
server-1 暂停文件 SHA256：
`856914c2f463b590eb615c4de9cc7168b2328087462fa56756c55523970be3e0`。
server-2 GPU1 暂停文件 SHA256：
`3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45`。

## 6. 原部署、环境和构建

| 项目 | server-2 原始路径 | server-1 历史路径（暂停） |
|---|---|---|
| hostname | dl01-MS-7C60 | dl02-MS-7B73 |
| 代码 | /bak_home/songcl/Project/AdaRES_project/AdaRES | /data1/songcl/Project/AdaRES_project/AdaRES |
| Python | /home/songcl/.conda/envs/adares_project/bin/python | /data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python |
| 数据 | /bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/ | /data1/songcl/Project/AdaRES_project/data/ |
| checkpoint | /bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints | /data1/songcl/Project/AdaRES_project/AdaRES_checkpoints |
| PointNet egg | /home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg | /data1/songcl/Project/AdaRES_project/runtime/pointnet2-0.0.0-py3.8-linux-x86_64.egg |
| 私有 CUDA lib | 原 conda/runtime 绑定见原收据 | /data1/songcl/Project/AdaRES_project/runtime/cuda113lib64 |

server-2 当前只读观察：Linux x86_64、Python3.8.13、PyTorch1.12.1+cu113、
CUDA toolkit11.3（nvcc11.3.58）、driver550.90.07、RTX3090。
主要包观察值：torchvision0.13.1+cu113、spconv-cu1132.3.6、
cumm-cu1130.4.11、torch-scatter2.0.9、gorilla-core0.2.7.8、
transformers4.31.0、numpy1.24.4、scipy1.10.1、dgl-cu1110.6.1、
tensorboardX2.6.2.2。完整观察值见 `runtime_packages_observed.txt`。

这不是经验证的新机器安装 lock：旧 requirements 含未固定版本和旧 CUDA 依赖，
pip 元数据可能有重复项，PointNet 来自独立 egg，原环境还有继承的 metadata warning。
优先按原环境做私有克隆/打包，在相容 Linux/CUDA 上验证；直接安装全部最新版不可取。

源码构建位置：

- `AdaRES/ipdn/lib/setup.py`：PointGroup 扩展；包含 CUDA 检测，不是 CPU-only 验证步骤。
- `AdaRES/pointnet2/setup.py`：PointNet2 扩展，`_ext_src` 含 C++/CUDA 源。
- `AdaRES/setup.py`：仅简要包元数据，不能单独视为完整依赖安装脚本。
- `AdaRES/ipdn/torch/nn_new`：定制 attention，不可随意替换为新版 stock torch。

原训练使用 `python -B -S` 与显式 `PYTHONPATH`：
代码目录、`ipdn/lib`、该项目 site-packages、PointNet egg。
server-1 另有私有 `LD_LIBRARY_PATH`。新环境应验证实际 import 来源，
避免用户 site-packages 或其他项目库串入。不要用本地 Windows 解析测试冒充 CUDA/native 通过。

## 7. GitHub 之外必须迁移的文件

**Git clone 不是完整可运行迁移。** 下列大文件保持原处，未上传为代码：

1. 精确原数据目录，尤其 `ScanRefer/ScanRefer_train_new.json`、
   `ScanRefer/ScanRefer_val_new.json`、`scanrefer/scene_graphs_new_train.json`、
   `scene_graphs_new_val.json`、`scannetv2/{train,val}/*_refer.pth`、
   `clip_feat/*.pth` 和 `scannetv2-labels.combined.tsv`。Linux 大小写有意义。
   沿用原预处理、划分、排列与图缓存，不触发缺文件后的重新生成。
2. 原 `sp_unet_backbone.pth`。不要用发布的 b005/CFNet 完整 checkpoint
   冒充 AdaRES-B0，也不要替换训练初始化来源。
3. `AdaRES/backbones` 的原 symlink 目标：
   `/bak_home/songcl/3D_visual_grounding/CFNet/backbones`；
   至少包含完整 `roberta-base` 权重、config、tokenizer/vocab/merges。
   dataset 明确使用 `local_files_only=True`，不能靠联网下载最新版替代。
4. server-1 和 server-2 各自原 `AdaRES_checkpoints`。保留 best、last、
   optimizer、scheduler、epoch、原 source/YAML、日志与 SHA，不能只拿模型权重。
5. 本地 `D:/project/AdaRES_project/Checkpoints`（约1.72GiB日志，精确值见发布清单）
   和未入 Git 的大型原始预测、训练诊断 capture、二进制缓存。
   发布清单逐项记录排除原因、字节数和 SHA256。
   两份 `pipeline/analyses/y56_cached_train_*_program.py` 是嵌入大型训练缓存的
   自包含诊断产物，也保持原处并列入单独迁移，不作为普通源码上传。
6. 原私有环境及必要扩展二进制可另行打包迁移；Git 保存了能取得的源码，
   不保证旧 `.so` 在不同 CUDA/ABI/硬件上可直接运行。

原生 loader 还可能在缺失 scene graph 时写回生成结果；新机缺依赖/数据时应先停下，
不要让自动回退改变原实验数据协议。

### 关键 checkpoint

Y100 原目录：
`/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y100_appearance_input_skip_res/20261111_085241/`

- `last.pth`：epoch30，optimizer544、scheduler30，
  SHA `6a203502e9b3b8d439438c0acd37df7d30c37025019a17c106982282e58f441c`。
- `best.pth`：epoch23，optimizer544、scheduler23，
  SHA `bc9a12627ab9d5a90056d91c675ef14cbe51f3f5fd7a9c1062365550a6feea00`。

Y88 原目录：
`/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y88_mask_query_quadratic_res/20261106_053553/`

- `last.pth`：epoch30，
  SHA `4f1a1d9e77d41027dc8b51613b8ee0dac4f8761401c9847f27aa3d2b1cec74b7`。
- `best.pth`：epoch21，
  SHA `6d1a7693807d5e0499571b615f0615d6d867d9f4edb8b19a94738e99f7f8e577`。

Y57 受保护的暂停保存路径：
`/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y57_peer_second_moment_res/20261021_004739/last.pth`。
以 `pipeline/checks/y57_user_pause_befeacf640bf499a9ff39fa6c9df712a.json`
中的暂停时 epoch27 证据为准；旧 resume 字段含更早 epoch/hash，不能误当当前恢复依据。

Y102 没有经核验的原生日志保存公告/终态 checkpoint 绑定，**不得按时间猜目录**。
其余逐实验路径见统计 CSV/注册表、post_training_freeze 与原终态证据。
本次未重新加载任何 checkpoint；上述完整性是原已有记录，不是新机器验收结果。

## 8. 新服务器接手顺序

1. 克隆私有仓库，确认 `main` 与发布收据的提交一致，保留全部 `history/*`。
   先读本文件、`AGENTS.md`、科学 goal 和各自 pause authorization。
2. 运行不导入训练库的源文件校验：
   `python -B scripts/handoff_20261117/validate_clone.py`。
   它只检查发布哈希/语法/原 goal，不启动 GPU 或旧服务器连接。
   两份已知历史诊断脚本的原始语法错误仍会令完整 AST 项返回非零；
   先确认哈希全匹配、错误仅是下文列明的旧文件，不为消除报告而改写旧证据。
3. 用户明确新服务器/GPU权限、迁移原运行还是开展新实验，以及暂停项是否仍保持暂停。
   本次上传本身不替代这些授权。
4. 单独复制原数据、RoBERTa、SPFormer、checkpoint、原预测及日志；
   对照原证据校验源/目的 SHA256、文件数量、划分、优化器和调度器状态。
   活动写入中的 Y102 不做无一致性保证的“完整 checkpoint 备份”。
5. 重建私有 runtime，核验 Python/CUDA/扩展 ABI 和 import 来源。
   在明确获准的空闲 GPU 上才做最小原生 YAML 数据加载、有限 loss/backward，
   并保留新环境实际日志，不沿用旧 native PASS 冒充新环境通过。
6. 新路径仅做明确登记的确定性映射，原 YAML/source/收据保留不改；
   不修改固定30轮、seed1999、数据、损失、selector、阈值来迁就环境。
7. 新 owner/调度/资源登记需单独授权和校验，不把旧 running.json 直接当作新机进程。
   旧 sync.sh 依赖 cygpath、原源 marker 和增量祖先关系，不适合空目录新机初始化。
8. 原 Y100/Y88 后评继续遵守原设备绑定；完整评测、成本已完成的 ID 不重放。
   先解决 Y102 真实停滞与数据/权重迁移，再决定后续科学工作。

## 9. 冻结科学协议

后期完整后评：candidate 与 Y36/W2/B1，每方法两次原生预测，每次9508表达式、
141场景，FIRST为主；whole-scene bootstrap10000、seed1999，
TRAIN raw-point cuts1487/3325，按原预注册子组伤害条件执行。
重复预测反映观察到的推断噪声，不是多训练 seed 的方差或普适界。

原成本比较：冻结 TRAIN8、largest-pair indices728/6780，串行 warm2/read3，
以及单独 native batch2、共20次含warmup的 fresh-optimizer memory-only stress。
不覆盖原权重；有限场景、串行顺序和 descriptive p95 不能证明全数据显存界或 Pareto。

保留全部负结果和不确定结果，不调验证阈值、温度、margin、loss weight、
feature subset 或 epoch，不自动重开 rejected/banned ID，不把同机制改名当新证据。

## 10. 验收与已知缺口

- 代码上传完成与远端提交身份，以本次发布收据和实际 GitHub ref 校验为准。
- 发布清单区分本地源码、server-2 补齐源码、冲突副本、分支独有源码、控制工具副本。
- Python AST/hash 检查不等于新环境 import、数据可读或 CUDA 数值检查通过。
- 完整 AST 检查发现两份原已有诊断脚本语法错误，按原字节保留：
  `pipeline/checks/terminal_readiness_adares-y69_e5ca4e47b09f466ea2a1dc215c0acf3c.py`
  第6行字符串未闭合；
  `scripts/probe_y49_train24_capture_1cb1.py` 第49行括号不匹配。
  不将它们作为迁移入口，不修改旧证据来伪造全量语法 PASS。
  模型、原生训练入口、补齐的共享源码未在本次 AST 检查中报告语法错误。
- 当前没有新服务器地址、资源授权，也没有实际迁移数据/权重或启动新训练。
- 原服务器数据总大小、全部远端 checkpoint 文件清单与新环境安装可运行性，
  不在本次源码只读备份中宣称已验收。
- 机密凭据不进仓库；SSH优先密钥，不携带密码文件，不推断 sudo 权限。

本次交接工具只在项目 `docs/`、`scripts/`、`archives/handoff_20261117/` 及独立 Git
发布索引/引用中工作，不改研究台账、科学 goal、原进程、原 checkpoint 或资源设置。
