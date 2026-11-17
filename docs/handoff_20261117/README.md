# AdaRES Project

AdaRES: 3D referring expression segmentation on ScanRefer, using a released
SPFormer sparse-UNet backbone and project-specific query/decoder/confidence
variants.

这是 **2026-11-17 的代码与研究交接快照**，不是新的训练授权，也不是已经迁移成功的声明。
本仓库为私有备份；原始数据、预训练权重、训练 checkpoint 和大型原始预测需单独迁移。

## 交接入口

- [迁移交接文档](MIGRATION_HANDOFF.md)：当前结论、暂停约束、环境、数据、权重、接手顺序。
- [实验统计](EXPERIMENTS.md)：85 个有训练记录的 ID 与全部候选的分类。
- [实验明细 CSV](experiment_inventory.csv)：逐实验配置、训练状态、指标、正式结论、路径和下一步。
- [实验明细 JSON](experiment_inventory.json)：可程序读取的同一份完整统计。
- [原服务器只读清单](server2_readonly_inventory.json)：实际运行环境、源码校验和 Y102 阻塞状态。
- [依赖版本观察值](runtime_packages_observed.txt)：用于重建环境的参考，不是可无条件安装的 lockfile。
- [发布清单](publication_manifest.json)：代码来源、SHA256、排除的产物与冲突处理。
- [快照校验](snapshot_verification.json)：文件哈希与 Python 语法检查的实际结果。
- [第三方源码说明](THIRD_PARTY_NOTES.md)：原有依赖和许可证边界。

## 当前研究状态

有训练记录的稳定 ID 共 **85** 个：完整结束 81、早停 1、低指标关闭 1、
用户暂停 1、登记运行中 1。155 条候选阶段记录不是 155 次训练。
Y100 的训练 best mIoU 为 **0.5164485573768616**，但正式冻结后评尚未完成，
不能据此宣布科学目标达成。

Y102 原进程 PID4747 仍存在，2026-11-17 的只读快照仍见 Train1/30、
step7/4686；不能把进程存活当成训练健康。server-1、server-2 GPU1、Y57
分别处于用户暂停状态，禁止自行恢复、迁移或换卡绕过暂停。

## 仓库结构

```text
AdaRES/                       模型、原生 YAML、训练/评测入口及补齐的共享运行源码
scripts/                      原实验检查、归档、部署、恢复及交接工具
pipeline/                     原始目标、实验注册、候选、收据和小型证据
docs/handoff_20261117/         本次交接文档、统计和校验清单
vendor/server2_source/         与本地版本不同的原服务器源码，按原字节保留
vendor/songcl_research_pipeline/  原控制工具的归档副本，不自动启用
```

GitHub `main` 是交接用工作树快照。`history/*` 分支保留原始本地分支的提交身份；
当前生产源提交为 `0c6c844719c8466ddd8888d054fe3b7ee658e643`，不要把交接提交号
回填到旧准入收据或原运行记录中。

新机器克隆后先阅读交接文档。不要直接启动旧 `remote_train.sh`、`sync.sh`、
`admit_dispatch_*` 或后台定时任务；其中保留了历史服务器和恢复流程。
