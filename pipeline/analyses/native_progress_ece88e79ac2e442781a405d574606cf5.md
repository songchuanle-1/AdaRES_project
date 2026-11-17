# Native Training Observation

- Project: AdaRES_project; token: ece88e79ac2e442781a405d574606cf5.
- Preflight snapshot: 2026-10-01T03:49:57.043422+00:00.
- Read-only observation completed locally: 2026-10-01T11:52:15.0878363+08:00.
- Authorized server-2 SSH BatchMode probe: GPU occupancy, original process arguments and bounded 1000-byte tails of each original log. No remote mutation.

| Candidate | GPU | PID | Native steps within tail | Memory MiB | Utilization | Preflight best mIoU | Preflight no-improve |
|---|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[15/30]1999 -> 2002/4686 | 17144 | 85% | 0.2079 | 0/30 |
| AdaRES-Y17 | 1 | 109632 | Train[19/30]102 -> 106/4686 | 22784 | 63% | 0.4939 | 0/30 |

- Original processes use /home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py with --num_gpus 1 and respective --gpu_ids 0/1.
- Y15 config: configs_song/model_paper5/adares_y15_query_completion_res.yaml
- Y17 config: configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml
- Y15 log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log
- Y17 log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y17_target_equivalent_pairing_res_20260930_230343.log
- Both original single-GPU processes remain active and native steps advance. Y17 preflight showed epoch 18, whereas the later native tail shows epoch 19; observations have different timestamps and are not conflated.
- No crash or OOM observed in these bounded samples; not an exhaustive fault audit. Occupancy is not standalone model peak memory.
- Acc@25, Acc@50, spmIoU, checkpoint contents and final 30-epoch efficacy were not examined and remain unknown. Early best values do not establish final efficacy, significance or goal attainment; historical unfavorable observations remain preserved.
- Capacity 2/active 2/running 2/waiting 0/admission budget 0; due research 0/new terminal events 0. Event mode reconciles prior artifact changes. Scientific stages remain training.
- No new hypothesis, scientific test, admission, dispatch, restart, model/configuration change, validation tuning or scheduler change.
- Completed champion remains W2 mIoU 0.5100 against target 0.515.
- Preserve original runs through 30 epochs. On actual terminal results apply frozen full-ScanRefer evaluation, scene-clustered uncertainty, noise/material-harm and Pareto review. Exact regrouped pair-off remains needed for Y17 causal attribution.
