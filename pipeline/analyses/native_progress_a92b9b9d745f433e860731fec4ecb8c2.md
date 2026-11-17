# Native Training Observation

- Project: AdaRES_project; token: a92b9b9d745f433e860731fec4ecb8c2.
- Preflight snapshot: 2026-10-01T02:49:26.996266+00:00.
- Local observation completed: 2026-10-01T10:49:53.0357586+08:00.
- Read-only SSH BatchMode to authorized server-2 checked GPU occupancy, original process arguments and bounded 1000-byte tails from both original logs. No remote mutation.

| Candidate | GPU | PID | Native steps within tail | Memory MiB | Utilization | Preflight early best mIoU | Preflight no-improve |
|---|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[14/30]2416 -> 2419/4686 | 16730 | 86% | 0.2054 | 0/30 |
| AdaRES-Y17 | 1 | 109632 | Train[17/30]2751 -> 2754/4686 | 22784 | 79% | 0.4923 | 1/30 |

- Both original processes use /home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py with their original configs, --num_gpus 1 and respective --gpu_ids 0/1.
- Y15 config: configs_song/model_paper5/adares_y15_query_completion_res.yaml
- Y17 config: configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml
- Y15 log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log
- Y17 log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y17_target_equivalent_pairing_res_20260930_230343.log
- Both original single-GPU processes remain active and native steps advance. No crash or OOM observed in bounded samples; not an exhaustive fault audit.
- Y17 latest preflight metric 0.4910 is below early best 0.4923. Preserve this and historical unfavorable observations without validation-driven protocol changes or premature terminal judgments.
- Acc@25, Acc@50, spmIoU, checkpoint contents and final 30-epoch efficacy were not examined and remain unknown. Occupancy is not isolated model peak memory.
- Capacity 2/active 2/running 2/waiting 0/admission budget 0; due research 0/new terminal events 0. Event mode reconciles prior artifact changes. Scientific stages remain training.
- No new hypothesis, scientific test, admission, dispatch, restart, model/configuration change, validation tuning or scheduler change.
- Completed champion remains W2 mIoU 0.5100 versus target 0.515. Early metrics and execution progress do not prove final efficacy, significance or goal attainment.
- Preserve original runs through 30 epochs. At actual terminal results apply frozen full-ScanRefer evaluation, scene-clustered uncertainty, noise/material-harm and Pareto review. Exact regrouped pair-off remains needed for Y17 causal attribution.
