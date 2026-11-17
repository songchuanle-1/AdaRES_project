# Native Training Observation

- Project: AdaRES_project; token: 068d922d34a544018dfe1f826d8c3cd5.
- Preflight snapshot: 2026-09-30T23:48:23.143423+00:00.
- Local observation completed: 2026-10-01T07:50:02.2824869+08:00.
- Read-only SSH BatchMode to authorized server-2 checked instantaneous GPU occupancy, original process arguments and bounded 1000-byte tails from each original log. No remote mutation.

| Candidate | GPU | PID | Native steps within tail | Memory MiB | Utilization | Preflight early best mIoU | Preflight no-improve |
|---|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[11/30]4148 -> 4151/4686 | 16730 | 62% | 0.1890 | 0/30 |
| AdaRES-Y17 | 1 | 109632 | Train[13/30]1687 -> 1690/4686 | 22784 | 62% | 0.4810 | 0/30 |

- Y15 original command: /home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y15_query_completion_res.yaml --num_gpus 1 --gpu_ids 0
- Y17 original command: /home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml --num_gpus 1 --gpu_ids 1
- Y15 original log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log
- Y17 original log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y17_target_equivalent_pairing_res_20260930_230343.log
- Both original single-GPU processes remain active and native steps advance. Y15 returned from ordinary epoch-10 validation to training; no restart was performed. No crash or OOM observed in these bounded samples; not an exhaustive fault audit.
- Acc@25, Acc@50, spmIoU, checkpoint contents and final 30-epoch efficacy were not examined this tick and remain unknown. Occupancy is not isolated model peak memory.
- Capacity 2/active 2/running 2/waiting 0/admission budget 0; due research 0/new terminal events 0. Scientific stages remain training. No new hypothesis, scientific test, admission, dispatch, restart, model/configuration change, validation tuning or scheduler change.
- Early validation best values are not final efficacy, significance or target attainment. Preserve prior unfavorable observations. Completed champion remains W2 mIoU 0.5100 versus target 0.515.
- Preserve original single-GPU runs through 30 epochs. At actual terminal results apply frozen full-ScanRefer evaluation, scene-clustered uncertainty, noise/material-harm and Pareto review. Exact regrouped pair-off remains needed for Y17 causal attribution.
