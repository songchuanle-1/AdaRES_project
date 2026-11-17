# Native Training Observation

- Project: AdaRES_project; tick token: e8113645dd2e4a9cbdc1ff17177ecf54.
- Preflight snapshot: 2026-09-30T21:48:42.061713+00:00.
- Local read-only observation completed: 2026-10-01T05:49:04.1740549+08:00.
- SSH BatchMode to authorized server-2 checked GPU occupancy, original process arguments and bounded 1000-byte tails of both original logs. No remote mutation.

| Candidate | GPU | PID | Native steps within tail | Memory MiB | Utilization | Preflight early best mIoU | Preflight no-improve |
|---|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[10/30]379 -> 383/4686 | 16728 | 83% | 0.1734 | 3/30 |
| AdaRES-Y17 | 1 | 109632 | Train[10/30]2483 -> 2486/4686 | 22784 | 38% | 0.4615 | 1/30 |

- Y15 original command: /home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y15_query_completion_res.yaml --num_gpus 1 --gpu_ids 0
- Y17 original command: /home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml --num_gpus 1 --gpu_ids 1
- Y15 log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log
- Y17 log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y17_target_equivalent_pairing_res_20260930_230343.log
- Both original processes remain active and native training steps advance. No crash or OOM observed in these bounded samples; not an exhaustive fault audit.
- Latest preflight metrics: Y15 0.1599, Y17 0.4566. Preserve both unfavorable observations. Early validation fluctuations and Y15's low metrics are not final efficacy, nor grounds to tune or terminate the fixed 30-epoch protocol.
- Acc@25, Acc@50, spmIoU, checkpoint contents and full 30-epoch efficacy were not examined this tick and remain unknown. Instantaneous occupancy is not standalone model peak memory.
- Preflight capacity 2/active 2/running 2/waiting 0/admission budget 0; due research 0/new terminal events 0. Event mode reconciles prior artifact changes, not a new scientific admission.
- Scientific stages remain training. No new hypothesis, scientific test, admission, dispatch, restart, model/configuration change, scheduler change or validation tuning.
- Completed champion remains W2 mIoU 0.5100 versus target 0.515. This execution check does not establish scientific benefit or target attainment.
- Preserve original single-GPU runs to 30 epochs. On actual terminal results apply frozen full-ScanRefer evaluation, scene-clustered uncertainty, measured-noise/material-harm and Pareto review. Exact regrouped pair-off remains needed for Y17 causal attribution.
