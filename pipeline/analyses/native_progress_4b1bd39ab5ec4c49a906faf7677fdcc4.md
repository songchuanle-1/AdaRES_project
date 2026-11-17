# Native Training Observation

- Project: AdaRES_project; tick token: 4b1bd39ab5ec4c49a906faf7677fdcc4.
- Preflight snapshot: 2026-09-30T19:49:56.502432+00:00.
- Local read-only observation completed: 2026-10-01T03:51:08.9997547+08:00.
- Authorized server-2 access used SSH BatchMode. GPU memory/utilization, original process arguments and bounded 1000-byte tails of the original logs were checked; no remote files or processes were modified.

| Candidate | GPU | PID | Native steps within tail | Memory MiB | Utilization | Preflight early best mIoU | Preflight no-improve |
|---|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[8/30]1670 -> 1673/4686 | 16728 | 56% | 0.1734 | 1/30 |
| AdaRES-Y17 | 1 | 109632 | Train[7/30]3640 -> 3643/4686 | 22782 | 28% | 0.4425 | 0/30 |

- Y15 original command: /home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y15_query_completion_res.yaml --num_gpus 1 --gpu_ids 0
- Y17 original command: /home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml --num_gpus 1 --gpu_ids 1
- Y15 original log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log
- Y17 original log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y17_target_equivalent_pairing_res_20260930_230343.log
- Both original single-GPU processes are still active and advancing. No crash or OOM was observed in these bounded samples; this is not an exhaustive fault audit.
- Y15 latest preflight metric 0.1683 is below its early best 0.1734; one non-improvement does not justify altering the frozen 30-epoch training protocol or making a terminal efficacy judgment.
- Acc@25, Acc@50, spmIoU, checkpoint contents and final 30-epoch efficacy were not examined this tick and remain unknown. Instantaneous GPU occupancy is not isolated model peak memory.
- Capacity 2/active 2/running 2/waiting 0/admission budget 0; due research 0/new terminal events 0. This event reconciles previous evidence changes, not a new design opportunity or scientific admission.
- Scientific stages remain training. No new hypothesis, admission, scientific test, dispatch, restart, configuration change, scheduler change or validation tuning was performed.
- Completed champion remains W2 mIoU 0.5100 versus target 0.515. Early metrics above are not final efficacy, statistical significance or goal attainment.
- Preserve both original runs to 30 epochs. On actual terminal results apply the frozen full-ScanRefer evaluation, scene-clustered uncertainty, measured-noise/material-harm and Pareto protocol. Exact regrouped pair-off remains needed for Y17 causal attribution.
