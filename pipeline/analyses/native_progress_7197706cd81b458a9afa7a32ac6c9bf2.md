# Native Training Progress - AdaRES-Y15 / AdaRES-Y18

- Tick token: 7197706cd81b458a9afa7a32ac6c9bf2.
- Preflight snapshot: 2026-10-01T18:48:01.586603Z.
- Local read-only SSH window: 2026-10-02T02:48:28.1960294+08:00 to 2026-10-02T02:48:28.4497252+08:00.
- BatchMode SSH exited 0; hostname dl01-MS-7C60 matched server-2.
- Probe: authorized GPU memory/utilization, original PID commands, 900-byte original-log tails. No launch, restart, checkpoint modification or GPU diagnostic.

| Candidate | GPU | Original PID | Device memory MiB | GPU utilization % | Native observed steps | Learning rate |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | 17282 | 75 | Train[28/30]3714 -> 3715 -> 3716 -> 3717/4686 | 0.000014 |
| AdaRES-Y18 | 1 | 7449 | 22718 | 81 | Train[5/30]1298 -> 1299 -> 1300 -> 1301/4686 | 0.000079 |

Original process commands:

```text
/home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y15_query_completion_res.yaml --num_gpus 1 --gpu_ids 0
/home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y18_point_measure_res.yaml --num_gpus 1 --gpu_ids 1
```

Original unique logs:

```text
/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log
/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y18_point_measure_res_20261001_233715.log
```

Begin monitor: Y15 best/latest0.2303, epoch28/30, no_improve0/30,
recent0.2238/0.2108/0.2116/0.2249/0.2303.
Y18 best0.4419, latest0.4372, epoch5/30, no_improve1/30,
recent0.3222/0.3864/0.4419/0.4372.
These intermediate validation readouts are not final trained-efficacy estimates.
The previous observation was Y15 Train[27/30]4299-4302 and Y18
Train[3/30]4536-4539; both original PIDs continue into later epochs.

No crash marker appeared in the bounded tails; this is not a full-log health
audit or proof of future training success. Device memory is not isolated
candidate peak memory. Preserve original fixed30epoch/seed1999, adverse
controls and frozen terminal evaluation; do not tune on intermediate validation.
Both stable candidates remain training. Final efficacy, subgroup harm,
goal attainment and accuracy/cost frontier remain unknown.
Active2/target2, budget0, due0: no duplicate admission or repeated closed probe.
Use local observation time because remote clock consistency is unresolved.
Execution closure is pending finish/postflight.
