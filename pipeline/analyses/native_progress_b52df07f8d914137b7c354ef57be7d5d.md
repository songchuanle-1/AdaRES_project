# Native Training Progress - AdaRES-Y15 / AdaRES-Y18

- Tick token: b52df07f8d914137b7c354ef57be7d5d.
- Preflight snapshot: 2026-10-01T19:49:31.743848Z.
- Local read-only SSH window: 2026-10-02T03:49:56.8099442+08:00 to 2026-10-02T03:49:57.0767488+08:00.
- BatchMode SSH exited 0; hostname dl01-MS-7C60 matched server-2.
- Probe: authorized GPU memory/utilization, original PID commands and 900-byte original-log tails. No launch, restart, checkpoint modification or GPU diagnostic.

| Candidate | GPU | Original PID | Device memory MiB | GPU utilization % | Native observed steps | Learning rate |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | 17282 | 80 | Train[29/30]3266 -> 3267 -> 3268 -> 3269/4686 | 0.000013 |
| AdaRES-Y18 | 1 | 7449 | 22718 | 53 | Train[6/30]3092 -> 3093 -> 3094 -> 3095/4686 | 0.000074 |

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

Begin monitor: Y15 best0.2303/latest0.2282, epoch29/30, no_improve1/30,
recent0.2108/0.2116/0.2249/0.2303/0.2282.
Y18 best/latest0.4617, epoch6/30, no_improve0/30,
recent0.3222/0.3864/0.4419/0.4372/0.4617.
Intermediate validation readouts do not establish final trained efficacy.
Both original PIDs remained present and native steps increased. Compared with
the previous observation, Y15 advanced from epoch28 and Y18 from epoch5.

No crash marker appeared in the bounded tails; this is not a full-log audit
or proof of future success. Device memory is not isolated candidate peak memory.
Both stable IDs remain training. Final efficacy, subgroup harm, goal attainment
and accuracy/cost frontier remain unknown. Active2/target2, budget0, due0:
preserve fixed30epoch/seed1999 and frozen terminal evaluation, without duplicate
admissions, validation tuning or repeated closed diagnostics. Use local
observation time because remote clock consistency is unresolved.
Execution completion is pending finish/postflight.
