# Native Training Progress - AdaRES-Y15 / AdaRES-Y18

- Tick token: e9b65ffe2a2844fbbe6064e74518af79.
- Preflight snapshot: 2026-10-01T20:48:02.623783Z.
- Local read-only SSH window: 2026-10-02T04:48:31.6156609+08:00 to 2026-10-02T04:48:31.8685035+08:00.
- BatchMode SSH exited 0; hostname dl01-MS-7C60 matched server-2.
- Scope: authorized GPU memory/utilization, original PID commands and 900-byte original-log tails. No launch, restart, checkpoint modification or GPU diagnostic.

| Candidate | GPU | Original PID | Device memory MiB | GPU utilization % | Native observed steps | Learning rate |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | 17282 | 43 | Train[30/30]2571 -> 2572 -> 2573 -> 2574/4686 | 0.000012 |
| AdaRES-Y18 | 1 | 7449 | 22718 | 82 | Train[7/30]4607 -> 4608 -> 4609 -> 4610/4686 | 0.00007 |

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

Begin monitor: Y15 best0.2303/latest0.1977, epoch30/30, no_improve2/30,
recent0.2116/0.2249/0.2303/0.2282/0.1977.
Y18 best/latest0.4678, epoch7/30, no_improve0/30,
recent0.3864/0.4419/0.4372/0.4617/0.4678.
These are intermediate monitor readouts, not terminal efficacy estimates.
Y15's epoch30 label does not prove completion: its original process remains
alive with only2574/4686 training steps observed. Both original PIDs and native
increasing steps were verified, advancing from the prior epoch29/epoch6.

No crash marker appeared in the bounded tails; this is not a full-log health
audit or proof of future successful training. Byte-cut progress-bar characters
are not failure evidence. Device usage is not isolated candidate peak memory.
Both stable IDs remain training. Final efficacy, material subgroup harm, goal
attainment and accuracy/cost frontier remain unknown. Preserve fixed30epoch/
seed1999 and frozen terminal evaluation without validation tuning or restart.
Active2/target2, budget0, due0: no duplicate admission or repeated closed probe.
Use local observation time because remote clock consistency is unresolved.
Execution closure is pending finish/postflight.
