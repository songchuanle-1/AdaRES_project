# Native Training Progress - AdaRES-Y15 / AdaRES-Y18

- Tick token: 25682ce427bd4919a3cfb6b7b92e48f5.
- Preflight snapshot: 2026-10-01T16:49:45.278139Z.
- Local read-only observation window: 2026-10-02T00:50:12.8587086+08:00 to 2026-10-02T00:50:13.0991150+08:00.
- SSH BatchMode probe exited 0; server-2 hostname dl01-MS-7C60 matched.
- Probe covered authorized GPU memory/utilization, original PIDs/commands and 900-byte original-log tails. No launch, restart, checkpoint modification or GPU diagnostic.

| Candidate | GPU | Original PID | GPU memory MiB | GPU utilization % | Native observed steps | Learning rate |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | 17282 | 51 | Train[27/30]150 -> 151 -> 152 -> 153/4686 | 0.000016 |
| AdaRES-Y18 | 1 | 7449 | 22714 | 78 | Train[2/30]3067 -> 3068 -> 3069 -> 3070/4686 | 0.000094 |

Both original processes remained present:

```text
/home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y15_query_completion_res.yaml --num_gpus 1 --gpu_ids 0
/home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y18_point_measure_res.yaml --num_gpus 1 --gpu_ids 1
```

Native progress was observed in the same unique logs:

```text
/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log
/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y18_point_measure_res_20261001_233715.log
```

The begin monitor observed Y15 best/latest mIoU0.2249, epoch27/30,
no_improve0/30; recent values0.2021/0.2238/0.2108/0.2116/0.2249.
Y18 best mIoU0.3222, epoch2/30, no_improve0/30. These are intermediate
native monitor readouts, not matched trained-efficacy estimates.
The previous tick observed Y15 epoch26 steps755-758 and Y18 epoch1
steps1427-1430; both have advanced without changing their original PIDs.

No crash marker appeared in the bounded tails. This does not certify full-log
health, absence of past faults or future successful training. GPU1's observed
22714MiB total device usage is not an isolated candidate peak-memory profile;
do not infer a resource-gate result from this device-level snapshot.

Scientific disposition remains training for both stable IDs. Final efficacy,
material subgroup harm and accuracy/cost frontier remain unknown. Preserve
fixed30epoch/seed1999 and frozen terminal evaluation; do not tune on these
validation values. Active2/target2, budget0, due0: no new admission or repeated
closed diagnostic. Use the local observation window, not the inconsistent
remote clock. Execution completion remains pending finish/postflight.
