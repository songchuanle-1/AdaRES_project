# Native Training Progress - AdaRES-Y15 / AdaRES-Y18

- Tick token: e15d9042c7e04912a5d44c307a66eba5.
- Preflight snapshot: 2026-10-01T15:49:34.482690Z.
- Local observation timestamp: 2026-10-01T23:50:37.0962355+08:00.
- Read-only SSH probe exited 0; hostname dl01-MS-7C60 matched server-2.
- Scope: hostname, authorized GPU memory/utilization, original PIDs and bounded 750-byte native log tails. No new training or diagnostic launched.

| Candidate | GPU | Original PID | GPU memory MiB | GPU utilization % | Native observed steps | Learning rate |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | 17282 | 82 | Train[26/30]755 -> 756 -> 757 -> 758/4686 | 0.000017 |
| AdaRES-Y18 | 1 | 7449 | 17138 | 89 | Train[1/30]1427 -> 1428 -> 1429 -> 1430/4686 | 0.0001 |

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

The begin monitor observed Y15 best mIoU 0.2238, latest 0.2116, epoch26/30,
no_improve2/30; recent values 0.1777/0.2021/0.2238/0.2108/0.2116.
Y18 had no validation result yet. Both original PIDs remained present, and
native steps increased. No crash marker was seen in the bounded tails;
this is not a full-log health audit or a claim that future training succeeds.

Scientific disposition: both candidates remain training. Trained efficacy,
goal attainment and accuracy/cost frontier remain unknown. Continue the
original fixed30epoch/seed1999 protocol and frozen terminal evaluation plans.
No restart, validation tuning, gate change or duplicate admission is justified.
Admission capacity is full (2/2), candidate budget0, and due diagnostics0.
Use local observation time because remote clock consistency is unresolved.
Execution closure is pending finish, not established by this observation.
