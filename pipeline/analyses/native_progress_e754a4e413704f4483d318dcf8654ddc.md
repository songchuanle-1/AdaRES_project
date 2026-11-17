# Native Training Observation

- Tick token: e754a4e413704f4483d318dcf8654ddc.
- Preflight snapshot: 2026-10-01T13:51:15.980811+00:00.
- Local observation: 2026-10-01T21:52:16.0994718+08:00.
- Authorized SSH host: songcl@192.168.1.163; observed hostname dl01-MS-7C60.
- Read-only bounded command: hostname, nvidia-smi query, ps for original PIDs 109632/18796, tail -c 1000 of original Y15 log. Exit 0.

| GPU | Memory MiB | Util percent | Original training PID |
|---|---|---|---|
| 0 | 17282 | 47 | 18796 present |
| 1 | 1 | 0 | 109632 absent |

Y15 PID18796 command:
`/home/songcl/.conda/envs/adares_project/bin/python -S tools/train.py configs_song/model_paper5/adares_y15_query_completion_res.yaml --num_gpus 1 --gpu_ids 0`

Original log:
`/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log`

Native log tail shows Train[24/30] at 1375, 1376, 1377, 1378, 1379 of 4686; learning rate 2e-05. Increasing steps in the original process confirm continued training without a restart. No crash marker in this bounded tail; this is not a full-log integrity claim.

Canonical preflight reports Y15 best mIoU 0.2238, epoch 24/30, no_improve 0/30; recent sequence 0.2002, 0.2077, 0.1777, 0.2021, 0.2238. Final trained efficacy remains unknown.

Y17 deathwatch advanced to 2/3 in the canonical preflight. Original PID absence is consistent with the preceding tick's saved final validation and epoch30 checkpoint evidence. No new Y17 evaluation was executed here; original exit code remains unknown. Preserve queue disposition until the canonical monitor reconciles it, without duplicate preflight or manual capacity mutation.

Mode event; active queue count 2; waiting 0; due candidates 0; admission budget 0. Actual original training process count observed here is 1. GPU1 being idle does not itself authorize a new admission.

Disposition: Y15 stays training under the frozen 30-epoch protocol; review its final checkpoint using the already frozen evaluation plan when terminal evidence becomes available. Y17 stays complete in research work, with efficacy inconclusive and no champion promotion. No new experiment, gate relaxation, scheduling change, or remote mutation.
