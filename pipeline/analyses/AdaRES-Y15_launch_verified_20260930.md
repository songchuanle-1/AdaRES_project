# Y15 Verified Native Launch

Local verification2026-09-30 19:50 and20:04 Asia/Shanghai.
Canonical pipeline_monitor.py, server-2 only; no manual training bypass.

Scoped git archive CRLF repair commit2bb1c107b27fbfdbdf2d6ffd25b43bb5bf4f3e6c
preserved tested source bytes; remote source marker matches this commit.
All six admission hashes validated. Admission receipt SHA256
d709959e12f690f73723cbdf8ae2f71782bb7abbc14fffe14183094c16cbe7f9.

Native command: dedicated adares_project Python -S tools/train.py
configs_song/model_paper5/adares_y15_query_completion_res.yaml
--num_gpus1 --gpu_ids0. PID18796; children19111/19159/19207/19255 all have
PPID18796, DataLoader workers, not independent training duplicates.
Unique log:
/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log.

Observed Train[1/30] at102/4686, subsequently1084/4686 with lr0.0001.
Queue and experiment both launch_verified=true, training_progress.
GPU0 initially88%/16152MiB, later91%/17528MiB; GPU1 0%/1MiB.
No full epoch or validation metric yet. This proves live native progression,
not trained efficacy, teacher necessity or goal completion.
Exactly30epochs, task min_epochs30/patience30, unchanged global queue policy.
Remote wall clock differs from local by about31minutes even after timezone
conversion: use local observation times, native monotonic steps and log identity;
do not silently interpret remote date as synchronized local time.
