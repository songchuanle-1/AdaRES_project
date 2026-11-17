# AdaRES-W2 launch verification

2026-09-27 03:24 Asia/Shanghai. The existing queue monitor dispatched the
receipt-backed W2 config on server-2 GPU0 after `sync.sh` verified commit
`deb70191a7b16dffa624acb1fb32e9d45bd8d551` against the remote source
marker. Server-6 was excluded. The unique training log is
`/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_w2_pointmass_competition_res_20260927_032211.log`;
its PID marker reads 20111. Native progress increased from `Train[1/30]
0/4686` to `120/4686` and then `215/4686`. At the latter check, GPU0 used
16890/24576 MiB with 56% utilization; GPU1 used 1 MiB and remained idle.
Worker processes share the parent training command and do not represent
additional independent GPU jobs. Neither validation mIoU nor final peak
memory is available.

The Windows `Pipeline-AdaRES_project` scheduled task remains Disabled.
`pipeline/execution_mode.json` is foreground. This project has no
`pipeline/closure_policy.json`, while the foreground `finish` command
requires an enabled closure policy; the remote checkout has no `.git`, so
enabling that policy without a compatible migration would invalidate its
safe fast-forward launcher. This operational mismatch must be resolved
separately; do not report a finalized frontend tick or stop the native
training because of it.
