# Native Progress - AdaRES_project

- Owned tick: 963720b211ae49b1a59707610cfbda4a; snapshot 2026-10-02T05:49:29.564457+08:00.
- Read-only SSH with BatchMode key access; verified hostname dl01-MS-7C60 on authorized server-2.
- Local observation window: 2026-10-02T06:11:48.9025196+08:00 to 2026-10-02T06:11:51.1823445+08:00. Remote clock is not assumed synchronized.

## Y18
- Original PID7449 present in both observations; original log AdaRES/logs/train_adares_y18_point_measure_res_20261001_233715.log.
- Native Train[9/30] last observed step4196 ->4199/4686 across two reads separated by two seconds; lr6.2e-05.
- Raw final1500byte tail SHA256: 102c22bcbb060691c70d5c7311f267d5760b1fc22f68598e371dbeaa5a1efb02 ->807a98412de5c5d564f442669342d3c181134acbfe9ac54fbbd6264cd2248536.
- GPU1 device observation:22718MiB/util51%; this is device usage, not isolated candidate peak. No full-log absence-of-error claim.
- Preflight best0.4882, epoch9, no_improve0/30. Continue original fixed30epoch training; final efficacy remains unknown.

## Y15
- Original PID18796 absent in both observations. GPU0 device observation:1MiB/util0%.
- Original log AdaRES/logs/train_adares_y15_query_completion_res_20260930_194938.log independently read at local06:12:37; final Val[30/30]363/363, mIOU0.222/Acc50.172/Acc25.354/spmIOU.203, then save last.pth.
- Raw final1400byte tail SHA256:1637f39f3de99cfb5da849e27457ec7410ad7a8d4a06b408671ad11841bd8181.
- Checkpoint metadata: last epoch30/best epoch27, best_miou0.23032023012638092. Exact checkpoint hashes are in y15_metadata_963720b2.json.
- Native training finished as observed; canonical queue is still running/deathwatch1/3 from the single preflight. Queue/experiments were not forcibly normalized.
- GPU0 full evaluations/decomposition finished. At06:12:37 nonblocking flock acquisition succeeded and was released; diagnostic process search returned1 with empty output. Persistent lock file existence does not imply a held advisory lease.
- Scientific disposition rejected by full validation, distinct from pending canonical lifecycle reconciliation. No kill, restart, new admission or validation tuning.
