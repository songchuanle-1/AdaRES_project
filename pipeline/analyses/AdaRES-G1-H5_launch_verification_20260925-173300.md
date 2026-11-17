# G1/H5 native launch verification

2026-09-25 17:33 Asia/Shanghai. Both model packages were committed
independently, then nine additive AdaRES files per commit were hash-verified
by `scripts/sync.sh server-2`. The remote `.codex_source_commit` advanced
to `dafa28c03e6d13d68d2a572d9e725f4ed40227dc`.

| ID | GPU | Committed candidate | Main PID | Unique log | Native progress |
|---|---:|---|---:|---|---|
| G1 | 0 | `0f5f695d478b87750d69561428f6421833e629a9` | 11694 | `logs/train_adares_g1_metric_targets_res_20260925_170802.log` | `Train[1/30]` 0 to 2967/4686 |
| H5 | 1 | `dafa28c03e6d13d68d2a572d9e725f4ed40227dc` | 36497 | `logs/train_adares_h5_late_global_query_res_20260925_173110.log` | `Train[1/30]` 0 to 184/4686 |

GPU memory at 17:33: GPU0 17392 MiB; GPU1 17016 MiB. These are live
used-memory readings, not training peak allocation. Validation and
efficiency metrics remain pending. No server-6 dispatch occurred.
