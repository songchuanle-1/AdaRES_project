# Native Training Progress

- Project: AdaRES_project; tick: 88b91ab5ea104504a163a8af84aaed46.
- Preflight snapshot: 2026-10-01T06:47:50.958556+00:00.
- Local live observation completed: 2026-10-01T14:49:24.8432386+08:00.
- Authorized server: server-2, songcl@192.168.1.163; key-based SSH / BatchMode.
- Read-only probe: nvidia-smi index/memory.used/utilization.gpu, ps for both original PIDs, final 1000 bytes of both original native logs.

| Candidate | GPU | Original PID | Native Log Progress | Memory MiB | Utilization | Preflight Best / Latest mIoU |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[18/30] 15 -> 19 / 4686 | 17144 | 41% | 0.2079 / 0.2018 |
| AdaRES-Y17 | 1 | 109632 | Train[23/30] 846 -> 850 / 4686 | 22784 | 67% | 0.5006 / 0.5002 |

Both original processes match the project Python, candidate config and single-GPU argument. Both native tails contain increasing steps. Compared with the prior local observation at 13:53:52+08, Y15 advanced from epoch17 and Y17 from epoch21. No restart, dispatch, training-code mutation, checkpoint mutation or protocol change occurred. No crash/OOM appeared in the bounded samples; this does not claim exhaustive log health.

Preflight saw capacity2, active2, running2, waiting0, due candidates0, terminal events0, admission budget0. Its Y15 metric count was17/30, no-improve2/30; Y17 count23/30, no-improve3/30. The later native epoch18 observation for Y15 is separate from the preflight metric count, not an invented epoch18 validation result.

Scientific state remains training under the frozen 30-epoch protocol. Y15 unfavorable observations are preserved. Intermediate best/latest metrics do not establish final efficacy, significance, Pareto benefit or goal attainment. Acc25/50, spmIoU, checkpoint contents and final full validation are not examined this tick. Completed champion remains W2 mIoU0.5100 versus target0.515.

Next action: continue both original runs; handle actual failure/terminal events and frozen full-validation, clustered uncertainty, noise, material-harm and Pareto checks when eligible. Exact regrouped pair-off control remains necessary for Y17 causal attribution. No new variant, admission or scientific conclusion is recorded.

Execution completion is separately established by foreground finish/abort and independent lease-release verification.
