# Native Training Progress

- Project: AdaRES_project; tick: 16dfe47b067a434980cb399a3c3fe524.
- Preflight snapshot: 2026-10-01T08:49:28.520384+00:00.
- Local live observation completed: 2026-10-01T16:50:30.3993345+08:00.
- Authorized server: server-2, songcl@192.168.1.163; key-based SSH / BatchMode.
- Read-only probe: nvidia-smi index/memory.used/utilization.gpu, ps for both original PIDs, final 1000 bytes of both original native logs.

| Candidate | GPU | Original PID | Native Log Progress | Memory MiB | Utilization | Preflight Best / Latest mIoU |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[19/30] 3721 -> 3724 / 4686 | 17144 | 85% | 0.2110 / 0.1970 |
| AdaRES-Y17 | 1 | 109632 | Train[26/30] 25 -> 29 / 4686 | 22784 | 79% | 0.5006 / 0.4992 |

Both original processes match project Python, candidate configuration and single-GPU arguments. Both native tails contain increasing steps. Compared with the prior local observation at 15:49:23+08, Y15 advanced from epoch18 to19 and Y17 from epoch24 to26. No restart, dispatch, training-code mutation, checkpoint mutation or protocol change occurred. No crash/OOM appeared in these bounded samples; this is not an exhaustive log audit. A truncated UTF-8 progress-bar fragment at the start of the Y15 byte tail is not evidence of a training exception.

Preflight saw capacity2, active2, running2, waiting0, due candidates0, terminal events0, admission budget0. Y15 monitor count19/30, no-improve1/30; Y17 count25/30, no-improve5/30. The later native epoch26 observation for Y17 is separate from the preflight metric count, not an invented epoch26 validation result.

Scientific state remains training under the frozen 30-epoch protocol. Unfavorable Y15 observations are preserved. Intermediate best/latest metrics do not establish final efficacy, significance, Pareto benefit or target attainment. Acc25/50, spmIoU, checkpoint contents and final full validation are not examined this tick. Completed champion remains W2 mIoU0.5100 versus target0.515.

Next action: continue both original runs; handle actual failure/terminal events and frozen full-validation, clustered uncertainty, noise, material-harm and Pareto checks when eligible. Exact regrouped pair-off control remains necessary for Y17 causal attribution. No new variant, admission or scientific conclusion is recorded.

Execution completion is separately established by foreground finish/abort and independent lease-release verification.
