# Native Training Progress

- Project: AdaRES_project; tick: 81ff2e4451ee499994fa66fe929b6422.
- Preflight snapshot: 2026-10-01T09:50:00.195606+00:00.
- Local live observation completed: 2026-10-01T17:51:00.2700464+08:00.
- Authorized server: server-2, songcl@192.168.1.163; key-based SSH / BatchMode.
- Read-only probe: nvidia-smi index/memory.used/utilization.gpu, ps for original PIDs, final 1000 bytes of both original native logs.

| Candidate | GPU | Original PID | Native Log Progress | Memory MiB | Utilization | Preflight Best / Latest mIoU |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[20/30] 3143 -> 3146 / 4686 | 17144 | 42% | 0.2110 / 0.2002 |
| AdaRES-Y17 | 1 | 109632 | Train[27/30] 2073 -> 2076 / 4686 | 22790 | 46% | 0.5030 / 0.5027 |

Both original processes match project Python, candidate configurations and single-GPU arguments. Both native tails show increasing steps. Relative to the prior local observation at 16:50:30+08, Y15 advanced from epoch19 to20 and Y17 from epoch26 to27. No crash/OOM appeared in the bounded samples; this does not establish exhaustive log health. A truncated UTF-8 progress-bar fragment at the start of the Y15 byte tail is not evidence of a training exception. No restart, dispatch, checkpoint mutation or protocol change occurred.

Preflight: capacity2, active2, running2, waiting0, due candidates0, terminal events0, admission budget0. Y15 monitor count20/30, no-improve2/30; Y17 count27/30, no-improve1/30. Monitor metric counts and native epoch labels are distinct observations, not proof of completed epochs or final validation.

Scientific state remains training under the frozen 30-epoch protocol. Y15 unfavorable observations are preserved. Intermediate best/latest metrics do not establish final efficacy, significance, Pareto benefit or target attainment. Acc25/50, spmIoU, checkpoint contents and final full validation are not examined this tick. Completed champion remains W2 mIoU0.5100 versus target0.515.

Next action: preserve both original runs; handle actual failure/terminal events and frozen full-validation, clustered uncertainty, noise, material-harm and Pareto checks when eligible. Exact regrouped pair-off control remains necessary for Y17 causal attribution. No new variant, admission or scientific conclusion is recorded.

Execution completion is separately established by foreground finish/abort and independent lease-release verification.
