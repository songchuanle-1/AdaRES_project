# Native Training Progress

- Project: AdaRES_project; tick: 31734ed5fc994e76ba88fdc748c01897.
- Preflight snapshot: 2026-10-01T10:47:52.403439+00:00.
- Local live observation completed: 2026-10-01T18:48:18.6493487+08:00.
- Authorized server: server-2, songcl@192.168.1.163; key-based SSH / BatchMode.
- Read-only probe: nvidia-smi index/memory.used/utilization.gpu, ps for original PIDs, final 1000 bytes of both original native logs.

| Candidate | GPU | Original PID | Native Log Progress | Memory MiB | Utilization | Preflight Best / Latest mIoU |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[21/30] 2369 -> 2372 / 4686 | 17144 | 98% | 0.2110 / 0.2077 |
| AdaRES-Y17 | 1 | 109632 | Train[28/30] 3753 -> 3756 / 4686 | 22790 | 78% | 0.5030 / 0.5021 |

Both original processes match project Python, candidate configurations and single-GPU arguments. Both native tails show increasing steps. Relative to the prior local observation at 17:51:00+08, Y15 advanced from epoch20 to21 and Y17 from epoch27 to28. No crash/OOM appeared in the bounded samples; this does not establish exhaustive log health. A truncated UTF-8 progress-bar fragment at the start of a byte tail is not evidence of a training exception. No restart, dispatch, checkpoint mutation or protocol change occurred.

Preflight: capacity2, active2, running2, waiting0, due candidates0, terminal events0, admission budget0. Y15 monitor count21/30, no-improve3/30; Y17 count28/30, no-improve2/30. Monitor metric counts and native epoch labels are distinct observations, not proof of completed epochs or final validation. Native ETA is not a promised completion time.

Scientific state remains training under the frozen 30-epoch protocol. Y15 unfavorable observations are preserved. Intermediate best/latest metrics do not establish final efficacy, significance, Pareto benefit or target attainment. Acc25/50, spmIoU, checkpoint contents and final full validation are not examined this tick. Completed champion remains W2 mIoU0.5100 versus target0.515.

Next action: preserve both original runs; handle actual failure/terminal events and frozen full-validation, clustered uncertainty, noise, material-harm and Pareto checks when eligible. Exact regrouped pair-off control remains necessary for Y17 causal attribution. No new variant, admission or scientific conclusion is recorded.

Execution completion is separately established by foreground finish/abort and independent lease-release verification.
