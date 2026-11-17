# Native Training Progress

- Project: AdaRES_project; tick: 96ddaa5d86b54af8bf7c4d642c7d8d35.
- Preflight snapshot: 2026-10-01T11:48:29.750302+00:00.
- Local live observation completed: 2026-10-01T19:48:57.3926791+08:00.
- Authorized server: server-2, songcl@192.168.1.163; key-based SSH / BatchMode.
- Read-only probe: nvidia-smi index/memory.used/utilization.gpu, ps for original PIDs, final 1000 bytes of each original native log.

| Candidate | GPU | Original PID | Native Log Progress | Memory MiB | Utilization | Preflight Best / Latest mIoU |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[22/30] 1798 -> 1801 / 4686 | 17282 | 69% | 0.2110 / 0.1777 |
| AdaRES-Y17 | 1 | 109632 | Train[30/30] 889 -> 893 / 4686 | 22790 | 81% | 0.5042 / 0.5042 |

Both original processes match project Python, candidate configurations and single-GPU arguments. Both native tails show increasing steps. Compared with the prior local observation at 18:48:18+08, Y15 advanced from epoch21 to22 and Y17 from epoch28 to30. No crash/OOM appeared in these bounded samples; this is not exhaustive log health. No restart, dispatch, checkpoint mutation or protocol change occurred.

Preflight: capacity2, active2, running2, waiting0, due candidates0, terminal events0, admission budget0. Y15 monitor count22/30, no-improve4/30; Y17 count30/30, no-improve0/30. Y17's monitor count30/30 does NOT establish completion: the original process is still running and the native tail shows only893/4686 steps of Train[30/30]. No final epoch30 validation, terminal exit or checkpoint contents were verified. Native ETA is not a promised completion time.

Scientific state remains training under the frozen 30-epoch protocol. Y15 unfavorable observations are preserved. Intermediate best/latest metrics do not establish final efficacy, significance, Pareto benefit or goal attainment. Acc25/50, spmIoU, checkpoint contents and final full validation are not examined this tick. Completed champion remains W2 mIoU0.5100 versus target0.515.

Next action: preserve both original runs; verify an actual terminal exit and final validation before reporting completion. Handle real failures and frozen full-validation, clustered uncertainty, noise, material-harm and Pareto checks when eligible. Exact regrouped pair-off control remains necessary for Y17 causal attribution. No new variant, admission or scientific conclusion is recorded.

Execution completion is separately established by foreground finish/abort and independent lease-release verification.
