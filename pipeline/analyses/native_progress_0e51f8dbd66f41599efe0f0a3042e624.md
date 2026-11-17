# Native Training Progress

- Project: AdaRES_project; tick: 0e51f8dbd66f41599efe0f0a3042e624.
- Preflight snapshot: 2026-10-01T07:48:21.978218+00:00.
- Local live observation completed: 2026-10-01T15:49:23.9554012+08:00.
- Authorized server: server-2, songcl@192.168.1.163; SSH key / BatchMode.
- Read-only probe: nvidia-smi index/memory.used/utilization.gpu, ps for original PIDs, final 1000 bytes of each original native log.

| Candidate | GPU | Original PID | Native Log Progress | Memory MiB | Utilization | Preflight Best / Latest mIoU |
|---|---|---|---|---|---|---|
| AdaRES-Y15 | 0 | 18796 | Train[18/30] 4253 -> 4256 / 4686 | 17144 | 53% | 0.2110 / 0.2110 |
| AdaRES-Y17 | 1 | 109632 | Train[24/30] 2837 -> 2840 / 4686 | 22784 | 44% | 0.5006 / 0.4990 |

Both original processes match project Python, candidate config and single-GPU arguments. Native steps increase in both bounded samples. Relative to the prior local observation at 14:49:24+08, Y15 advanced within epoch18 and Y17 from epoch23 to24. No crash/OOM appears in these bounded samples; this is not an exhaustive log audit. No restart, dispatch, checkpoint mutation or training protocol change occurred.

Preflight: capacity2, active2, running2, waiting0, due candidates0, admission budget0, terminal events0. Y15 monitor count18/30, no-improve0/30; Y17 count24/30, no-improve4/30. Monitor metric counts and native training epoch observations are reported separately; neither proves completed epochs or final validation.

Scientific state remains training under the frozen 30-epoch protocol. Prior unfavorable Y15 observations are retained. Intermediate metrics do not establish final efficacy, significance, Pareto benefit or target attainment. Acc25/50, spmIoU, checkpoint contents and final full validation are not examined this tick. Completed champion remains W2 mIoU0.5100 versus target0.515.

Next action: preserve both original runs; handle actual failure/terminal events and frozen full-validation, clustered uncertainty, noise, material-harm and Pareto checks when eligible. Exact regrouped pair-off control remains necessary for Y17 causal attribution. No new variant, admission or scientific conclusion is recorded.

Execution completion is separately established by foreground finish/abort and independent lease-release verification.
