# Native Training Progress

Local observation2026-10-01 00:49:51+08, event lease1be915df814d4aea85afb82d892ab0d4.
Preflight00:49:14: two running, no waiting, no due candidates, budget0,
no terminal/crash reported. Goal remains .515 versus completed W2 .5100.

| Experiment | GPU | PID | Native Progress | Monitor Best mIoU |
|---|---|---|---|---|
| AdaRES-Y17 | 1 | 109632 | Train[3/30]2409->2412/4686 | .3557 |
| AdaRES-Y15 | 0 | 18796 | Train[5/30]3267->3270/4686 | .1627 |

Read-only live command/GPU/log-tail check. Original singleGPU commands,
candidate configs, PIDs and unique logs retained; no restart or source update.
GPU0 memory16590MiB/util67%; GPU1 memory22782MiB/util29%.
GPU1 high memory occupancy is observed, not evidence of OOM or stagnation:
native steps continue. No independent checkpoint content or Acc25/50/spmIoU
inspection; unavailable measurements remain unknown.

Monitor sequences Y17 .2778,.3557; Y15 .0760,.1239,.1221,.1627.
Early validation improvements are normal training progress, not complete
30epoch efficacy, goal success, statistical significance or causal proof.
No threshold/temperature/coefficient/feature/epoch tuning, new admission,
dispatch, diagnostic experiment or scientific decision this tick.
Keep both fixed30epoch protocols and prospective terminal evaluation plan.
No user action required; routine monitoring only.
