# Y42 prospectively frozen resource and outcome protocol

Before GPU measurement: same released-backbone seed1999 W2/Y42 initial shared
tensors, native TRAIN first8 distinct scenes, fixed largest pair dataset indices
728/6780, native train batch2, serial predict batch1, two warmups/three readings.
All readings retained, no configuration selected from results.

Admission feasibility budget: new parameters below 1 percent of W2, single
authorized GPU1, peak reserved below 24 GiB, no OOM/nonfinite loss/gradient.
Report observed paired latency and memory regressions even when within budget.
No claim of Pareto improvement from initialized feasibility.

Posttraining lightweight criterion: report parameters, same paired median
latency and peak memory for trained checkpoints; no Pareto superiority unless
non-worse latency/memory with a better target metric is actually supported.
Do not retrofit budgets or thresholds to the readings.

Fixed posttraining: original30/seed1999/best native checkpoint, no validation
epoch choice, full ScanRefer9508/141/twice each Y42/W2/B1, FIRST primary,
scene bootstrap10000/seed1999, TRAIN cuts1487/3325. Harm trigger is a strictly
negative upper scene-clustered95% interval versus W2 in any preregistered
small/medium/large or single/multi-object group. Crossing intervals are
inconclusive, not equivalence/no harm. Report repeat noise from both passes
including expression-level disagreement. Official unique/multiple metadata
remains unknown unless actually supplied. Goal .515 unchanged.
