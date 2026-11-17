# AdaRES-Y18: Point-Measure Mask Supervision

## Hypothesis
The W2 decoder exposes masks on unequal-size superpoints but matched mask BCE
and Dice use equal superpoint measure. Replacing only matched mask supervision
by the exact original-point measure will remove a train-time metric-measure
mismatch without changing candidate count, selection, thresholds, or inference.
This is a falsifiable diagnosis, not a promise of trained improvement.

## Frozen Controls And First Test
Stable ID AdaRES-Y18; hypothesis_id point-measure-mask-supervision.
Matched control is the unchanged W2 package/checkpoint on identical frozen
train inputs, predictions, Hungarian matches and random state. The proposed
loss is exactly equivalent to broadcasting each superpoint logit to its
original points and applying binary BCE and soft Dice there, without allocating
the broadcast. Counts and positive counts are derived from the current
augmented gt_pmasks and superpoints, never from validation predictions.
No scalar weight, temperature, schedule, or threshold search is permitted.

First test: CPU analytic exhaustive/random cases compare compressed
point-measure BCE/Dice values and gradients against explicit original-point
expansion. Equal-size pure-label cases must agree with the existing measure;
unequal-size or mixed-label cases must exhibit the intended distinction.
Then inspect the native criterion/collation alignment and perform train-only
native captures on prediction-blind distinct scans, including a largest-size
stress case. Record each expression's point count, superpoint count, occupancy,
native matched mask loss/gradient, repeated controls and resource cost.
An exact no-op on actual data or reversed/unexplained native behavior is a
diagnostic failure to investigate, not an admission PASS.

## Prospective Gates
Before any admission: independent physical package/config; unchanged released
backbone and inference; native loss/backward reaches intended decoder tensors;
finite nonzero gradients; no foreign variant imports; native training entry
and checkpoint resume tested; standalone GPU1 resources checked under an
exclusive lease. GPU0 Y15 is never interrupted. All required admission gates
must have real hashed evidence. No unrun test is PASS.

## Post-Training Plan
Fixed 30 epochs, seed1999 and released SPFormer initialization, same W2 training
configuration apart from physical candidate package. Full official ScanRefer
validation, primary mIoU and Acc25/Acc50/spmIoU, W2/B1 matched frozen comparators,
128 active queries and one selected query. Scene-clustered paired intervals;
fixed train-size cuts1487/3325, official unique/multiple only if metadata exists.
Independent repeat-derived numerical uncertainty; intervals crossing benefit
or material-harm boundaries remain inconclusive. No universal noise bound is
inferred from two passes. Report standalone latency/peak memory and parameter
count; no lightweight Pareto superiority without matched measurements.
Prior adverse/closed evidence is not erased. Novelty and trained efficacy
remain unproved; no claim based solely on analytic equivalence or gradients.
