# Y39 Prospective Resource Screen

Tick 48678b818b134bea9d9c3c3e1f3d3305, after native controls and before
resource measurement. This does not replace or rewrite the earlier freeze.

Use the fixed inherited resource protocol: native batch2, first TRAIN pair
and the previously frozen largest TRAIN pair728/6780. Serial W2/Y39,
released SPFormer initialization, seed1999, shared tensors exactly equal.
Two warmups and three readings per pair. Prediction-blind TRAIN8 plus the
same largest pair, batch1 inference, two warmups and three readings.
Report parameters, allocated/reserved peaks, median and descriptive p95 time.

Feasibility bound: added parameters below one percent of W2; peak reserved
below the physical 24GiB card; no OOM or nonfinite native loss/gradients.
Record latency and memory regressions rather than selecting a new threshold,
dimension, subset or implementation from measured times.
Pretraining feasibility does not establish a trained lightweight Pareto result.
Any trained promotion still requires the frozen paired cost/quality review.

All adverse, incorrect-input and repeated readings are retained. Y37/GPU0
and existing checkpoint/source bytes are untouched. Each GPU1 test requires
a fresh live idleness check and an exclusive diagnostic lease.
