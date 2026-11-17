# Y41 prospective resource screen

Recorded before resource measurements for tick 5e172bd47ee3450dbee928bdecaede7f.

Inherit the existing prediction-blind resource protocol: released SPFormer,
seed1999, exactly equal shared initial W2 tensors; native batch2 first TRAIN
pair and frozen maximum TRAIN pair728/6780. Two warmups/three readings each.
Inference uses the first eight distinct TRAIN scenes plus those two maximum
scenes, serial batch1, two warmups/three readings; original128/oneargmax.

Feasibility: added parameters below one percent of W2, peak reserved below the
physical24GiB card, no OOM/nonfinite loss/gradients. Record time/memory regressions,
including ties/dense nearest-edge stress, without selecting a new cutoff,
dimension, radius, subset or implementation based on measured costs.

This initialized feasibility screen does not establish trained Pareto benefit.
Frozen full evaluation, subgroup harm/noise and trained paired cost review are
required before scientific promotion. Preserve Y39/GPU1, Y37 sources/checkpoints
and all original results. Every GPU0 test needs a fresh exclusive lease and
live idleness check. GPU1 is not an overflow slot.
