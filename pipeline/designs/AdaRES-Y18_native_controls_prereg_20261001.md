# Y18 Native Controls, Frozen Before Expanded Captures

The first train8 capture has real point-measure gradient separation. Next use
the first64 distinct training scans in source ordering, four expressions per
scan, in two independent captures with seed1999+batch offset, same W2 checkpoint.
No validation data or outcome-dependent scan selection. Freeze final-layer
versus auxiliary-layer reporting separately: capture order is all expressions
at final layer, then at each of six auxiliary layers; do not call these seven
layers independent examples.

For each identical matched mask, compare:
1. Original W2 binary superpoint BCE/Dice.
2. Exact original-point BCE/Dice with true positive counts.
3. Mass-only: same true counts but original native binary labels.
4. Occupancy-only: unit counts with true fractional point occupancy.
5. Equal-size point-measure control: every count replaced by its expression
   mean, preserve true fractional occupancy. This retains point-scale Dice
   smoothing but removes spatial point-count information.

Freeze readouts: value/gradient separation, cosine of each negative-loss
gradient against negative soft original-point IoU gradient, final-layer scene
means, counts of favorable/unfavorable/ties, and exact same-run repeat noise.
A population trend is diagnosis only, never proof of trained efficacy.
Mixed labels, mass information and smoothing must remain distinct confounds.
Native score targets, matching, indicators and inference remain unchanged.
If point alignment is worse at released initialization, investigate actual
native loss/backward before declaring gates; do not delete adverse cases.

Next released-init8 native augmented train batches with existing30-epoch config,
gradient clipping and optimizer. Fixed stress scans from the previously
prediction-blind full-train metadata manifest, largest points/superpoints.
Use separate isolated runs for candidate and unchanged W2; no co-resident
comparison. Match random state and data. Native train entry must resolve Y18
correctly. Native gorilla checkpoint must restore model/optimizer/scheduler
before continuation; record numerical differences after continuation without
claiming bitwise or whole-dataloader resume.

Frozen inference identity: same W2 checkpoint and train8 inputs, exact native
prediction routine retained. Counterfactual labels may affect training loss
but must not alter inference. All diagnostic work uses exclusive GPU1 lease
and live idle checks, never GPU0. No admission before evidence-backed review.
