# Y15 Stress And Recovery Review

2026-09-30 16:37 Asia/Shanghai. Lease a1526671b2ae48f7a80c4f8128bf645a.
Scientific status validation; no admission receipt/commit/queue/full training.

## Executed Evidence

562 actual train scans screened without model predictions. Frozen stress
scans0054/0497/0465; maximum paired batch1055503 points/7618 superpoints.
Three measured native optimizer steps per batch in separate processes:

| Arm | Max reserved GiB | Large-batch step seconds | Parameters |
|---|---|---|---|
| W2 | 14.078125 | .900-.907 | 164629637 |
| Teacher pre-hardening | 14.7109375 | 1.315-1.326 | 164959365 |
| Physical GT | 14.0859375 | .918-.922 | 164959365 |
| Hardened teacher | 14.708984375 | 1.309-1.329 | 164959365 |

No OOM/nonfinite losses; singleton remainder is explicitly not batch2.
Teacher is ~43% slower than GT on this stress pair, not an inference cost.

Native gorilla save/resume for both packages restored model/AdamW/PolyLR
exactly before continuation. Only known Roberta nonpersistent buffer keys
allowed. New9 tensor gradients remain active. Continuation is not bitwise
equal: hardened teacher max model difference3.05e-5 versus warmed repeat1.62e-5;
loss difference0, gradient difference1.49e-8. GT model difference3.99e-5,
warmed repeat1.53e-5, loss difference4.77e-7. No epsilon invented and no
whole-dataloader/RNG resume claim.

Frozen augmented8-scene specificity:32 current material expressions,
all finite gradients;5 teacher targets exactly equal GT. Mean isolated
extra-query gradient difference norm teacher-GT .11536, teacher-wrong2.45926.
Wrong identities are same-scene different objects. Teacher necessity remains
unproved. Material flags repeat stable8/8, selection flips0; one raw candidate
binary flip occurred in scene0002; preserved, not exact-repeat PASS.

Released-init train8 inference: all counterfactual and repeat seed/selection/
selected-binary flips0. Probability differences~1e-6, not bitwise equality.
Teacher absent from inference. Timing .064-.123s depending on scan, added
candidate small but noisy overhead, not population Pareto evidence.

Hardened provider verifies frozen B4 hash, isolates Python/NumPy/CPU/CUDA RNG,
includes object_ids, detects device mismatch and accepts injected diagnostic
targets without regenerating teacher. Wrong hash rejected; all four RNG
states restored exactly; tools.train two native steps updated all9 tensors.
Pre-hardening evidence preserved separately. Training objective unchanged.

## Scientific Gate Review

| Gate | Current evidence | Disposition |
|---|---|---|
| diagnosis | train64 complementary masks, repeated/wrong-language controls | supported, not efficacy |
| controls | language/slot mechanism and isolated wrong-object BCE | active mechanism, teacher superiority unknown |
| native_backward | hardened two batches plus native train2 steps | supported tested code |
| resources | clean largest-train pair, +329728 params, teacher-free inference | executable24GiB; Pareto efficacy unknown |
| independence | physical Y15/GT/local teacher packages | source audit/commit manifest still to finalize |
| constraints | fixed30/released backbone/authorized singleGPU, frozen evaluation | no receipt or dispatch yet |

The remaining launch work is operational review/source identity, not another
small-cohort mIoU/significance threshold. Do not add repeated equivalent gates.
Next owned tick: verify final package import/hash independence, preserve
versioned evidence, commit only tested candidate code/config, author six-gate
receipt/decision and queue through canonical dispatcher if all checks pass.
One Y15 candidate only: GT is causal control, not independent slot filler.
If dispatch occurs verify increasing native Train[epoch/30] steps, not PID only.
Novelty and trained teacher benefit remain unverified.

## Failure Preservation

Metadata expression-count bug, basename checkpoint path, strict Roberta buffer
keys, reused mutating optimizer config and teacher injection each have separate
failure records. Corrected diagnostic bugs without resetting optimizer, tuning
protocol or rewriting prior evidence. All probes ended;16:36:55 live GPU0/1
0%/1MiB and both diagnostic flock locks free. Local lease clock authoritative.
