# Y15 Native Integration Preregistration

Registered before new native outcomes, 2026-09-30 Asia/Shanghai.
This implements the existing AdaRES-Y15 candidate, not a new admission.

## Insertion And Matching

Append one query only at the final candidate pool. Original K spatial seed
indices and the six auxiliary outputs remain unchanged. This avoids inventing
a spatial seed for a nonspatial query. The supplemental query uses the pooled
pre-fusion projected language, an independent learned slot and one 8-head
cross-attention over student mask features, then LayerNorm.
The unchanged shared competition indicator sees all K+1 candidates.
For ScanRefer's single target, append the supplementary query matched to the
target union in the final criterion only. Preserve original auxiliary indices,
sampling loss and maps. The final loss uses the expanded match count; BCE
averages across the native matched queries plus the supplemental query. For
the supplemental query only, replace its BCE target with the detached teacher
mask in material training cases. Normalize its BCE by point mass. Keep GT
dice, score, indicator and semantic targets. There is no additional teacher
coefficient or separate additive loss. GT-only is the same reduction with the
GT target. Teacher necessity is NOT established by the existing projection.

## Frozen Mechanism Test

- Sampling unit: actual scan_id, not expression chunks. Use the first two
  material scans for target-path controls and at least two real batches from
  distinct training scans. No validation data or threshold search.
- All arms start from exactly the same shared weights, new initialization,
  RNG and batch. Teacher, GT-only, language permutation and slot-only have
  identical K+1 counts and parameter allocation.
- Expected readouts: correct/permuted language changes the extra query;
  teacher/GT changes only the extra-query BCE target; K native masks and all
  auxiliary masks are invariant when the extra slot is switched off.
- Require native model loss -> backward to every new trainable tensor,
  finite nonzero tensor norms, at least two real batches; record failures.
- Report point-mass oracle coverage, duplicated mask overlap, selected
  candidate, all tensor gradients, repeat max differences and detachment.
  New random-query task gain is descriptive, not an admission requirement.
- Teacher mask construction in the probe uses training-only oracle teacher
  masks and W2 references. It is an external probe input, not yet a full
  training target supplier. Missing teacher input must raise explicitly.
- Resource probe compares parent and candidate on the same real batches and
  warmed student-only inference. Do not pass resources from cold times.
- No full-model launch until training target supplier, six gates, physical
  independence and the post-training Pareto/harm plan are verified.

## Scope Amendment

The original draft left the insertion point unresolved. Final-only insertion
is now frozen before outcomes because native matching depends on seed indices.
Least-squares projection remains diagnosis, not an added regression loss.
Teacher target supply during tools/train.py remains executable pending work.
The GT configuration is runnable for integration tests and is NOT admitted.
