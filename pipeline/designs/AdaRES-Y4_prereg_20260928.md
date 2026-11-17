# AdaRES-Y4: seed-supervision versus mask-oracle coverage

Frozen 2026-09-28 12:56 Asia/Shanghai before Y4 execution. Distinct from
Y2 target-size association and Y3 point-mass identity ablation: this tests
whether the indicator's seed-instance positive labels omit the best-IoU
query, despite that query's mask being available at inference.

Run the unchanged W2 best model on the same first 64 train scenes and first
32 val scenes without augmentation, seed 1999, twice per scene. Reconstruct
the positive query set by indexing the expanded `sp_ins_labels` at
`fps_seed_sp[sample_inds]`, then matching the expression's object_id(s), as
`Criterion.match_sampled_ins` does for in-object sampled seeds. Note that
the native loss also force-matches a nearest query for each GT instance:
therefore this reconstruction is an **in-object subset**, not an exact full
positive set. Report its coverage of the oracle and native selected query
both among all expressions and among material selection errors
(oracle-selected IoU > 0.05). A one-step cyclic shift of query seed labels
within the expression is the negative control. Record eligible expressions
with at least one in-object query, query-count distributions, and repeat
mismatches. Scene is the unit, expression is nested.

Expected direction: among material errors, the true in-object label covers
oracle much less often than it covers selected, but exceeds rotated-label
oracle coverage. If the supposed supervision mismatch is not enriched,
reject the label-coverage rationale. This test cannot identify the full
nearest-query matching set nor demonstrate a new loss has native gradients,
and is not a training admission or validation tuning exercise.

## Execution amendment, 2026-09-28 12:57 Asia/Shanghai

The initial diagnostic raised `IndexError` before writing any result:
`training=False` does not populate `sp_ins_labels`. No Y4 outcome was seen.
Use the inference batch's `gt_spmasks[index]` indexed by sampled query seed
superpoint instead. This tests target-support seed coverage, **not** the
native instance-ID positive label or forced nearest match. Keep the same
cohorts, rotation control, and expected directional contrast. Do not claim
exact training-label coverage from this proxy.

## Interface correction, 2026-09-28 13:03 Asia/Shanghai

Inspection of the native `scannetv2.py` collate function shows `sp_ins_labels`
contains **one entry per scene**, not one per expression. The first IndexError
was caused by indexing this list with expression index. The proxy result has
already been seen and remains labeled as a proxy. A fresh execution will
add per-expression `sp_ins_labels[0]` target-ID match and the same rotated
control; the original proxy output remains immutable. This prospective
correction tests the training loss's in-object sampled-seed subset, still
excluding its forced nearest-query assignment. It is not a fully reconstructed
criterion label set or an efficacy test.
