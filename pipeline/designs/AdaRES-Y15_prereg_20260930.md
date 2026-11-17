# AdaRES-Y15 Preregistration: Cross-Model Candidate Complementarity

2026-09-30 Asia/Shanghai, before native Y15 capture.
Stable ID: AdaRES-Y15. Hypothesis ID: complementary-mask-candidate-pool.
Motivation: metric-aligned query evidence may be insufficient if useful masks
are absent from the student's candidate pool. Y14 changed only selection;
Y15 tests candidate-space information rather than another score gate.

## Diagnosis And Derivation

Unchanged W2/B4 checkpoints, first 64 distinct training scans, the first
expression item per scan (at most 32 expressions), no augmentation.
The sampling correction is identical to Y14; actual scan_id is the unit.
Training labels are permitted for diagnosis, never an inference selector.

For target e, let a=max_i IoU(W2_i,e), b=max_j IoU(B4_j,e),
a2=max_i IoU(W2_repeat_i,e). The cross-model oracle gain is
G_cross=max(a,b)-a, and matched equal-count repeat gain is
G_repeat=max(a,a2)-a. These controls both double the candidate count.
Test whether scene-mean G_cross-G_repeat is positive on the fixed train cohort.
Record sign test descriptively, not as a pretraining admission requirement.

The language control uses the first cyclically subsequent expression in the
same item with a different object-ID set, evaluates its B4 masks on target e,
and forms G_wrong=max(a,max_j IoU(B4_wrong_j,e))-a.
If no different-object expression exists, mark that control unavailable, not
zero. Compare G_cross and G_wrong only on the matched eligible cohort.
True target-specific evidence requires a positive cross-minus-repeat direction
and positive cross-minus-wrong direction, with repeat-stable material events.
A language-unrelated pool improvement does not support the claimed targeted
distillation rationale.

Material events use the already-frozen oracle gap 0.05: b-a>0.05.
Repeat W2 and B4 consecutively on the same batch without reseeding. Record
teacher oracle identity, gain changes and material-event flips; no selected
query change is required for this mask-availability diagnosis.

## Controls And Implementation

All binary masks threshold at 0.5; compute point-count-weighted IoU exactly.
Use same W2/B4 data and GPU execution as Y14, no coefficient or threshold search.
Compare direct B4 teacher-oracle mask with its Hungarian-aligned W2 counterpart
to quantify missing mask shape rather than only ranking disagreement.
Return per-expression objects/annotations, real scans, matched-control counts,
parameters, elapsed time and GPU peak memory.
No validation evaluation is authorized by this preregistration.

## Interpretation And Next Action

If directions are reversed or wrong-language control explains the effect,
reject this target-specific candidate-space realization with evidence.
If supported, preregister a student residual-mask distillation mechanism on
the same stable ID, explicitly train-only teacher supervision and student-only
inference. No teacher oracle is available at inference.
Do not implement or launch full training based on the oracle gap alone.
Native backward, lightweight resources, independent package, constraints and
post-training efficacy plan are still required. If inconclusive, preserve
exact pending evidence/next action instead of inventing rejection.

## Novelty And Risks

Mask distillation and teacher-student 3D referring segmentation already exist.
This diagnostic is not a novelty claim. A subsequent method must distinguish
target-specific missing-mask transfer from ordinary confidence pseudo-labeling,
and demonstrate student-only inference cost and genuine controlled behavior.
Wrong-language masks can still share scene geometry; this is an intentional
strong control. Available expression subsets may differ across scans.
