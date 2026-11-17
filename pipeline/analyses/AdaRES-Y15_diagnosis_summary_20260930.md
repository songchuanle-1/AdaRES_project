# Y15 Candidate Completion: Actual Evidence And Remaining Gates

Foreground token 0eff5f5d4e4847debba4208c85035a3b, 2026-09-30.

## Native Train64 Diagnosis

1938 expressions, 64 actual scans, fixed W2/B4 weights.
Cross-model oracle pool scene gain +0.0116798, repeated W2 pool 0,
wrong-language B4 pool +0.0014083; cross-minus-wrong +0.0102715.
121 material teacher benefits, zero repeat material flips, zero teacher
oracle identity mismatch and zero repeated oracle-gap change.
This supports candidate-space complementarity, not deployable selection.

## Native Gradient Tests

First two scans / first four expressions: eight native mask-head gradients
finite/nonzero, one teacher gradient cosine -0.9748 to GT. No material cases.
Prospectively selected first two material scans: four material plus two control
expressions, all teacher gradients finite/nonzero, repeat logit-gradient
differences zero. Two material cases are exactly GT-only gradients; one
case's teacher/GT cosine is weaker than wrong-language control.
No tools/train.py integration or new module gradient gate was claimed.

## Student-Basis Projection

Two shape failures were preserved then fixed by explicit expression-expanded
superpoint slicing, consistent with native decoder source.
Frozen CPU float64 weighted least squares with no ridge/parameter scan.
Self-mask reconstruction relative errors approximately 1e-14.
First two material scans: 3/4 material teacher projections improve oracle,
one worsens; insufficient for efficacy or a population claim.

Prospectively expanded first eight material scans: 31 expressions total,
23 material; 17 positive / 6 negative projected gains, mean +0.0145934.
Material teacher/GT-only/shifted projected mean IoU:
0.7179021 / 0.7253393 / 0.1475330. Peak reserved 4.97 GiB.
These are oracle, diagnosis-selected TRAIN values, not test mIoU.
The teacher is not demonstrated superior to GT-only projection.

## Disposition And Next Action

Y15 remains design, not rejected and not admitted. Query completion is a
credible testable direction; naive teacher BCE alone is not justified.
See pipeline/designs/AdaRES-Y15_query_completion_design_20260930.md.
Next native task: independently implement one language-conditioned extra query,
match GT-only and extra-capacity controls, validate actual native loss/gradients,
target detach and student-only inference. No coefficient or val32 rescan.
Novelty, efficacy, deployed resource Pareto and noise floor are unresolved.
Six-gate receipts and training queue remain absent.

## Timing Caveat

Remote logs report a clock inconsistent with local tick wall time
(server date was 2026-09-29 21:31 -04 while local was 2026-09-30 10:03 +08).
Use local foreground lease timestamps for deadline/receipt and monotonic
elapsed times for diagnostics; do not infer scheduling times from remote logs.
