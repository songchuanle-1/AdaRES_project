# Y15 Student Query-Basis Representability Preregistration

2026-09-30, before projection execution. Same first-two-material-scan manifest
from the frozen Y15 diagnosis. All native weights unchanged.

## Falsifiable Mechanism

The student's decoder makes mask logits x=Fq, with superpoint feature rows F
and query vector q. If teacher missing-mask logits can be represented in this
basis but no current query produces them, the deficit is candidate generation,
not solely unavailable backbone features. If projection is inaccurate, a new
query generator alone cannot be justified.

Capture the actual final dec.x_mask output F and teacher oracle logit target
t=logit(clamp(pB,1e-7,1-1e-7)). The fixed numerical clamp is not a mask cutoff.
Point mass D=diag(n_s) gives the weighted least-squares problem:

q*=argmin_q ||D^(1/2)(Fq-t)||_2^2
q*=(D^(1/2)F)^+ D^(1/2)t.

Use CPU float64 torch.linalg.lstsq driver=gelsd with its standard rcond,
no ridge coefficient or numerical-parameter scan. Apply binary threshold 0 to
projected logits (equivalent to the native sigmoid threshold 0.5).
Use teacher logits directly from native mask output before sigmoid, rather than
reconstructing saturated probabilities. The t=logit(p) formula describes the
same mathematical target; unsaturated native logits avoid a clamp artifact.

Matched controls:
1. Student's own oracle logits: exact basis reconstruction sanity check.
2. Circularly shift teacher logits one superpoint while holding F and D fixed.
3. GT-only signed target tGT=2y-1, projected with the same solver.

Report point-IoU against target, weighted residual relative to target norm,
rank, own reconstruction error and material subgroup. No statistical claim from
two selected scenes. Projection is an oracle feasibility test, not inference
or training benefit. GT-only recovery may show teacher supervision redundant.

## Decision

Positive teacher recovery beyond current W2 candidates plus separated shifted
control motivates an inference-valid query-generation hypothesis under Y15.
Teacher and GT-only recovery must be compared before claiming teacher necessity.
No numerical scans, validation, parameter update, candidate admission or
training launch. Negative reconstruction/debug failures require diagnosis.
