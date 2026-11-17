# AdaRES-Y38 Prospective Diagnosis

Registered before route review or native outcomes, 2026-10-10.
Owning tick48678b818b134bea9d9c3c3e1f3d3305; no gate PASS claimed.

Hypothesis: explicit point-mass-weighted soft-mask change history across the
existing supervised decoder stages may supply confidence information missing
from final-mask-only readouts. The causal proposal is an indicator-only
sequence encoder, not changing masks, queries, losses, thresholds or epochs.
Final query tokens already encode decoder history indirectly; whether an
explicit history path adds useful information is unknown.

Matched controls: identical final masks, final query/features and point masses,
but distinct intermediate mask paths; identical-history repeats, superpoint
permutation and wrong-history pairing. A physical W2 source audit must verify
what the current indicator actually consumes. Compare relevant historical
realizations by executable source/fixtures before declaring independence.

First test: inspect the physical W2 decoder and run CPU paired fixtures with
equal final states and different intermediate soft-mask churn. Freeze a
readout only if these checks establish a nontrivial additional information
path. A synthetic difference is feasibility, not native incidence or benefit.
Then use frozen native TRAIN scenes for incidence, wrong-history controls,
all-new-tensor gradients, unchanged mask/loss behavior and resources.

No validation mIoU or parameter search. Existing SPFormer backbone, native128
queries/one argmax, fixed30/seed1999, optimizer/scheduler and auxiliary losses
remain unchanged. No queue entry or training admission before all six gates.
Post-training comparisons must include W2/B1, repeated FIRST-primary full
ScanRefer evaluation and scene-clustered predeclared subgroup-harm review.
No efficacy, equivalence, necessity or literature novelty assertion.
