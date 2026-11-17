# AdaRES-Y3: trained point-mass identity test

Frozen 2026-09-28 12:49 Asia/Shanghai before any Y3 intervention.
Y1 found a W2 selection deficit despite similar oracle masks; Y2's target-size
association was not stable at scene level. Y3 tests the different, inference-
valid question: does W2's learned indicator actually depend on the *correct*
superpoint point masses?

On the unchanged W2 best checkpoint, fixed first 64 non-augmented train
scenes and first 32 validation scenes, capture the final indicator inputs.
Recompute the native score exactly and compare its selected-query point IoU
against (a) uniform mass 1 for every valid superpoint and (b) cyclically
rotated valid point masses within each expression. Keep query embeddings and
mask logits fixed. The oracle IoU is identical within each expression.
Predeclared positive direction: native real mass is better than both ablations
in scene-mean selected IoU and in count of material corrected selections
(>0.001), not merely score-margin movement. Each scene is a unit;
expressions are nested. Repeat every native forward twice and fail on native
score reconstruction mismatch. No tuning, no test-derived threshold, and no
claim that a frozen-weight ablation estimates retraining efficacy. If the
rotated control is as good as native, the specific point-mass identity
mechanism is disconfirmed on this cohort.
