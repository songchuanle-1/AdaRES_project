# AdaRES-Y1: Paired W2/B4 error-source audit

Frozen before the new native test on 2026-09-28 Asia/Shanghai. This is a
diagnosis, not a new trained variant and not a gate admission.

## Falsifiable hypothesis
The W2 point-mass competition changes the learned query ranking more than
the candidate masks. On the same fixed first 64 non-augmented ScanRefer
training scenes, W2's selected-query mean point IoU is far below B4-E30's
historical 0.82256. If candidate availability remains comparable but
selection drops, a distinct ranking mechanism may be justified. If oracle
quality or cohort alignment accounts for the difference, ranking-only work
is not justified by this observation.

## Matched control and units
Run W2 and B4-E30 best checkpoints on exactly the same first 64 training
scenes with identical data settings (training=False, aug=False,
with_elastic=False, lang_num_max=32), the same native predictor thresholds,
and fixed seed 1999. Scene is the sampling unit, expressions are nested;
report paired per-expression selected and oracle IoU, oracle rank and
repeatability. Also run the same comparison on a frozen first 32 validation
scenes as a diagnostic of cohort transfer, with **no tuning or threshold
selection** based on validation outcomes. No parameter search.

Record unmatched expression IDs or count differences as a test failure.
Report training and validation descriptive results separately, not as a
population significance claim. Repeat native prediction twice per scene;
do not overinterpret score-margin differences if repeated masks differ.
Use the published B4/W2 checkpoints unchanged. If a split cannot be
completed safely this tick, retain the stable Y1 ID and exact next command
as pending rather than claiming a PASS.

## Decision boundary
The test identifies a bottleneck, not a candidate's efficacy. A new
independent trainable mechanism would still require preregistered causal
control, native backward, resources, physical independence, constraints,
and frozen post-training comparison. No 30-epoch dispatch on this audit alone.
