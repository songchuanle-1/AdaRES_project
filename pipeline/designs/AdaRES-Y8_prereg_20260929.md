# AdaRES-Y8 preregistration: competition-graph score diffusion

Date: 2026-09-29 Asia/Shanghai

## Frozen question

Does one fixed step of score diffusion over the model's own mask-overlap graph
change the selected query toward the oracle more often than a degree-preserving
row-permuted graph control?

W2's indicator score is `s in R^Q`. The predicted mask overlap graph is

`O_ij = <p_i, p_j> / (sum(p_i) + sum(p_j) - <p_i, p_j>)`,  `O_ii = 0`,

with `p_i = sigmoid(mask_logits_i) * point_mass`. Row normalization gives

`P = O / (O 1)`, and the frozen diffusion readout is

`s' = 0.5 s + 0.5 P s`.

The coefficient is fixed at one half before execution and is not scanned.

## Matched control

For each expression, eight fixed row permutations `pi_m` are drawn once with
seed 1999. The control graph is `P[pi_m, :]`; each row still sums to one and has
the same overlap weights, but those weights are attached to the wrong query
rows. The control readout is the mean over all eight permutations.

## Cohort and decision rule

- Model: W2 30-epoch best checkpoint, unchanged.
- Cohort: first 32 non-augmented ScanRefer validation scenes, seed 1999.
- Primary readout: selected-query point IoU.
- Pass only if all three hold:
  1. Scene-mean paired delta `IoU(diffused) - IoU(raw)` is positive.
  2. Scene-mean paired delta `IoU(diffused) - IoU(control mean)` is positive.
  3. A two-sided scene-level sign test on `IoU(diffused) - IoU(raw)` over
     nonzero scenes has `p < 0.05`.

Failure rejects this graph-diffusion route. A pass is a diagnosis result, not
training admission; native backward, resource, independence, constraints, and
an admission receipt would still be required for any trainable realization.

## Independence from prior routes

This is not D1: D1 learned a terminal pairwise query comparison. Y8 is a
parameter-free, inference-available one-step propagation over fixed predicted
mask overlap. It does not use native labels, oracle membership, or validation
thresholds. It is not C2: no positive/negative margin is optimized.

## Repeatability

Each scene is forwarded twice. Raw float mismatches are reported separately
from changes in selected-query identity and material-error status. Any
selection-identity instability is reported and prevents fine-grained claims.
