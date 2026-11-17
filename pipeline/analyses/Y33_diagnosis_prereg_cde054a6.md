# Y33 Final Mask-Conditioned Token Attention

Date: 2026-10-08. Prospective protocol under owned tick
cde054a6b0254e37a0c0d71febbcc497. No efficacy or novelty claim.

## Falsifiable Hypothesis

Final confidence can learn to read individual language-conditioned superpoint
tokens within its own soft mask support, rather than only their first moments.
W2 already has upstream visual attention; the hypothesis concerns a separate
final confidence readout, not global blindness or necessity. Y28's rejected
foreground/background mean-pooling realization remains rejected.

## Frozen Intervention

Physical W2 copy. Final layer only: query/key/value projections at native
d_model=256; scaled dot-product scores plus log soft foreground/background mask
support and log point mass. Exclude padding before normalization and aggregation.
Concatenate normalized query and foreground-minus-background attended values;
Linear(512,256), GELU, Linear(256,2) residual into final native indicator.
Initialize after parent layers inside fork_rng; no new scaling coefficient,
temperature, threshold, loss weight or feature search. Auxiliary outputs and
all other native losses/queries/masks remain unchanged.

## Discrimination And Controls

First execute 32 CPU cases using the actual new class. Construct equal-mass
supports with equal FG/BG means but different token distributions: native mean
pooling is identical while query-conditioned attention should differ. Repeats,
joint support permutation, joint query permutation and padded corruption must
preserve the output to numerical tolerance. Every new tensor must receive
finite nonzero gradients in nondegenerate cases; degenerate uniform FG=BG
controls are not required to have key/value gradients.

Then prediction-blind first 64 distinct TRAIN scene indices/up to four
descriptions, no outcome selection: cache physical W2 final tokens/query/masks,
audit attention versus shuffled-token/wrong-query-mask controls and exact
point-IoU oracle gaps as diagnosis only. Observe, do not require a fixed effect
size. TRAIN8/W2 plus released-init4 cached native losses include disabled,
repeat, mask-detach, wrong masks and token permutation controls. Unexpected
adverse behavior must be investigated, not hidden or threshold-tuned.

## Remaining Gates

Native full two-step backward and actual model/AdamW/PolyLR resume; all seven
new tensors finite/nonzero and updated. Matched serial native TRAIN/inference
costs including frozen stress indices728/6780. Exact physical/config identity,
no cross-variant imports and unchanged goal constraints. Six gate outputs and
mandatory motivation review precede admission. A test bundle is not admission.

## Post-Training Protocol

Only after real admission: fixed30 epochs/seed1999/released SPFormer backbone,
native128 queries/one standard argmax. Full ScanRefer validation9508 expressions,
141 scenes; candidate/W2/B1 each twice, first pass primary, repeats estimate
empirical noise. Whole-scene bootstrap10000/seed1999; size cuts from TRAIN
1487/3325 raw target points; meaningful subgroup material harm when the paired
scene-clustered interval upper bound is below zero. Intervals crossing zero
are inconclusive. Lightweight Pareto includes trained parameters, paired
latency and peak memory, not just added parameter count. No validation tuning,
GT inference input, GPU0 access or checkpoint/source overwrite.
