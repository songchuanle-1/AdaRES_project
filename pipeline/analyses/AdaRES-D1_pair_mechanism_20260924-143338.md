# AdaRES-D1 pairwise mechanism counterfactual

2026-09-24 14:33 Asia/Shanghai. Read-only, fixed first 16 ScanRefer
validation scenes (441 expressions), server-2 GPU1. D1 continues normal
training on GPU0. The D1 `best.pth` used here has metadata epoch 4 and
full-validation best mIoU 0.44097984.

## Intervention

With the trained D1 model, decoder query features, mask logits, and confidence
head held fixed, replace only the learned per-query softmax rival weights
`softmax(pair_score(q_i, q_j, overlap_ij, area_delta_ij))` with equal weights
over all *other* queries. Recompute the competitor context and the two
overlap/area aggregate features before the same confidence head. The
unmodified forward output was checked against a reconstruction from the
original learned weights with `torch.testing.assert_close`. The top-1
prediction uses the original RES argmax inference in both branches. Ground
truth is used after prediction for diagnostic IoU and oracle rank only.

## Observation

| Fixed subset metric | Learned weights | Uniform rivals |
|---|---:|---:|
| Top-1 mIoU | 0.41114 | 0.39926 |
| Oracle mask mIoU (identical masks) | 0.83894 | 0.83894 |
| Mean rank of the oracle query | 18.17 | 19.02 |
| Disjoint rival mass (`soft overlap < 0.1`) | 0.83956 | 0.87698 |

The intervention changed the selected query for 76/441 expressions.
Of these, 18 changed selections improved IoU by >0.05, 10 worsened it by
>0.05, and 31 had an absolute IoU difference <1e-5. Oracle rank improved
on 159 expressions and worsened on 101. Per-scene mean differences are
positive on 9 scenes, negative on 6, and zero on 1.

A fixed-seed (1999), 10,000-resample *scene-cluster* bootstrap of the
expression-weighted mIoU difference (learned minus uniform) yielded
95% percentile interval [-0.00469, 0.02829]; 8.56% of resamples
were nonpositive. This is exploratory uncertainty on only 16 scenes,
not an independent full-validation significance test.

## Interpretation and limits

- The learned pair-score route is causally used by the epoch-4 model:
  removing its nonuniform weighting changes ranking on 76 examples.
  The positive subset delta (+0.01188) is **not** robustly established
  across scenes because the interval crosses zero.
- Learned weighting gives *less* mass to disjoint rivals than uniform
  weighting under this probe. This does not show that disjoint comparison
  itself helped; soft overlap and the `0.1` partition are diagnostic,
  not an inference threshold or tuned model parameter.
- Uniform rival weights are out-of-distribution for the trained head;
  this paired intervention measures sensitivity, not the effect of
  training a matched uniform-context control from scratch. There is
  only one seed and one early checkpoint. No claim about final D1 vs B4
  or paper-level novelty follows from this experiment.
- Repeat this *fixed* intervention after the D1 30-epoch protocol or
  legitimate terminal state, and compare full-validation metrics to B4.
  A specifically nonlocal claim needs a further predeclared control
  that isolates disjoint rivals without tuning on this validation subset.

Artifacts: `pipeline/analyses/AdaRES-D1_pair_mechanism_16scenes_epoch4_20260924.json`
(raw per-expression rows), `AdaRES/tools/diagnose_adares_d1_pair_mechanism.py`
(native forward probe), and
`AdaRES/tools/analyze_adares_d1_pair_mechanism.py` (scene bootstrap).
An initial probe failed before output because the uniform weights lacked
the expanded-expression batch dimension; the corrected probe then
completed twice with the same top-1 results. No checkpoint, queue, config,
optimizer, or running training process was changed.
