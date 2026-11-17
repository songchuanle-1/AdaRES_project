# Motivation Review - AdaRES_project

## 1. Goal

Goal description: "AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
ScanRefer target0.515, W2 historical champion0.5100, B1 baseline0.5020.
Phase1 unmet, Phase2 unapproved. Applicable protocol: released SPFormer sparse
UNet, standard single-query indicator, fixed30, seed1999, no validation tuning
of thresholds/temperature/loss weights/features/epochs. Newer user whitelist
GPU1 only overrides historical goal allowance of GPUs0/1; server6 forbidden.
Historical B3 hold already expired at its terminal; at most two independent
candidates, current capacity1 and admission budget1. B2/B4 checkpoint-resume
constraints remain unchanged; their old completed work is not repeated here.

## 2. Story Status

| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Confidence can fail to choose a better native candidate | Prior TRAIN evidence | Existing Y27/Y28/Y30 diagnoses | Which evidence remains useful after full training? |
| Ordered query-mask history improves trained confidence | Same Y30 realization rejected | Frozen medium-group harm CI upper below0 | No universal temporal-family conclusion |
| Explicit within-mask shape supplies missing direct information | Y31 diagnosis candidate created first | Prereg918cbca3, actual source/CPU0187b810 | Native support, causality/backward and resources pending |

## 3. Experiment Assessment

| Experiment | mIoU | Motivation Relation | Contribution |
|---|---|---|---|
| W2 | Historical0.5100 | Point-mass overlap confidence | Current champion, not target reached |
| Y30 | Full0.504229394/0.504314270 | Ordered states and mask evolution | Medium-vs-W2 delta-0.017392276 CI[-0.029706029,-0.004614514]; reject same realization |
| Y28 | Prior full negative small-group outcome | Visual-content readback | Preserve harm; not a ban on every geometry path |
| Y27 | Prior frozen full inconclusive | Relative peer centroids | Not a champion; centered second moments are distinguishable from centroids |

## 4. Gap Analysis

The actual W2 confidence forward accepts query/mask/padding/point mass, not
spatial positions or second moments. Decoder upstream geometry remains; direct
absence is not proof of representational absence or causal necessity. New Y31
uses existing expression-expanded superpoint positions and a fixed complete
symmetric covariance representation. Approximation omits within-superpoint
variance; it must not be called exact raw-point shape.

## 5. Design Rationale

Candidate was recorded before broad review, not retrospectively after success.
CPU32 actual-class tests separate equal-centroid/equal-mass shapes, verify
translation/query/point permutation and gradients, but do not establish native
admission or trained gain. The new path aggregates mask-supported geometry
before query-conditioned confidence fusion; it is not a normalization-only
repair or feature-subset scan. A primary literature example, Huang et al.,
Mask Scoring R-CNN, CVPR2019, https://arxiv.org/abs/1903.00241, motivates testing
confidence against mask evidence generally; it does not validate this 3D
covariance implementation. Literature novelty remains unknown.

## 6. Next Batch

Only stable AdaRES-Y31, design exp_design_Y31_626ebd89.md. Execute frozen native
TRAIN64 diagnosis, matched controls, real tools.train optimizer steps and
resume, resource and independence checks. Do not admit based on CPU success.
