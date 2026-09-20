================================================================================
  Motivation Review — AdaRES_project
  2026-09-20 09:37:08
================================================================================

## 1. Goal
  - Description: improve a faithful standard-indicator 3D referring
    expression segmentation baseline by learning query confidence from
    metric-aligned evidence.
  - Dataset: ScanRefer
  - Target: mIoU >= 0.515
  - Champion: AdaRES-B1, mIoU 0.502
  - Gap: 0.013
  - Constraints: server-2 GPU0/1 only; no validation scans over thresholds,
    temperatures, loss weights, feature subsets, or epochs.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|------------|--------|----------|-----|
| Candidate masks are adequate but query selection is limiting | confirmed | B0 top-1 0.202 vs oracle 0.847 on 124 expressions | Need a reliable confidence mechanism |
| A learned indicator is stronger than an analytic occupancy score | confirmed | B1 0.502 vs B0 0.333 | Still 0.013 below target |
| IoU-calibrated supervision improves the same indicator | running | B2 is training | No validation result yet |
| Indicator needs direct evidence from its own predicted mask | untested | B1/B2 indicator reads only query latent | Requires a new information path |

## 3. Experiment Assessment
| Experiment | mIoU | Relation to Motivation | Contribution |
|------------|------|------------------------|--------------|
| AdaRES-B0 | 0.333 | Analytic metric-aligned occupancy score | Revealed a 0.645 oracle gap and isolated ranking as the bottleneck |
| AdaRES-B1 | 0.502 | Faithful learned indicator baseline | Established the official strong baseline |
| AdaRES-B2 | pending | Changes indicator supervision to matched-query IoU | Tests whether the target, rather than the input evidence, is limiting |

## 4. Gap Analysis
  - Verified: mask generation contains substantially better candidates than
    top-1 selection exposes.
  - Missing: the indicator sees query latent state but does not directly see
    the mask that will be evaluated or the scene evidence inside that mask.
  - Distinguishable hypotheses: B2 tests supervision calibration; B3 tests
    missing information flow. Their conclusions remain separable.

## 5. Design Rationale
  - AdaRES-B3 pools scene features inside and outside each query's own soft
    mask and predicts confidence from query state, foreground evidence, and
    foreground-background contrast.
  - A positive result would show that mask-conditioned scene evidence closes
    the ranking gap. A negative result would rule out missing mask evidence
    and redirect work toward query competition or assignment.

## 6. Next Batch
  - One candidate only: AdaRES-B3 mask-evidence indicator.
  - It launches only if AdaRES-B2 finishes below mIoU 0.515.
  - See `pipeline/designs/exp_design_20260920-093708.md`.
================================================================================
