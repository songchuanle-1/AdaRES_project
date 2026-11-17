# Motivation Review - AdaRES_project

Tick 8fd5e469dee241da8f21da7c870c0d3f. Stable Y36 diagnosis and actual CPU
control preceded this review. No scientific novelty claim.

## 1. Goal

Goal description: "AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
Goal file pipeline/goal.json, SHA53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966.
Historical W2 .5100; target .515; gap .005; B1 .5020.
Phase1 unmet, Phase2 unapproved. Released SPFormer sparseUNet, own decoder,
native128, one argmax, seed1999, fixed30. No validation tuning of thresholds,
temperature, loss weights, features or epochs. Existing B2/B4 extension,
B3 terminal dependency and historical results unchanged. Authorized only
server2 GPU0/1, one independent single-GPU task/card; server6 banned.
No released b005/CFNet as B0, ranking-margin/occupancy-weight scan or scheduler
changes. Explicit 2026-10-09 GPU0 grant supersedes old reservation; preserve Y35.

## 2. Story Status

| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Point-mass competitor confidence | Historical W2 champion, goal unmet | W2 full first .510007498, repeat .509909353 this tick | Residual confidence errors; mechanism necessity unknown |
| Final query-to-token readout Y33 | Canonical fixed30 complete; frozen comparison pending | Best epoch20 .505136788; full first .505137821 | Fresh B1, paired scene CI and trained cost |
| Local spatial edge confidence Y36 | CPU structurally distinguishable only | 94c47cbf,32cases/equal-marginal delta .005216168 | Prediction-blind native diagnosis/controls/backward/resources |
| Evidence-to-word attention Y35 | Original GPU0 training | Preflight12/30/roundedbest .4961 | Frozen terminal evaluation later |

## 3. Experiment Assessment

Y31 global covariance paired CI crossed0 against W2 and B1; inconclusive,
not equivalence or no harm. Y28 direct means small-target harm, Y30 history
medium-target harm and Y34 actual existing-overlap-path counterexample stay
rejected with original evidence. Y36 is not a rerun of any rejected realization:
the new direct confidence graph reads spatial edge-assigned contextual features.
Actual W2 final confidence class consumes query competition/pointmass overlap,
while Y31 global covariance lacks this edge-feature readout. Upstream spatial
attention exists; no claim of global spatial blindness or missing information
everywhere in the decoder.

## 4. Gap Analysis

Identical global evidence can coexist with different local feature-boundary
compatibility. The CPU fixture holds final queries, mask-coordinate moments
and feature-mask marginals fixed while changing local edge assignments.
This demonstrates possible structural information, not native incidence,
causal necessity, semantic utility or expected mIoU improvement.
The first CPU run failed its intended equal-mean fixture due floating reduction
order. Exact dyadic integer fixture repaired without model/tolerance changes;
original code and failure log retained. Other controls are unchanged.

## 5. Design Rationale

Use one fixed nearest-distance graph with exact tie averaging; no k/radius
selection. Boundary/interior edge messages meet the stated mechanism rather
than changing numerical scale on the existing graph. Native TRAIN64 is
prediction-blind; GT is diagnostic-only. Matching W2/released-init controls,
all8 finite/nonzero native gradients, physical2steps/resume, disabled-parent
identity and largest-pair resource measurements determine executability.
Any adverse initialization must be investigated with original data retained.
Positive tiny gradients or synthetic discrimination cannot prove benefit.

## 6. Next Batch

See pipeline/designs/exp_design_Y36_8fd5e469.md. One potential GPU1 candidate,
not admitted yet. Complete frozen Y33 measurement before any deployment-marker
advance. All six gates and smell review required; keep capacity empty on
counterevidence or incomplete tests. Literature novelty review not performed.
