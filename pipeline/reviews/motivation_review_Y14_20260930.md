# Motivation Review - AdaRES_project / Y14

Created: 2026-09-30 Asia/Shanghai. Foreground token:
`0eff5f5d4e4847debba4208c85035a3b`.

## 1. Goal

Goal description: "AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
The authoritative constraints are pipeline/goal.json: server-2 GPU0/1 only,
one task per GPU, released SPFormer backbone, no validation tuning, fixed
30-epoch protocol, and reporting accuracy, active queries, latency and memory.
W2 is 0.5100; the goal gap is 0.0050. Phase 2 has not started.

## 2. Story Status

| Hypothesis | Status | Evidence | Missing |
|---|---|---|---|
| The same query index identifies the same mask across models | refuted | Y12 | do not reuse raw indices |
| Mask-space matching contains correspondence information | supported on a diagnostic cohort | Y13 Hungarian/cyclic similarity 0.21915/0.04485 | independent trained efficacy |
| Equal confidence mixing reliably improves selection | rejected by old frozen rule | Y13 p=0.11018 and one repeat mismatch | adaptive reliability diagnosis |
| Prediction-only reliability can identify when a second model helps | untested | Y14 design | scene-held-out training and matched controls |

## 3. Experiment Assessment

W2's full validation mIoU is not interchangeable with Y13's val32 point-IoU.
Y13's mean positive direction motivates a new diagnosis, not a PASS or a
rescan. Its old rejection and artifacts remain unchanged.

## 4. Gap Analysis

The remaining question is whether correspondence quality, confidence
disagreement and prediction entropy provide target-specific reliability
information on unseen training scenes. A six-parameter gate is a screening
instrument, not sufficient evidence of a novel architecture.
Dual-model deployment may violate a lightweight Pareto requirement even if
the gate improves selection. Repeated native forwards may also be unstable.

## 5. Design Rationale

Y14 changes the tested quantity from constant mixing to prediction-conditioned
mixing learned without validation labels. State shuffling removes the relation
between a query and its reliability features; cyclic assignment removes mask
correspondence. Scene-level out-of-fold evaluation tests generalization of this
relation without choosing a validation coefficient.

## 6. Next Batch

Execute `pipeline/designs/AdaRES-Y14_prereg_20260930.md` under the same stable
ID. A diagnostic gate fit is not an admitted full-model training run.
If evidence supports the mechanism, later investigate a separately registered
single-model distillation path, measuring its own controls and native loss;
do not claim or queue that untested path now.
