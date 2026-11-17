# Motivation Review - AdaRES_project
2026-09-24 11:21 Asia/Shanghai

## 1. Goal
- "AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
- B1 baseline 0.5020; B4-E30 champion 0.5094; Phase 1 gap 0.0056. Phase 2 pending.
- Hard constraints: server-2 GPUs 0/1, independent single-GPU runs; released SPFormer backbone, project decoder; report mIoU, Acc@0.25/0.5, active queries, latency and peak memory. No validation tuning of thresholds, temperatures, weights, feature subsets or epoch protocol; no server-6, substituted released checkpoint, ranking-margin or occupancy-weight scan.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Local query competition aids confidence | supported at checkpoint level | B4-E30 0.5094 vs B1 0.5020 | Causality/efficiency not fully isolated |
| Full-query evidence reaches disjoint alternatives | testing | D1 training on server-2 GPU0 | Final metrics and trained resources unknown |
| Decoder-layer mask stability identifies missed query | disconfirmed on bounded subset | E1 checkpoint diagnosis: oracle more stable in 56/240 material errors | Full-validation generality unknown |

## 3. Experiment Assessment
| Experiment | mIoU | Motivation contribution |
|---|---:|---|
| B1 | 0.5020 | Faithful learned-indicator baseline |
| B2-E30 | 0.5058 | Metric-indicator calibration below B4 |
| B4-E30 | 0.5094 | Local competition champion |
| B3 / B5 | 0.5014 / 0.5030 | Scene pooling / token grounding did not improve champion |
| C1-E30 / C2-E30 | 0.5008 / 0.5014 | Listwise / hard-negative routes closed |
| D1 | running | Independent full-query pair mechanism not yet evaluated |

## 4. Gap Analysis
- Existing 16-scene B4 diagnostic: 240/441 material top-1 errors, 169/240 selected/oracle pairs with zero overlap, 148/240 oracle ranks beyond ten. This motivates D1's nonlocal evidence, not an additional near-duplicate.
- E1 probed an alternate temporal decoder signal. Although stability correlated with mask quality within expressions (mean rank correlation 0.6837), selected masks were usually more stable than missed oracles. It does not diagnose an actionable ranking defect under the fixed definition.

## 5. Design Rationale
- E1 would have learned query confidence from cross-layer mask persistence, distinct from D1 pairwise query comparison. A positive diagnostic would have justified a separate model path; the negative selected/oracle contrast rejects it before expensive implementation.
- No other candidate has diagnosis, controls, native backward, resources and independence today. Tuning an alternate stability definition on validation would violate the goal.

## 6. Next Batch
- Candidate budget 1; admitted 0. E1 rejected on disconfirming diagnosis; no filler variant. GPU1 remains open for a genuinely admissible later design.
- See `pipeline/designs/exp_design_20260924-112140.md` and `pipeline/analyses/AdaRES-E1_stability_diagnosis_20260924-112140.md`.
