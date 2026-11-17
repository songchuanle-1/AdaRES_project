# Motivation Review - AdaRES_project

2026-09-26 19:02 Asia/Shanghai

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 0.5020; B4-E30 0.5094; gap 0.0056. Constraints: server-2 GPUs 0/1 only, one independent task per GPU, released backbone and project decoder, fixed B2/B4 continuation, report mIoU/Acc@0.25/Acc@0.5/active queries/latency/peak memory, no validation tuning of thresholds/temperatures/weights/features/epochs. Bans: no server-6, no released checkpoint as B0, no margin/occupancy-weight scan.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Competition-aware confidence | Partial | B4-E30 0.5094 | Below target, no fresh matched control |
| Metric-IoU targets | Negative | G1 0.4980 | Do not tune targets |
| Language perturbation stability | Confounded | O1 train and validation selected mIoU decline | Semantic-preserving paired control |
| Appearance counterfactual | Negative | O2 train mIoU 0.831 to 0.286 | Localized evidence rather than global shift |

## 3. Experiment Assessment
B1 0.5020, B4-E30 0.5094, D1 0.5080, F4 0.5060, H5 0.5061, G1 0.4980. None reaches 0.515. Prior N1/N2/N3 and earlier negative diagnoses remain closed under their specific evidence. O1/O2 are diagnostic probes, not trained experiments.

## 4. Gap Analysis
The fixed B4 cohort has large oracle-selection headroom, but the oracle label is unavailable at inference. A confidence mechanism must isolate query quality without changing the expression or destroying a complete modality. Native backward, matched training controls, active-query/latency and training memory measurements are absent for O1/O2.

## 5. Design Rationale
O1 tested whether language evidence dependence distinguishes incorrect selection, using fixed content-token interventions and repeat control on training and validation scenes. O2 separately tested whether 2D appearance dependence distinguishes errors, using zero-input intervention and repeat control on training scenes. The diagnoses do not justify a trainable confidence mechanism; a token-position, feature-subset, or fusion-weight scan would violate the goal and smell check.

## 6. Next Batch
No admission. See `pipeline/designs/exp_design_20260926-190200.md`; future independently justified route must pass all six gates.
