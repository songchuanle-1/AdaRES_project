# Motivation Review - AdaRES_project

2026-09-26 14:13 Asia/Shanghai

## 1. Goal
AdaRES seeks to improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 is 0.5020; B4-E30 remains champion at 0.5094, gap 0.0056. All seven goal constraints and three bans remain in force.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Competition-aware confidence | Partial gain | B4-E30 0.5094 | Target not met; historical continuation is not a fresh control |
| Metric-aligned indicator targets | Training | G1 28/30, best 0.4983 at preflight | Final outcome pending |
| Late global query communication | Closed | H5 0.5061 after 30 validations, below B4-E30 | No trained gain |

## 3. Experiment Assessment
H5's native access and backward gates were valid, but its trained result fell 0.0033 below champion. J1 and J2 probed distinct evidence and crowding mechanisms at the fixed B4 checkpoint. Neither paired diagnosis supported its causal direction. Earlier H1-H4, E1, F1-F3 diagnoses were negative; D1, F4, B3, B5 and C1/C2 training did not beat B4-E30.

## 4. Gap Analysis
There is substantial bounded oracle-mask headroom (0.87316 versus 0.49186 selected subset mIoU), but prior standalone signals and altered information paths have not reliably turned it into generalizing confidence. G1 remains the only active test. Full-run active-query, latency and peak-memory measurements remain missing for H5.

## 5. Design Rationale
J1 tested language-matched foreground-minus-background scene evidence. Oracle contrast exceeded selected in 123/240 errors, near an even split. J2 tested overlap crowding; only 78/240 missed oracles were more crowded. Neither proceeds to a candidate without a new independent mechanism and prospective diagnosis. Their observational negatives do not imply all possible routes are exhausted.

## 6. Next Batch
No admission this cycle (budget one). Resume new design when G1 reaches terminal state or a specifically articulated independent mechanism has an executable diagnostic and matched control. See `pipeline/designs/exp_design_20260926-141300.md`.
