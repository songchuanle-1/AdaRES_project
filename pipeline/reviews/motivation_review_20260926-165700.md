# Motivation Review - AdaRES_project

2026-09-26 16:57 Asia/Shanghai

## 1. Goal
Goal: "AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515." B1 0.5020, B4-E30 champion 0.5094, gap 0.0056. Seven goal constraints and three bans apply; server-2 GPUs 0/1 only, one training task per GPU, released backbone and project decoder, no validation hyperparameter scan.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Competition-aware confidence | Partial | B4-E30 0.5094 | Below target; no fresh seed-matched control |
| Metric-aligned indicator targets | G1 queue running, deathwatch 2/3 | Targeted log shows Val[30/30] mIoU 0.497 | Monitor terminal reconciliation and best metric |
| Seed containment | Disconfirmed as stated | M1 oracle higher 48/240 errors | No admission |
| 2D mask-seed appearance support | Disconfirmed as stated | M2 oracle higher 80/240 errors | No admission |

## 3. Experiment Assessment
B1 0.5020 establishes baseline. B4-E30 0.5094 is champion. B2-E30 0.5058, C1/C2-E30 0.5008/0.5014, F4 0.5060, D1 0.5080 and H5 0.5061 do not improve it. G1 is not assigned a terminal state from an absent process or the last validation alone. H/J/K/L fixed-cohort screens rejected their specific mechanisms. M1 support-only 0.43714 and M2 appearance-only 0.11752 are below B4 fixed-subset 0.49186.

## 4. Gap Analysis
The fixed cohort's oracle mIoU 0.87316 leaves selection headroom but does not establish any usable confidence signal. Full G1 terminal outcome, active queries, latency, peak memory, and subgroup effects remain open. No new candidate passes six hard gates.

## 5. Design Rationale
M1 asked whether inconsistent mask/seed containment causes B4 errors; M2 asked whether projected 2D appearance can supply independent mask-seed support. Either positive directional signal would have motivated a matched B4 control and candidate-specific backward/resource tests. Both were contradicted on the fixed cohort, so training a renamed feature-fusion or weight variant would not test an evidence-backed mechanism.

## 6. Next Batch
No admission. See `pipeline/designs/exp_design_20260926-165700.md`. Resume after normal G1 terminal reconciliation with a new error diagnosis, or independent prospective evidence specifying a stable candidate and native next action.
