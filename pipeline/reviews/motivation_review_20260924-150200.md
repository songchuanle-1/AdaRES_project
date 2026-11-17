# Motivation Review - AdaRES_project

2026-09-24 15:02 Asia/Shanghai

## 1. Goal
The goal is ScanRefer mIoU >= 0.5150, from a faithful learned-indicator
baseline. B1 is 0.5020; B4-E30 is 0.5094 (gap 0.0056). Allowed GPUs are
server-2 0/1, one experiment per GPU. New runs use fresh 30 epochs from the
released SPFormer sparse-UNet backbone. No validation hyperparameter search.
Report mIoU, Acc@0.25, Acc@0.5, active queries, latency and peak memory.
See `pipeline/goal.json` for the complete hard constraints and bans.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Query competition improves confidence | Supported | B4-E30 0.5094 | Causal isolation |
| Full-query pair evidence improves selection | Running | D1 on GPU0 | Trained result |
| Mask entropy, compactness and seed alignment rerank misses | Rejected | F1/F2/F3 bounded diagnoses | No tuning on validation |
| Updated decoder neighborhoods improve target evidence | Proxy only | F4 113 better / 16 worse | Native intervention and training |

## 3. Experiment Assessment
| Experiment | mIoU | Contribution |
|---|---:|---|
| B1 | 0.5020 | Learned-indicator baseline |
| B4-E30 | 0.5094 | Strongest matched-budget protocol control |
| C1-E30/C2-E30 | 0.5008/0.5014 | Ranking losses failed to improve |
| D1 | pending | All-query confidence modeling, distinct from F4 |
| F1/F2/F3 | no training | Predicted diagnostic directions disconfirmed |

## 4. Gap Analysis
F4's 16-scene diagnostic finds that fixed seed neighborhoods often cover
less final predicted foreground for missed-oracle queries. It does not test
causality, real fine-attention sampling or ground-truth foreground coverage.
Whether reanchoring helps after fresh training is unknown.

## 5. Design Rationale
F4 modifies decoder information access, whereas D1 modifies the confidence
indicator. With B4's top-k budget held fixed, a previous-layer mask provides
a causal candidate for where the next layer should gather spatial evidence.
The direction was fixed before training; neither threshold nor weight is
selected on validation.

## 6. Next Batch
Admit at most one F4 run on GPU1 after native backward, matched-B4 resource
control and all six gate proofs. If these fail, retain the stable F4 ID in
research work with the exact failure; do not occupy the GPU for its own sake.
See `pipeline/designs/exp_design_20260924-150200.md`.
