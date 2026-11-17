# AdaRES-Y14: Frozen Reliability Gate Diagnosis

2026-09-30 Asia/Shanghai. Token `0eff5f5d4e4847debba4208c85035a3b`.
Original Y14 design and Y12/Y13 rejected evidence are retained.

## Actual Work And Sampling Fault

The first 64-item capture contained 24 independent scans. Fit rejected it
before optimization. Source inspection confirmed dataset items partition a scan
into expression chunks. A prospective amendment, before fit and validation,
selected the first chunk for each of exactly 64 train and 32 validation scans.
Expression identity now includes scan_id, object_ids and ann_id.
Failed capture/fit logs are preserved alongside corrected logs.
Y13's historical 32 items contain only 13 independent scans; its original
item-level sign test is not an independent-scene inferential result.
No historical result, preregistration, rejection or source hash was rewritten.

## Implementation Checks

Weighted superpoint mask IoU and entropy match expanded point calculations.
Analytical pairwise-loss gradient matches autograd, max error 7.45e-9.
Distinct-scan sampling and whole-scene fold isolation tests passed.
These synthetic checks establish implementation properties, not efficacy.

## Train-Only Cross-Fitting

1938 expressions, 64 scans, 4 scene-held-out folds, 30 full-batch AdamW epochs,
lr=0.0001, weight_decay=0.0005, tau=0.25. All 12 fits have finite nonzero first
gradients for all six parameters. Effective feature rank is 3, not 6.
Native captures freeze both backbones. Backward uses cached native features;
it is not a tools/train.py native-loss integration PASS.
Final weights SHA-256:
`508a676774b3338bf39dc4d41af1a988470c6ac8cebc75c2a43f7fbb8cf6ff05`.

| Arm | OOF mean point IoU |
|---|---|
| W2 | 0.8001691 |
| Y13 reference | 0.8323527 |
| Y14 gate | 0.8299743 |
| State-shuffled control | 0.8299821 |
| Assignment-shuffled control | 0.7701000 |

Y14-minus-state scene delta -0.00000739; 63 ties and 1 negative.
Gate/state repeated-selection mismatch 1 each; assignment control 0.
W2/B4 native top-1 mismatches 0/1; 8 expressions change assignments.
Train capture 93.70 seconds, peak reserved 6.23 GiB.

## One Frozen Distinct-Scan Validation

1020 expressions, 32 actual scans. Train/val scan overlap rejected by code;
the executed check passed. Source hashes were identical at train and validation.

| Arm | Point IoU | Acc@0.25 | Acc@0.5 |
|---|---|---|---|
| W2 | 0.5338233 | 0.6352941 | 0.5872549 |
| Y13 reference | 0.5322638 | 0.6313725 | 0.5843137 |
| Y14 gate | 0.5310865 | 0.6303922 | 0.5784314 |
| State-shuffled control | 0.5310865 | 0.6303922 | 0.5784314 |
| Assignment-shuffled control | 0.5151723 | 0.6147059 | 0.5647059 |

Y14-minus-W2 scene delta -0.00270050, 14 positive/18 negative, p=0.596615.
Gate and state control select the same query for every expression.
Y14-minus-assignment scene delta +0.0158226, p=0.110184.
All three arms have zero validation selection repeat mismatches. Assignments
still change in 3 expressions (2, 12 and 36 matched-query changes); stable
top-1 does not certify a stable correspondence path.
128 active candidates; 329,259,274 dual-model parameters.
Validation capture 53.62 seconds, peak reserved 7.12 GiB.
These timings include dual repeated forwards and CPU matching, not a calibrated
single-pass deployment latency. Full ScanRefer mIoU and spmIoU are unmeasured.

## Decision

Reject this frozen Y14 realization. Task benefit is negative and no
query-feature-specific effect survives the state control. Do not change learning
rate, epochs, features or coefficients to rescue this result.
Native integration, matched deployed latency and resource Pareto gates remain
unpassed; no full-model admission, receipt, queue entry or training launch.
Y15 is a separately preregistered candidate-space diagnostic, not Y14 retuning.
No novelty claim, unbiased generalization claim or global route exhaustion.
