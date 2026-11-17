# Motivation Review - AdaRES_project

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515.
All goal constraints, GPU0/1-only authorization, fixed30/seed1999 and
validation-search bans remain unchanged.

## 2. Story Status
Y43's trained asymmetry realization did not reach the target and its small
target subgroup triggered frozen harm review. Y44's full review is pending.
Both shutdown-interrupted original runs resumed and completed30, not restarted.
W2 remains the scientific comparator.

## 3. Experiment Assessment
Y45 native TRAIN8/32 expressions/128 queries: point targets equal literal
counts; all29 nonidentical cyclic wrong truths separate; same binary labels
contain multiple confusion partitions (20 matched and32 unmatched cases).
Historical B2/G1/C1 functions give identical loss/gradient on32 equal-IoU
witnesses, while categorical TP/FP/FN gradients differ. These are different
sources of evidence, not efficacy and not yet new-path native gradients.

## 4. Gap Analysis
Explicit error-partition supervision is unavailable in the parent final
matched-label classifier. Whether the parent hidden state already makes it
redundant is untested. Useful learnability, conditional control response and
resource cost are required next. The original scalar-confidence premise was
disconfirmed and its historical preregistration is preserved.

## 5. Design Rationale
SAME Y45 keeps the mask generator and inference argmax, adds an independent
final query/foreground/background decomposition readout and a native
TRAIN-only likelihood. This creates an explicit supervision/information path,
not a scalar-weight, threshold or temperature search or Y29 renamed ranking.
Multiplying parent matching odds by estimated metric odds is an unverified
modeling assumption; adverse results must be investigated and retained.

## 6. Next Batch
No admission now. Build physically independent seven-file package, then native
correct/wrong/collapsed/disabled/repeat/gradient controls, resources and restore.
Only six actual gates and tested-byte commit can authorize canonical dispatch.
Preserve original checkpoints and GPU0 PID19723. No novelty or human judgment
is claimed; literature/derivation support remains limited.
