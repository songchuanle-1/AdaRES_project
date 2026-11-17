# Native diagnostic cycle: S1 and S2

2026-09-26 23:25 Asia/Shanghai. Prepared monitor/goal preflight used once. Server-2 GPU0 key SSH; fixed first 64 ScanRefer training scenes, 1660 expressions, B4-E30 best checkpoint. Native read-only interventions; no training or validation tuning.

## S1: Superpoint-order equivariance

Reverse decoder superpoint order, remap sampled seeds, and restore mask coordinates. Unchanged-input repeated readout mismatch 0. Among 388 material selection errors (oracle minus selected IoU > 0.05), missed oracle more mask-stable in 0; selected/oracle mean mask stability 1.00000/0.99849. Mean selected mask IoU across order reversal 0.99939, while same selection survived 56.51% of expressions. Frozen matched-score averaging changed 129 selections and reduced mIoU from B4 0.82256 to 0.80927; order-only confidence mIoU 0.78493. The instability is not a useful metric-aligned selector. Reject S1, no training gate.

## S2: Local 2D-feature support

For each expression's selected mask, cyclically permute 2D features within its predicted superpoints; control permutes an equal number outside. Retain feature distribution and language. The final seed-matched native run yielded 0 repeated-forward mismatches, selected seed retained in 1653/1660 inside and 1660/1660 outside interventions. Inside/outside selected-mask IoU 0.96763/0.99722 and selection flips 614/54. Mean mIoU after intervention 0.80720/0.82306 versus B4 0.82256. Inside-minus-outside confidence drop was higher in 259/388 material errors; seed-matched mean difference 0.02666 for errors versus 0.01424 for correct selections. Earlier overlap-matched deltas were inflated by query reassignment and superseded. The association is not a deployable selection rule.

Frozen top-two matched-confidence rule tested each query with its own inside-mask permutation. It achieved 0.82007 mIoU versus unchanged B4 0.82256 and outside-mask control 0.82271, with 150 changed outcomes. The readout does not establish a causal improvement, and the inside intervention changes masks; no confidence pathway is admitted. The 6.31 GiB inference reservation is not a backward/training resource measurement.

## Gates and disposition

Both routes fail frozen diagnosis/control selection improvement. S1 and the specific S2 top-two retention rule are rejected. No native backward, training-resource qualification, independent model/config package, admission receipt, sync for training, dispatch or queue mutation. The probe scripts were copied to server-2 for read-only execution only. No global hypothesis-exhaustion claim; revisit S2 only with a predeclared inference-valid mechanism that survives an equal-area outside control and a paired native training diagnosis, not a weight/threshold/top-k scan.
