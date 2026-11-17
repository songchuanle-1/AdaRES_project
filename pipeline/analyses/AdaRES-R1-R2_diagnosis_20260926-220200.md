# Native diagnostic cycle: R1 and R2

2026-09-26 22:02 Asia/Shanghai. Prepared preflight used once. Server-2 GPU0 key SSH, fixed first 64 ScanRefer training scenes, 1660 expressions, B4-E30 best checkpoint; no training or queue mutation. R1/R2 are inference-only diagnostics and cannot establish validation efficacy.

## R1: Matched-positive supervision versus final mask quality

Reconstructed standard positive assignments from training instance labels and sampled seeds, with a nearest-center fallback. Among 388 material errors (oracle IoU minus selected IoU > 0.05), 179 selected queries were labeled positive while the oracle was negative; 83 had the reverse, 80 had both positive, 46 both negative. Standard positive queries comprise 4.53% of queries. Selected/oracle/positive-restricted oracle mean IoU: 0.82256/0.91035/0.89203. Constraining the *existing* B4 score to oracle positive labels would yield 0.87880, but those labels are not available at inference. Repeated forwards produced zero readout mismatches in the final run. This documents supervision headroom, not an admissible positive-filter method. G1 directly replaced indicator targets with detached mask IoU and finished at 0.4980 versus B4-E30 0.5094; C1 listwise indicator finished at 0.5008. Do not rename or rerun either route.

## R2: Overlap-weighted peer confidence

Predeclared directed peer confidence was soft-mask-IoU-weighted mean of *other* queries' B4 probabilities. Controls: one-query cyclic rotation of the same peer probabilities; unweighted overlap degree. The missed oracle had greater directed peer support in 287/387 material errors versus 198/387 rotated peers and 102/387 plain overlaps. On the same 1660 examples, support-only mIoU was 0.78019; frozen confidence times peer support was 0.79612, versus B4 0.82254. Rotated and plain products reached 0.72816 and 0.79839, respectively. Product flipped 650 selections. The repeated native pass had 30 readout mismatches at the strict 1e-4 threshold, and 387 rather than 388 material errors; the weaker direction cannot support admission. This specific propagation rule is rejected. The first pass was directionally similar (288/388) but was superseded by the product probe.

## Gates and next action

R1 is an observational ceiling, not a proposed executable mechanism: oracle-positive filtering is label leakage, and direct metric targets/listwise alternatives G1/C1 already failed. Reject positive-filtering as a candidate; reopen only upon independent inference-available evidence for a distinct mechanism, not a rename of G1/C1 or a validation proxy scan. R2 fails the frozen diagnosis/control intervention. Neither passes native backward or training-resource gates; no admission receipt, sync, dispatch or training launch. The 3.48 GiB R2 inference peak is not training peak. No global hypothesis exhaustion claim.
