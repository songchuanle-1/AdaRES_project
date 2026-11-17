# Native diagnostic cycle: T1 and T2

2026-09-26 23:57 Asia/Shanghai. Prepared monitor/goal preflight used once. Server-2 GPU0 key SSH, fixed first 64 ScanRefer training scenes, 1660 expressions, B4-E30 best checkpoint. Read-only inference, no validation tuning or training. Both probes repeated unchanged forwards with zero readout mismatches.

## T1: Strongest-competitor causal influence

B4's indicator computes an overlap-weighted peer context. T1 removed each query's highest-overlap competitor from that context and recomputed its trained head; the matched control removed its lowest-overlap peer. Among 388 material errors, the missed oracle gained more confidence from the closest-peer removal in 216, versus only 2 for the distant-peer control. The fixed closest-peer readout changed 47 selected queries and achieved 0.82133 mIoU, below unchanged B4 0.82256. Distant-peer removal changed none and retained 0.82256. The specific causal readout has no metric-aligned benefit. Inference peak reservation was 3.48 GiB, not training memory.

## T2: Local 3D-feature support

Separately from S2's 2D features, T2 cyclically permuted 3D superpoint features inside each of the top two candidate masks, with an equal-count outside-mask control. Selected seed was retained in 1658/1660 inside and 1660/1660 outside interventions. Inside/outside mask IoU was 0.96619/0.99592; the intervention changes masks and thus cannot isolate confidence alone. Inside-mask confidence drop exceeded outside in 287/388 material errors. The frozen top-two matched-confidence rule yielded 0.81370 versus B4 0.82256 and outside control 0.82307. Full post-intervention mIoU was 0.80091 inside versus 0.82279 outside. Inference peak reservation was 6.02 GiB, not native backward or training peak.

## Gates and disposition

Both fixed rules fail diagnosis/control improvement. Reject T1 and T2; no backward, training-resource qualification, isolated model/config, admission receipt, training sync, dispatch or queue mutation. Reopen only on new independently justified metric-aligned evidence with a prospectively specified matched native control, not a feature, weight, peer or top-k scan. This is bounded route exhaustion for this tick, not global scientific exhaustion.
