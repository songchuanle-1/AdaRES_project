# Experiment Result Analysis - AdaRES_project
Experiment: AdaRES-C2-E30 | State: done_complete

## 1. Basic Info
- ID: AdaRES-C2-E30
- Config: configs_song/model_paper5/adares_c2_hard_negative_margin_e30_res.yaml
- Server: server-2 | GPU: 1
- Resumed: 2026-09-23 08:17:42 +08:00 from C2 epoch-10 last.pth with model,
  optimizer, and scheduler; terminal reconciliation: 2026-09-24 01:47:04.
- State: done_complete; 30/30 validations and final last.pth save recorded.
- Sources: pipeline/experiments/AdaRES-C2-E30.json and remote log
  /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/resume_adares_c2_hard_negative_margin_e30_res_20260923_081742.log.

## 2. Best Metrics
| Metric | Value | @ Epoch |
|---|---:|---:|
| mIoU | 0.5014 | not established from rounded stdout |
| Acc@0.50 | unavailable at exact best checkpoint | - |
| Acc@0.25 | unavailable at exact best checkpoint | - |
| spmIoU | unavailable at exact best checkpoint | - |

- Terminal record reports best mIoU 0.5014. Rounded validation stdout
  reaches 0.501 in several epochs; it cannot uniquely identify the precise
  best epoch or its aligned auxiliary metrics.
- Active queries, inference latency, and peak memory were not measured here.
- Best checkpoint save was recorded at
  /bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_c2_hard_negative_margin_e30_res/20260922_194627/best.pth;
  checkpoint contents were not loaded this tick.

## 3. Convergence Summary
- Original C2 reached 0.4980 in its first 10 epochs. The fixed continuation
  improved the terminal best to 0.5014 by epoch 30, with late validations
  fluctuating around 0.497-0.501 at stdout precision.
- Epoch 30 validation: mIoU 0.498, Acc@0.50 0.544, Acc@0.25 0.598,
  spmIoU 0.464. This is not the best-checkpoint metric tuple.
- Remote log clock and checkpoint directory timestamps differ from the local
  launch/reconciliation times; do not infer measured runtime from them.

## 4. Comparison
- vs AdaRES-B1 baseline: -0.0006 mIoU (0.5014 vs 0.5020).
- vs AdaRES-B4-E30 champion: -0.0080 (0.5014 vs 0.5094).
- vs AdaRES-B2-E30: -0.0044 (0.5014 vs 0.5058).
- vs original C2 10-epoch result: +0.0034 (0.5014 vs 0.4980).
- vs target: -0.0136 (0.5014 vs 0.5150).
- C2-E30 includes a B4-E30 warm start plus 30 adaptation epochs; there is
  no identical-budget B4-only adaptation control. These checkpoint comparisons
  do not isolate the hard-negative loss effect.

## 5. Conclusion
- Unsuccessful improvement. Retain AdaRES-B4-E30 as champion and close this
  tested hard-negative margin route; do not scan margin, weights, or epochs.
- C1-E30 remains active. No crash repair, new variant admission, or queue
  mutation follows from this terminal event.
