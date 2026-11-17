================================================================================
  Experiment Result Analysis - AdaRES_project
  Experiment: AdaRES-B2-E30 | State: done_complete
================================================================================

## 1. Basic Info
  - ID: AdaRES-B2-E30
  - Config: configs_song/model_paper5/adares_b2_metric_indicator_res.yaml
  - Server: server-2 | GPU: 1
  - Started: 2026-09-22 09:37:09 +08:00 | Finished: 2026-09-22 19:47:07 +08:00
  - Duration: 10h 09m 58s
  - State: done_complete (30/30 validations)
  - Protocol: resumed AdaRES-B2 from its epoch-20 last checkpoint with optimizer and PolyLR state preserved

## 2. Best Metrics
| Metric | Value | @ Epoch |
|---|---:|---:|
| mIoU | 0.5058 | 27 |
| Acc@0.50 | 0.558 | 27 |
| Acc@0.25 | 0.602 | 27 |
| spmIoU | 0.471 | 27 |
| spAcc@0.50 | 0.523 | 27 |
| spAcc@0.25 | 0.593 | 27 |

Best checkpoint: `/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b2_metric_indicator_res/20260921_210602/best.pth`

Active-query count, latency, and peak memory were not retained in the terminal
training evidence and are not inferred.

## 3. Convergence Summary
  - Resumed historical best: mIoU 0.504610 at epoch 20.
  - Validation mIoU for epochs 21-30 was approximately
    0.502, 0.505, 0.501, 0.505, 0.503, 0.503, 0.506, 0.504, 0.504, 0.500.
  - Best first reached at: epoch 27; the exact terminal best is 0.5058.
  - Subsequent no-improve epochs: 3 at normal 30-epoch completion.
  - Final epoch mIoU: 0.500 versus extension best 0.5058.
  - Convergence shape: a small mid-extension rise followed by a modest decline;
    the extra epochs improved the 20-epoch B2 result by only 0.0008.

## 4. Comparison
  - vs baseline AdaRES-B1: +0.0038 mIoU (0.5058 vs 0.5020).
  - vs AdaRES-B2 at 20 epochs: +0.0008 mIoU (0.5058 vs 0.5050).
  - vs AdaRES-B4 at 20 epochs: -0.0012 mIoU (0.5058 vs 0.5070).
  - vs AdaRES-B4-E30: -0.0036 mIoU (0.5058 vs 0.5094).
  - vs project target: -0.0092 mIoU (0.5058 vs 0.5150).

## 5. Conclusion
  - Conclusion: valid completion but failed to improve the champion or reach the
    target.
  - Explanation: the fixed extension slightly improved B2, while B4 retained a
    clear 30-epoch advantage, so the metric-indicator route remains inferior to
    query competition.
  - Recommendation: close B2-E30. Do not launch a duplicate, extension, or
    tuning-only successor; require a structurally distinct mechanism for any
    future metric-aligned candidate.

================================================================================
