================================================================================
  Experiment Result Analysis - AdaRES_project
  Experiment: AdaRES-B4-E30 | State: done_complete
================================================================================

## 1. Basic Info
  - ID: AdaRES-B4-E30
  - Config: configs_song/model_paper5/adares_b4_query_competition_res.yaml
  - Server: server-2 | GPU: 0
  - Started: 2026-09-22 09:33:23 +08:00 | Finished: 2026-09-22 19:47:05 +08:00
  - Duration: 10h 13m 42s
  - State: done_complete (30/30 validations)
  - Protocol: resumed AdaRES-B4 from its epoch-20 last checkpoint with optimizer and PolyLR state preserved

## 2. Best Metrics
| Metric | Value | @ Epoch |
|---|---:|---:|
| mIoU | 0.5094 | 27 |
| Acc@0.50 | 0.562 | 27 |
| Acc@0.25 | 0.608 | 27 |
| spmIoU | 0.474 | 27 |
| spAcc@0.50 | 0.524 | 27 |
| spAcc@0.25 | 0.601 | 27 |

Best checkpoint: `/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b4_query_competition_res/20260921_210214/best.pth`

Active-query count, latency, and peak memory were not retained in the terminal
training evidence and are not inferred.

## 3. Convergence Summary
  - Resumed historical best: mIoU 0.507217 at epoch 20.
  - Validation mIoU for epochs 21-30 was approximately
    0.502, 0.505, 0.508, 0.508, 0.503, 0.506, 0.509, 0.507, 0.507, 0.502.
  - Best first reached at: epoch 27; the exact terminal best is 0.5094.
  - Subsequent no-improve epochs: 3 at normal 30-epoch completion.
  - Final epoch mIoU: 0.502 versus extension best 0.5094.
  - Convergence shape: an early plateau followed by a late peak and then a
    decline; the fixed extension improved the 20-epoch B4 result by 0.0024.

## 4. Comparison
  - vs baseline AdaRES-B1: +0.0074 mIoU (0.5094 vs 0.5020).
  - vs AdaRES-B4 at 20 epochs: +0.0024 mIoU (0.5094 vs 0.5070).
  - vs AdaRES-B2 at 20 epochs: +0.0044 mIoU (0.5094 vs 0.5050).
  - vs AdaRES-B2-E30: +0.0036 mIoU (0.5094 vs 0.5058).
  - vs project target: -0.0056 mIoU (0.5094 vs 0.5150).

## 5. Conclusion
  - Conclusion: valid improvement; promote B4-E30 to the best completed result,
    but the Phase 1 target remains unmet.
  - Explanation: query competition preserved its advantage over B2 under the
    matched fixed 30-epoch protocol and reduced the target gap to 0.0056.
  - Recommendation: use B4-E30 as the next champion reference. Do not launch a
    duplicate or tuning-only successor; the next design must address the
    remaining gap with a structurally independent, admissible mechanism.

================================================================================
