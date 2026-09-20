================================================================================
  Experiment Result Analysis — AdaRES_project
  Experiment: AdaRES-B1 | State: done_complete
================================================================================

## 1. Basic Info
  - ID: AdaRES-B1
  - Config: configs_song/model_paper5/adares_b1_standard_res.yaml
  - Server: server-2 | GPU: 0,1
  - Started: 2026-09-18 01:08:54 -0400 | Finished: 2026-09-18 12:50:57 -0400
  - State: done_complete

## 2. Best Metrics
  | Metric | Value | @ Epoch |
  |--------|-------|---------|
  | mIoU | 0.502 | 18 |
  | Acc@50 | 0.553 | 18 |
  | Acc@25 | 0.610 | 18 |
  | spmIOU | 0.467 | 18 |

## 3. Convergence Summary
  - Best first reached at epoch 18.
  - Final epoch mIoU was 0.496 versus the historical best of 0.502.
  - Validation improved steadily from 0.342 at epoch 1 and plateaued near 0.50.
  - The trailing TensorBoard writer EOFError occurred after checkpoints were saved and did not invalidate the run.

## 4. Comparison
  - vs AdaRES-B0: +0.169 mIoU.
  - vs target 0.515: -0.013 mIoU.

## 5. Conclusion
  - Conclusion: effective baseline.
  - The standard learned indicator resolves most of B0's query-ranking failure.
  - AdaRES-B2 now tests whether supervising indicator confidence with true matched-query IoU can close the remaining 0.013 gap.
================================================================================
