================================================================================
  Experiment Result Analysis — AdaRES_project
  Experiment: AdaRES-B2 | State: done_complete
================================================================================

## 1. Basic Info
  - ID: AdaRES-B2
  - Config: configs_song/model_paper5/adares_b2_metric_indicator_res.yaml
  - Server: server-2 | GPUs: 0,1
  - Completed: 2026-09-20 09:18:19 -0400
  - State: done_complete (20/20 validations)

## 2. Best Metrics
| Metric | Value | Epoch |
|---|---:|---:|
| mIoU | 0.505 | 19 |
| Acc@0.50 | 0.555 | 19 |
| Acc@0.25 | 0.607 | 19 |
| spmIoU | 0.471 | 19 |
| spAcc@0.50 | 0.525 | 19 |
| spAcc@0.25 | 0.597 | 19 |

Checkpoint: `/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b2_metric_indicator_res/20260919_204155/best.pth`

## 3. Convergence Summary
  - mIoU rose from 0.325 at epoch 1 to 0.505 at epoch 19.
  - Final epoch mIoU was 0.497, below the epoch-19 best.
  - The trajectory improved steadily through late training and did not justify early stopping.
  - A TensorBoardX writer-thread EOFError appeared only after final validation and `last.pth` were saved; it did not invalidate the run.

## 4. Comparison
  - vs AdaRES-B1 baseline: +0.003 mIoU (0.505 vs 0.502).
  - vs project target: -0.010 mIoU (0.505 vs 0.515).
  - Acc@0.25 was 0.003 below B1, while Acc@0.50 was 0.002 above B1.

## 5. Conclusion
  - Conclusion: marginally effective, target not reached.
  - True-IoU supervision improves top-1 ranking slightly, but supervision alignment alone is insufficient to close the remaining gap.
  - AdaRES-B3 is the correct independent successor because it changes the confidence input evidence rather than tuning B2.

================================================================================
