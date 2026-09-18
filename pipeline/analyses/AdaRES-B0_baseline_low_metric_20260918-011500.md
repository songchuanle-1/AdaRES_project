================================================================================
  Experiment Result Analysis - AdaRES_project
  Experiment: AdaRES-B0 | State: closed_low_metric
================================================================================

## 1. Basic Info
  - Config: configs_song/model_paper5/adares_b0_res.yaml
  - Server: server-2 | GPUs: 0,1
  - Training: 20/20 epochs completed, seed 1999
  - Checkpoint: AdaRES_checkpoints/adares_b0_res/20260917_044215/best.pth
  - Exit note: the tensorboard writer raised EOFError during shutdown
    after the final checkpoint had already been written.

## 2. Best Metrics
| Metric | Value |
|---|---:|
| mIoU | 0.333 |
| Acc@0.50 | 0.260 |
| Acc@0.25 | 0.490 |
| spmIOU | 0.311 |

## 3. Oracle Diagnostic
| Metric | Value |
|---|---:|
| Validation expressions inspected | 124 |
| Top-1 mIoU | 0.201523 |
| All-query oracle mIoU | 0.846950 |
| Oracle gap | 0.645427 |
| Oracle query ranked first | 4 / 124 |

## 4. Conclusion
  - Candidate generation is strong: the oracle is far above the target.
  - The metric-aligned occupancy score is a poor ranking function.
  - AdaRES-B0 is closed as a baseline revision, not as evidence against
    occupancy supervision.
  - Next: AdaRES-B1 faithful standard-indicator baseline, followed by
    AdaRES-B2 metric-calibrated indicator.

## 5. Crash Info
  - No training crash. The process exited after epoch 20.
