================================================================================
  Experiment Result Analysis - AdaRES_project
  Experiment: AdaRES-B3 | State: done_complete
================================================================================

## 1. Basic Info
  - ID: AdaRES-B3
  - Config: configs_song/model_paper5/adares_b3_mask_evidence_res.yaml
  - Server: server-2 | GPUs: 0,1
  - Remote log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_b3_20260920_091819.log
  - State: done_complete (20/20 validations)

## 2. Best Metrics
| Metric | Value | Epoch |
|---|---:|---:|
| mIoU | 0.5014 | 20 |
| Acc@0.50 | 0.554 | 20 |
| Acc@0.25 | 0.609 | 20 |
| spmIoU | 0.464 | 20 |
| spAcc@0.50 | 0.509 | 20 |
| spAcc@0.25 | 0.601 | 20 |

Checkpoint: `/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b3_mask_evidence_res/20260920_091833/best.pth`

## 3. Convergence Summary
  - The remote log contains 20 validation records and ends at mIoU 0.501.
  - The best result was reached at epoch 20, after a late rise from 0.498 at epoch 18.
  - The TensorBoardX EOFError occurred after final validation and best-checkpoint save, so it did not invalidate the run.
  - Active-query count, latency, and peak memory were not present in the terminal training log and remain unmeasured for B3.

## 4. Comparison
  - vs AdaRES-B1 baseline: -0.0006 mIoU (0.5014 vs 0.502).
  - vs AdaRES-B2 champion: -0.0036 mIoU (0.5014 vs 0.505).
  - vs project target: -0.0136 mIoU (0.5014 vs 0.515).
  - Acc@0.25 was 0.609, above B2 by 0.002 but below B1 by 0.001; mIoU ranking still favors B2.

## 5. Conclusion
  - Conclusion: completed without demonstrated improvement; retain B2 as champion.
  - The one-seed result does not establish equivalence or refute every possible mask-evidence mechanism.
  - Do not repeat this configuration or add a tuning-only successor. Any next candidate must test a structurally distinct causal hypothesis.

================================================================================
