================================================================================
  Experiment Result Analysis - AdaRES_project
  Experiment: AdaRES-B4 | State: done_complete
================================================================================

## 1. Basic Info
  - ID: AdaRES-B4
  - Config: configs_song/model_paper5/adares_b4_query_competition_res.yaml
  - Server: server-2 | GPU: 0
  - Remote log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_b4_query_competition_res_20260921_155700.log
  - State: done_complete (20/20 validations)

## 2. Best Metrics
| Metric | Value | Epoch |
|---|---:|---:|
| mIoU | 0.5070 | 19 |
| Acc@0.50 | 0.556 | 19 |
| Acc@0.25 | 0.607 | 19 |
| spmIoU | 0.472 | 19 |
| spAcc@0.50 | 0.524 | 19 |
| spAcc@0.25 | 0.600 | 19 |

Checkpoint: `/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b4_query_competition_res/20260921_032553/best.pth`

## 3. Convergence Summary
  - The remote log contains 20 validation records.
  - B4 reached its best mIoU at epoch 19, then recorded 0.506 at epoch 20.
  - The best run improved steadily through the middle epochs and retained a
    small late-epoch advantage over the prior champion.
  - Active-query count, latency, and peak memory were not present in the
    terminal training log and remain unmeasured for B4.

## 4. Comparison
  - vs AdaRES-B1 baseline: +0.0050 mIoU (0.5070 vs 0.5020).
  - vs AdaRES-B2 champion: +0.0020 mIoU (0.5070 vs 0.5050).
  - vs project target: -0.0080 mIoU (0.5070 vs 0.5150).
  - Acc@0.25 was 0.607, equal to B2; Acc@0.50 was 0.556, above B2 by 0.001.

## 5. Conclusion
  - Conclusion: completed with demonstrated improvement; promote B4 to champion.
  - The result supports relative query competition as a useful ranking signal,
    but the one-seed result does not establish that B4 closes the target gap.
  - Retain B5 as the independent token-grounding test. Do not launch a duplicate
    or tuning-only B4 successor.

================================================================================
