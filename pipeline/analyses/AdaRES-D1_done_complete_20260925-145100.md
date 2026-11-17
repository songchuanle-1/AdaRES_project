================================================================================
  Experiment Result Analysis - AdaRES_project
  Experiment: AdaRES-D1 | State: done_complete
================================================================================

## 1. Basic Info
- ID: AdaRES-D1
- Config: configs_song/model_paper5/adares_d1_full_query_comparison_res.yaml
- Server: server-2 | GPU: 0
- Started: 2026-09-24 10:47:09 +08:00 | Reconciled: 2026-09-25 14:47:04 +08:00
- State: done_complete; 30/30 validations and final last.pth save.
- Sources: pipeline/experiments/AdaRES-D1.json and remote
  logs/train_adares_d1_full_query_comparison_res_20260924_104702.log.

## 2. Best Metrics
| Metric | Value | @ Epoch |
|---|---:|---:|
| mIoU | 0.5080 | 15, rounded stdout |
| Acc@0.50 | 0.561 | 15, rounded stdout |
| Acc@0.25 | 0.611 | 15, rounded stdout |
| spmIoU | 0.473 | 15, rounded stdout |

- The terminal record supplies best mIoU 0.5080. Epoch and auxiliary metrics
  are rounded stdout values, not full-validation checkpoint re-evaluation.
- Best-checkpoint fixed first 16 scenes (441 expressions): effective active
  queries 16.05 (inverse concentration of indicator probabilities), latency
  14.12 ms/expression and peak allocated inference memory 3570 MiB. These are
  not full-validation latency or training peak memory. B4-E30's same subset
  profile was 11.68, 9.55 ms/expression and 2477 MiB, respectively.
- Native three-pair backward probes reserved 7.740-10.105 GiB for D1 versus
  3.930-6.396 GiB for B4. Not worst-case or full-run measurements.

## 3. Convergence Summary
- All 30 validations completed. Rounded mIoU peaked at 0.508 in epoch 15;
  epoch 30 was 0.503, Acc@0.50 0.552, Acc@0.25 0.601, spmIoU 0.469.
- Training rose quickly and then fluctuated around 0.50; no terminal crash
  is indicated. No epoch-count or validation-parameter tuning follows.
- At the predeclared terminal fixed-subset intervention, learned pairwise
  weights gave 0.49531 versus uniform rival weighting 0.50004. Selection
  changed on 95/441 expressions. Scene-cluster bootstrap 95% interval for
  learned-minus-uniform mIoU was [-0.01889, 0.00909]. This out-of-distribution
  intervention is neither a separately trained uniform control nor proof
  of a negative full-validation effect. At epoch 4 the same intervention
  had the opposite sign (+0.01188), also with an interval crossing zero.
  Raw paired rows: AdaRES-D1_pair_mechanism_16scenes_epoch15_20260925.json.

## 4. Comparison
- vs B1 baseline: +0.0060 mIoU (0.5080 vs 0.5020).
- vs B4-E30 champion: -0.0014 mIoU (0.5080 vs 0.5094).
- vs Phase 1 target: -0.0070 mIoU (0.5080 vs 0.5150).
- D1 is a fresh 30-epoch released-backbone run. B4-E30 used a preserved
  optimizer/scheduler continuation from its 20-epoch checkpoint, not a
  separately fresh seed-matched adaptation control.

## 5. Conclusion
- Full-query pairwise confidence did not beat the champion or meet the target.
  Close this tested route; do not relaunch, scan rival definitions on
  validation, or use the diagnostic uniform intervention as a tuned model.
- The fixed-subset terminal evidence does not support a learned-weight gain
  over uniform rivals, despite improved mean oracle rank (14.04 vs 16.61).
  A nonlocal causal claim would require an independently predeclared,
  trained matched control. No such candidate passes admission this tick.
================================================================================
