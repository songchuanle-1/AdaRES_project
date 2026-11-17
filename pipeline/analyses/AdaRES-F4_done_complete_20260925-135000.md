================================================================================
  Experiment Result Analysis - AdaRES_project
  Experiment: AdaRES-F4 | State: done_complete
================================================================================

## 1. Basic Info
- ID: AdaRES-F4
- Config: configs_song/model_paper5/adares_f4_mask_reanchor_res.yaml
- Server: server-2 | GPU: 1
- Started: 2026-09-24 15:10:27 +08:00 | Reconciled: 2026-09-25 13:47:04 +08:00
- State: done_complete; 30/30 validations and final last.pth save recorded.
- Sources: pipeline/experiments/AdaRES-F4.json and remote log
  /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_f4_mask_reanchor_res_20260924_151020.log.

## 2. Best Metrics
| Metric | Value | @ Epoch |
|---|---:|---:|
| mIoU | 0.5060 | 21 at rounded stdout precision |
| Acc@0.50 | 0.556 | 21, rounded stdout |
| Acc@0.25 | 0.611 | 21, rounded stdout |
| spmIoU | 0.472 | 21, rounded stdout |

- The terminal record gives best mIoU 0.5060. The epoch assignment and
  auxiliary metrics use rounded stdout, not a loaded best checkpoint.
- Active queries, full inference latency, and training peak memory were not
  measured at the terminal checkpoint. Earlier bounded native resource probes
  are not full-run measurements.

## 3. Convergence Summary
- All 30 validations completed. Rounded best 0.506 occurred at epoch 21.
- Epoch 30: mIoU 0.499, Acc@0.50 0.549, Acc@0.25 0.599, spmIoU 0.465.
- The series rose to approximately 0.50 and fluctuated or declined after
  its epoch-21 peak. No crash is indicated by the terminal record.

## 4. Comparison
- vs B1 baseline: +0.0040 mIoU (0.5060 vs 0.5020).
- vs B4-E30 champion: -0.0034 mIoU (0.5060 vs 0.5094).
- vs Phase 1 target: -0.0090 mIoU (0.5060 vs 0.5150).
- B4-E30 is the existing 30-epoch protocol comparison, but its epoch-20
  continuation preserved optimizer state; the protocol is not a separate
  freshly trained seed-matched control.

## 5. Conclusion
- F4 did not improve the champion or reach the target. Close the tested
  predicted-mask centroid reanchoring route; do not tune neighborhood,
  thresholds, loss weights, or epochs on validation.
- The 16-scene foreground-coverage proxy supported testing but did not
  predict improvement over B4-E30. Continue D1 monitoring independently.
================================================================================
