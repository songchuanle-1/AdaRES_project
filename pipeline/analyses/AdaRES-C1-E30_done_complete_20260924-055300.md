# Experiment Result Analysis - AdaRES_project
Experiment: AdaRES-C1-E30 | State: done_complete

## 1. Basic Info
- ID: AdaRES-C1-E30
- Config: configs_song/model_paper5/adares_c1_listwise_expected_mask_e30_res.yaml
- Server: server-2 | GPU: 0
- Resumed 2026-09-23 08:16:41 +08:00 from C1 epoch-10 last.pth with optimizer and scheduler.
- Terminal reconciliation: 2026-09-24 05:47:02 +08:00; 30/30 validations.
- Sources: pipeline/experiments/AdaRES-C1-E30.json and its remote_log field. The recorded launch-to-reconciliation interval is not measured training runtime.

## 2. Best Metrics
| Metric | Value | @ Epoch |
|---|---:|---:|
| mIoU | 0.5008 | not established from terminal record |
| Acc@0.25 | unavailable at best checkpoint | - |
| Acc@0.50 | unavailable at best checkpoint | - |
| spmIoU | unavailable at best checkpoint | - |
| Active queries | unavailable | - |
| Inference latency | unavailable | - |
| Peak memory | unavailable | - |

## 3. Convergence Summary
- The original C1 first 10 epochs reached 0.4950 in terminal reporting; the fixed continuation improved its best to 0.5008 after all 30 validations.
- The terminal record does not specify the exact best epoch or aligned auxiliary metrics. Do not infer them from rounded validation stdout.

## 4. Comparison
- vs B1 baseline 0.5020: -0.0012 mIoU.
- vs B4-E30 champion 0.5094: -0.0086 mIoU.
- vs target 0.5150: -0.0142 mIoU.
- A B4-only matched adaptation control was not run; these comparisons do not isolate the effect of C1's listwise loss.

## 5. Conclusion
- Close the tested C1 listwise expected-mask route after its fixed continuation. Keep B4-E30 as champion; do not tune loss weights or extend epochs.
- D1 is a separate unadmitted nonlocal-information candidate and requires its own controls and admission gates.

## 6. Crash Info (if crashed)
- Not applicable.
