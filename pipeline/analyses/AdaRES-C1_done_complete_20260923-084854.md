# Experiment Result Analysis - AdaRES_project
Experiment: AdaRES-C1 | State: done_complete

## 1. Basic Info
- Config: configs_song/model_paper5/adares_c1_listwise_expected_mask_res.yaml.
- Server: server-2 | GPU: 0 | Budget: completed 10/10 epochs.
- Started 2026-09-22 22:31:45 +08:00; terminal recorded 2026-09-23 08:10:22.
- Recorded interval 9h 38m 37s includes reconciliation delay, not measured runtime.
- Warm-start: B4-E30 best checkpoint, mIoU 0.5094.
- Evidence: pipeline/experiments/AdaRES-C1.json;
  pipeline/reports/summary_20260923-081039_manual.md;
  pipeline/decisions/20260923-081618_manual_batch.md.
- Recorded evidence only; no new remote log or checkpoint inspection this tick.

## 2. Best Metrics
| Metric | Value | Epoch / evidence |
|---|---:|---|
| mIoU, terminal report | 0.4950 | 10, manual terminal report |
| best_miou, checkpoint metadata | 0.4946361 | epoch-10 last.pth, manual decision |
| Acc@0.5 | 0.540 | 10, manual terminal report |
| Acc@0.25 | unavailable | not in inspected C1 evidence |
| spmIoU | 0.462 | 10, manual terminal report |
| Active queries | unavailable | not retained |
| Latency | unavailable | not retained |
| Peak memory | unavailable | not retained |

Reported and checkpoint mIoU differ by 0.0003639. Preserve both evidence
precisions; do not claim that reported 0.4950 is exact checkpoint precision.

## 3. Convergence Summary
- Manual terminal report identifies final epoch 10 as best and records both
  best and last checkpoint saves. Original fixed 10-epoch training completed.
- Reported transient mIoU 0.020 remains unexplained; full trajectory not
  reverified. Earlier memory's 0.4917 was a nonterminal best observation.
- Epoch-10 checkpoint was subsequently resumed as AdaRES-C1-E30.

## 4. Comparison
- Versus B1 0.5020: reported delta -0.0070.
- Versus B4-E30 0.5094: reported delta -0.0144.
- Versus C2 10-epoch adaptation 0.4980: reported delta -0.0030.
- Target gap to 0.5150: 0.0200 using terminal reporting.
- No matched B4-only adaptation control: parent-checkpoint comparisons do not
  isolate loss effects. C1/C2 E30 add 30 adaptation epochs after B4 warm-start
  training and are not total-compute-matched to B4-E30.

## 5. Conclusion
- Original C1 failed to improve the champion under its tested protocol.
- Retain B4-E30. No causal claim about ranking versus mask degradation or the
  transient collapse; no generalization claim from this single run.
- The later manual decision at 08:16:18 authorizes fixed C1/C2 continuation
  through epoch 30 with optimizer/scheduler restored. Leave these runs intact;
  do not count them as newly admitted autonomous structural variants.
- Defer final continuation-route judgment until terminal evidence. No further
  epoch extension, weight/margin/threshold scan, new candidate or bugfix here.

## 6. Crash Info (if crashed)
- Not applicable: done_complete; no listed crash.
