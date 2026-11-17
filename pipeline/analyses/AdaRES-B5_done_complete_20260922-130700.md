================================================================================
  Experiment Result Analysis - AdaRES_project
  Experiment: AdaRES-B5 | State: done_complete
================================================================================

## 1. Basic Info
  - ID: AdaRES-B5
  - Config: configs_song/model_paper5/adares_b5_token_grounding_res.yaml
  - Server: server-2 | GPU: 1
  - Started: 2026-09-21 19:32:33 +08:00 | Finished: 2026-09-22 09:18:09 +08:00
  - Duration: 13h 45m 36s
  - State: done_complete (20/20 validations)

## 2. Best Metrics
| Metric | Value | Epoch |
|---|---:|---:|
| mIoU | 0.5030 | not captured |
| Acc@0.50 | unavailable | - |
| Acc@0.25 | unavailable | - |
| spmIoU | unavailable | - |

The terminal experiment record and ledger preserve only the final best mIoU and
validation count. No local or currently reachable remote training log was
available for this tick, so missing metrics and the best epoch are not inferred.

## 3. Convergence Summary
  - Final protocol: 20/20 validations completed.
  - Best epoch and post-best no-improvement count: not captured.
  - Convergence shape: unavailable from the retained terminal evidence.
  - The run completed normally; this is not a crash or early-stop result.

## 4. Comparison
  - vs AdaRES-B1 baseline: +0.0010 mIoU (0.5030 vs 0.5020).
  - vs AdaRES-B3: +0.0016 mIoU (0.5030 vs 0.5014).
  - vs AdaRES-B2: -0.0020 mIoU (0.5030 vs 0.5050).
  - vs champion AdaRES-B4: -0.0040 mIoU (0.5030 vs 0.5070).
  - vs project target: -0.0120 mIoU (0.5030 vs 0.5150).

## 5. Conclusion
  - Conclusion: failed to improve the champion; the token-grounding hypothesis
    is not supported by this one-seed result.
  - The result is above the faithful baseline but below B2 and B4, so direct
    query-to-phrase token compatibility did not close the target gap.
  - Do not launch a renamed duplicate or tuning-only B5 successor. Retain the
    possibility that token evidence is partly redundant with existing decoder
    language attention as a limitation, not an established causal conclusion.

================================================================================
