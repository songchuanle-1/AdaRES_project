# AdaRES-W2 Result Analysis

## 1. Basic Info
- ID: AdaRES-W2; config: `configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml`.
- Server: server-2 GPU0. Started 2026-09-27 03:22:18 Asia/Shanghai. Last checkpoint recorded 2026-09-28 00:08:33 Asia/Shanghai; monitor reconciled `done_complete` on 2026-09-28 10:46:26 Asia/Shanghai.
- Native run completed 30/30 validations; checkpoint remains on server-2.

## 2. Best Metrics
| Metric | Value | Epoch |
|---|---:|---:|
| mIoU | 0.5100 (unrounded 0.5100098848) | 22 |
| Acc@0.25 | 0.611 | 22 |
| Acc@0.5 | 0.563 | 22 |
| spmIoU | 0.474 | 22 |
| Active queries, latency, peak memory | unavailable | unavailable |

## 3. Convergence Summary
- Best reached at epoch 22 of 30; final epoch mIoU 0.5016716719. There are eight subsequent validation epochs without a new best.
- The 30-epoch training log is preserved under `Checkpoints/server_2/adares_w2_pointmass_competition_res/20260926_145041/20260926_145041.log` (SHA-256 `32a371448b26d6e498a1f00868de92d93fdef9ec0943b77471fa023e79752f00`). Automatic log pull was unavailable; this one log was copied directly from the experiment's remote checkpoint directory.

## 4. Comparison
- Versus AdaRES-B1 (0.5020): +0.0080 mIoU.
- Versus the completed AdaRES-B4-E30 champion (0.5094): +0.0006 mIoU at reported precision. The B4 extension is not a fresh seed-matched 30-epoch control.
- Versus the fixed goal (0.515): -0.0050 mIoU.

## 5. Conclusion
- W2 completed and narrowly improved the recorded champion, but the goal is unmet. The small difference is not a significance or causal claim without matched repeats and scene-level uncertainty.
- Preserve this result; do not relaunch W2 or tune its count transform on validation. Further training requires a separately motivated, six-gate independent candidate.
