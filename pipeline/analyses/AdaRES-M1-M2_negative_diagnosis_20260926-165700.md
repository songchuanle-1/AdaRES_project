# Fixed native screens: seed containment and 2D appearance support

2026-09-26 16:57 Asia/Shanghai. B4-E30 best checkpoint, first 16 ScanRefer validation scenes, 441 expressions, 240 material selection errors (oracle minus selected mask IoU > 0.05). Read-only server-2 GPU1 inference. Raw paired rows: `AdaRES-M1_native_16scenes_20260926.json` and `AdaRES-M2_native_16scenes_20260926.json`.

| Route | Predeclared mechanism | Paired evidence and standalone control | Decision |
|---|---|---|---|
| M1 seed containment | A query whose mask retains its own sampled seed should be more credible than a displaced query. This tests mask membership, not F3's distance to seed or G1's target label. | Oracle support higher in 48/240 material errors. Mean selected 0.97223 versus oracle 0.61828. Support-only mIoU 0.43714 versus B4 indicator 0.49186. | Reject directional mechanism. No threshold or fusion scan. |
| M2 2D appearance support | The predicted mask's pooled projected 2D appearance should agree with its own sampled seed appearance, supplying information beyond the 3D decoder coherence of L1. | Oracle support higher in 80/240. Mean selected 0.95035 versus oracle 0.91840. Support-only mIoU 0.11752 versus B4 0.49186. | Reject directional mechanism. No feature or fusion scan. |

The M2 diagnostic hook initially used decoder input index 7 and raised `IndexError`; corrected to index 6, then reran the unchanged test successfully. Python syntax checks passed. The tests use already fused model features and are observational fixed-subset diagnostics, not population efficacy estimates. Both routes fail the diagnosis hard gate. No candidate-specific backward, resource profile, independent model package, admission receipt, or dispatch is claimed.

One targeted G1 live-log check reached Val[30/30] and printed mIoU 0.497, Acc@0.5 0.542, Acc@0.25 0.602, spmIoU 0.465 and a last checkpoint. The prepared queue remains running with deathwatch 2/3; terminal reconciliation and best metric belong to the next scheduled monitor.
