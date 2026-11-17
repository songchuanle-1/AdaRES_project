# AdaRES-D1 Six-Gate Review

2026-09-24 08:50 Asia/Shanghai. This is a review of bounded evidence, not an admission receipt or launch.

| Gate | Assessment | Evidence and limitation |
|---|---|---|
| Diagnosis | Pass | B4-E30 fixed 16-scene/441-expression diagnostic: 240 material selection errors, 191 selected/oracle pairs with overlap <0.5. Nonlocal comparison is testable; subset is not causal or full-validation evidence. |
| Controls | Pass for fresh run | D1 and B4 config files are byte-identical: seed 1999, released SPFormer pretrain, 30 epochs, batch size 2, data, optimizer, schedule, inference. B4-E30 completed the protocol via epoch-20 continuation preserving optimizer/scheduler. This is an existing protocol control, not a warm-start adaptation control. No D1 warm start is proposed. |
| Native backward | Pass | D1 full-model loss and finite nonzero pairwise, indicator, decoder gradients on three native ScanRefer batch pairs; padded-mask invariance and competitor sensitivity in module test. |
| Resources | Pass with bounded risk | Peak reserved 8.443, 7.740, 10.105 GiB on tested pairs, under server-2's 24 GiB GPU capacity; fixed-subset inference 3570 MiB allocated and 12.69 ms/expression for untrained D1. Worst-case training memory and trained-model latency remain unknown; monitor OOM and measure final metrics. |
| Independence | Pass | Dedicated package and unique config; B4 uses overlap-normalized context, while D1 learns query-to-query pair scores across all rivals using latent and mask evidence. Not C1/C2 ranking loss, scalar scan, seed repeat or pure ablation. |
| Goal constraints | Pass for proposed experiment | server-2 single GPU 0 or 1 only; fixed fresh 30 epochs, released backbone, no validation tuning or banned checkpoints. Report mIoU, Acc@0.25, Acc@0.5, active queries, latency and peak memory after training. |

Local syntax compile and B4/D1 config hash parity passed this tick. Native evidence is recorded in the three D1 probe analyses; local Python has no PyTorch, so native tests were executed earlier on server-2.

**Dispatch status: blocked, not admitted.** D1 package/config are uncommitted. The project lacks `pipeline/closure_policy.json` and `scripts/sync.sh`; its legacy `remote_train.sh` archives only committed `HEAD`. A targeted read-only SSH check found that the server-2 production code directory is not a Git checkout, so closure safe-sync cannot verify a remote commit there. Queueing now would use code without D1. Do not enqueue or invoke monitor again until candidate code is committed without sweeping unrelated changes, a non-destructive verified deployment route and remote source identity are established, and a hashed receipt matches the admitted commit. Use the monitor dispatch path and verify actual training steps beyond PID.

Full theoretical derivation and literature novelty are not claimed. Do not create a second filler variant in event mode.
