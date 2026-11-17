# AdaRES-F4 native six-gate review

2026-09-24 15:07 Asia/Shanghai. Server-2 GPU1, independent committed
F4 package and exact B4 config copy, native `adares_project` Python.

| Gate | Evidence | Assessment |
|---|---|---|
| Diagnosis | Fixed 16-scene F4 analysis: 113/240 missed-oracle center neighborhoods increase coverage, 16 decrease; means 0.14642 to 0.23379 | Bounded proxy pass, not causal mIoU evidence |
| Controls | Configs are byte-identical (SHA256 1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03); same B4 package files apart from decoder and label | Fresh 30 epoch B4-E30 protocol control; historical B4 continuation from epoch 20 is not a separately fresh 30-epoch seed-matched run |
| Native backward | F4 training pairs [0,1], [2,3], [8,9], [256,257]: finite losses and nonzero finite indicator/decoder gradients; edge tests verify padding invariance, predicted foreground anchor and empty foreground seed fallback | Pass for tested batches |
| Resources | On [256,257], F4 and B4 both reserve 6.396 GiB, run 23.67s vs 24.25s; [0,1] both reserve 4.75 GiB, 19.21s vs 19.72s | Pass bounded 24 GiB single GPU; no claim about dataset maximum or steady throughput |
| Independence | F4 reroutes fine cross-attention and query self-attention after layer 0 via prior mask centroid; D1 modifies confidence across all query pairs | Pass |
| Constraints | Unique `adares_f4_mask_reanchor_res.yaml`, released backbone, fresh seed 1999 and 30 epochs, server-2 GPU1 only, no validation tuning | Pass |

F4 native JSON stdout values: [0,1] loss 13.5044, indicator gradient 2.4792,
out_norm 0.6379, reserved 4.750 GiB; [2,3] loss 5.9053,
gradients 0.5991/0.2224, reserved 4.693 GiB; [8,9] loss 6.9760,
gradients 0.8499/0.2844, reserved 4.699 GiB; [256,257] loss 11.1013,
gradients 1.6587/0.9167, reserved 6.396 GiB. B4 [256,257] loss
11.0877, gradients 1.6584/0.9118, reserved 6.396 GiB.

Code was additively deployed with `scripts/sync.sh server-2`: eight committed
files hash-verified and remote `.codex_source_commit` matches
597e56a852f1039bc44e72ffcdba6f1a363e07bd. GPU0 D1 was untouched.
Training's maximum memory and final mIoU remain unmeasured. A process
alone is not proof of live steps; verify Train[1/30] advancement.
