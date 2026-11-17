# AdaRES-D1 live diagnosis

2026-09-24 14:21 Asia/Shanghai. Read-only inspection while D1 continues
training on server-2 GPU0. The fixed-subset checkpoint probe ran on GPU1.

## Health and identity

- Production PID 32442 was alive at Train[5/30], iteration 2278/4686.
  GPU0 used 23808/24576 MiB; GPU1 returned to idle (1 MiB).
- The remote `.codex_source_commit` equals the D1 admission commit
  `d7064cd697e651eeb79b2522e10c44c0f55edb15`. D1 and B4 YAML files
  have the same SHA256; both use the same seed, data, training and inference
  protocol. No skipped unstable backward, CUDA OOM or traceback was found.
- The D1 best checkpoint metadata is epoch 4, best mIoU 0.44097984; the
  checkpoint includes model, optimizer and scheduler.

## Full validation, matched training epoch

| Epoch | B4 mIoU | D1 mIoU |
|---:|---:|---:|
| 1 | 0.318 | 0.354 |
| 2 | 0.373 | 0.407 |
| 3 | 0.423 | 0.424 |
| 4 | 0.440 | 0.441 |

Numbers above are rounded training-log validation summaries. Epoch 4 D1
Acc@0.50 0.490, Acc@0.25 0.548, spmIoU 0.406. B4 epoch 4:
0.482, 0.552, 0.405 respectively. There is no evidence of early
optimization collapse, but early parity does not establish a final gain.

## Fixed-subset selection diagnosis

The same first 16 validation scenes and 441 expressions were evaluated
without updates using the D1 epoch-4 best checkpoint and the archived B4-E30
best-checkpoint diagnostic. The predictor and threshold/IoU computation were
copied from the B4 diagnostic; only the model package differs.

| Measure | B4-E30 final best | D1 epoch 4 |
|---|---:|---:|
| Top-1 mask mIoU | 0.49186 | 0.41114 |
| Best available query (oracle) mIoU | 0.87316 | 0.83894 |
| Material selection errors (oracle - top-1 > 0.05) | 240/441 | 293/441 |
| Oracle query among top 3 in material errors | 20 | 22 |

Paired per-expression D1-minus-B4 differences: top-1 -0.08072, oracle
-0.03423, and oracle-to-top-1 gap +0.04649. Top-1 improved by >0.05
for 57 expressions and worsened by >0.05 for 139. Of B4's 240 material
errors, D1 no longer has a material error on 44, retains one on 196,
and introduces 97 on examples where B4 had none.

The oracle computation uses ground truth **only for diagnosis** and is not
an inference method or reported validation result. B4 is fully trained
while D1 is only at epoch 4; these differences cannot be attributed to
architecture without a matched-epoch or completed-protocol comparison.
The fixed 16 scenes are not the full validation set. The fresh D1 model
can change both candidate mask quality and query ranking, so a lower
top-1 score alone cannot identify which component caused the change.

## Decision and next measurement

- Leave D1 running under its admitted protocol and existing pipeline
  early-stop rule. Do not tune loss weights, thresholds or epochs from this
  validation subset, and do not enqueue another candidate based on it.
- At a prospectively fixed later checkpoint or at the terminal result,
  repeat the same subset with the same predictor and compare full-validation
  mIoU against B4-E30. Separately report the oracle ceiling and selection
  gap; measure trained D1 latency/peak memory before claiming efficiency.
- Evidence: `pipeline/analyses/AdaRES-D1_selection_16scenes_epoch4_20260924.json`,
  `pipeline/analyses/AdaRES-B4-E30_selection_16scenes_20260924.json`,
  the remote D1 training log and the D1 checkpoint metadata.

No model weights, training config, queue, scheduler or running process were
modified. The read-only predictor script was staged in the project tools
directory on server-2 and is present locally as
`AdaRES/tools/diagnose_adares_d1_selection.py`.
