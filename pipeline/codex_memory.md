# AdaRES_project Research Memory

## Identity

AdaRES-B0 is a self-authored baseline, not a wrapper around an existing
checkpoint. Code lives in `AdaRES_project/AdaRES` and the working remote tree
is `/bak_home/songcl/Project/AdaRES_project/AdaRES`.

## Design

- Encoder: released SPFormer sparse U-Net backbone, loaded with a
  shape-compatible filter (the released file carries legacy criterion
  buffers that break a plain strict=False load).
- Text: RoBERTa token features, projected and normalized.
- Decoder: 6 layers of language self-attention, language-scene cross
  attention, query-scene cross attention, query self-attention and FFN.
- Occupancy field: a language-conditioned per-superpoint target occupancy
  estimate, independent of the query set.
- Query score: the point-weighted soft IoU between the query's own mask and
  that occupancy field. The score enters the indicator logit directly, so
  ordering is preserved and gradients stay bounded.
- Loss: Hungarian matching BCE/Dice plus occupancy BCE and a pairwise
  logistic ranking loss on the true per-query IoU.

## Engineering Fixes Applied

- Shape-compatible pretrain loading for the released backbone checkpoint.
- 0-dim loss terms reshaped to [1] before DDP reduce_dict.
- Logger guarded on non-main DDP ranks.

## Current Status

- AdaRES-B0 completed 20 epochs at mIoU 0.333 and is closed as a low-metric
  baseline revision. Its masks were strong (all-query oracle 0.847 on 124
  validation expressions) but its metric-aligned ranking score was poor.
- AdaRES-B1 completed 20 epochs. Best ScanRefer mIoU is 0.502 at epoch 18,
  with Acc@0.50 0.553, Acc@0.25 0.610, and spmIOU 0.467. It is now the
  official project baseline and is 0.013 below the target.
- The original B2 watchdog stopped safely because the smoke harness expected
  B0-only `occupancy_loss` and `iou_ranking_loss` keys. The harness was made
  model-agnostic and the real-batch smoke then passed.
- AdaRES-B2 metric-calibrated indicator is training on server-2 GPUs 0 and 1.
  Launcher PID 36242; log `train_adares_b2_20260919_204142.log`.
- AdaRES-B3 mask-evidence indicator is scientifically admitted and waiting.
  It changes the confidence information path, not a numeric setting: each
  query pools foreground and complementary background scene features through
  its own soft mask. Module gradients and a real ScanRefer scene smoke passed.
  The conditional watcher launches B3 only if B2 finishes below mIoU 0.515.
- This Codex thread has an active native background goal. It automatically
  continues the research loop without user prompts or an hourly scheduler:
  monitor the current run, perform terminal analysis and evidence-based repair,
  and maintain one scientifically admitted successor candidate.
- Target: ScanRefer mIoU >= 0.515.
