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

- 20-epoch training running on server-2 GPUs 0 and 1, seed 1999, effective
  batch size 2, grad clip 0.1, AdamW lr 1e-4 (the published recipe).
- Smoke test before launch: finite loss and gradients, full predict keys.
- Target: ScanRefer mIoU >= 0.515.
