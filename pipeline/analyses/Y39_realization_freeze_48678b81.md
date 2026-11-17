# AdaRES-Y39 Fixed Realization

Frozen after CPU/source and TRAIN partition diagnosis, before native model
response/backward/resources. No validation outcomes or numeric search.

Compute centered population variance over ALL32 native point backbone feature
channels inside each superpoint before native mean pooling. Preserve the
existing pooled mean, masks, standard losses,128 queries and one argmax.
Replicate variance over expressions exactly as existing scene feature expansion.
Only the final indicator adds the new readout; all auxiliary indicators stay W2.

Readout: log1p(nonnegative variance), shared32->256->256 point encoder with
GELU. Aggregate point-mass soft-foreground and soft-background encoded features;
concatenate original query, foreground-minus-background and their midpoint.
Fixed768->256->2 GELU head, eight new trainable tensors. Masks are detached
inside this added path; no GT, coordinates, feature subsets or learned
thresholds. Width256 is the native decoder width, not a searched value.

Controls: enabled vs disabled physical W2, shuffled variance-to-superpoint
pairing, wrong mask pairing, repeat, joint superpoint permutation, padding
poisoning, mask detach, zero variance and raw point permutation. Require
finite/nonzero native gradient on every new tensor; preserve final masks,
sampling and non-indicator losses. Investigate adverse native losses before
any gate review; tiny cohorts do not estimate population efficacy.

First8 distinct TRAIN scenes and first4 from released-backbone initialization,
seed1999. Trained W2 is a diagnostic checkpoint only, never training pretrain.
Native TRAIN largest-pair728/6780 batch2 stress, separate inference repeats,
parameter/memory/time comparison and optimizer/scheduler resume are pending.

Important negative diagnosis: first64 TRAIN scenes/256 expressions have
mean exact whole-superpoint ceiling0.9997711704, median1,29 nonzero gaps.
Do not claim quantization explains the .515 target deficit. Point variance
may be indirectly represented by backbone means; mechanistic necessity and
the proposed confidence association remain unproved.

Post-training: fixed30/seed1999 from released SPFormer, frozen ScanRefer
full9508/141 twice EACH candidate/W2/B1, FIRSTprimary, scene10000bootstrap1999,
TRAINcuts1487/3325. Subgroup material harm if CIupper<0 versus either.
Report empirical repeat noise and trained paired cost; target.515 unchanged.
No admission/PASS yet, and no claim of external literature novelty.
