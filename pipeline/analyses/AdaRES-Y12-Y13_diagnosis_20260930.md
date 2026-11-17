# AdaRES Y12/Y13: cross-model confidence and mask-space alignment

2026-09-30 Asia/Shanghai. Foreground design lease
`fd4a6799f51d41be83e15b6668f35e8d`, first 32 non-augmented ScanRefer validation
scenes (851 expressions), seed 1999, W2 and B4-E30 best checkpoints unchanged.
Both runs used server-2 GPU0, repeated forward, and diagnostic point IoU rather
than full ScanRefer mIoU.

| ID | Frozen intervention | Result | Disposition |
|---|---|---|---|
| Y12 raw-index committee | Standardized W2 + B4 indicator margins at identical query indices; eight B4-index permutations | W2/committee/control mean IoU 0.52963/0.48556/0.42300. Committee-minus-W2 scene paired delta `-0.03611`, 8 positive, 23 negative, 1 tie, `p=0.01067`. Repeat identity mismatch 0. B4 query evaluated on W2 masks was only 0.33277. | Reject. Equal query indices do not establish semantic correspondence across independently trained models. |
| Y13 mask-aligned committee | Hungarian one-to-one assignment maximizing W2-B4 binary-mask IoU, then standardized W2 + aligned B4 score; eight cyclic assignment controls | W2/committee/control mean IoU 0.52963/0.53909/0.45401. Hungarian/cyclic matched-mask similarity 0.21915/0.04485. Committee-minus-W2 scene paired delta `+0.00521`, 21 positive, 11 negative, `p=0.11018`; committee-minus-control `+0.07310`, `p=1.93e-5`. One repeated-forward identity mismatch. | Reject under the frozen rule. The positive direction and strong match-specific control justify a separately trained reliability gate, not a coefficient scan of Y13. |

No candidate passed all four diagnostic conditions, so no inference module was
admitted, no native backward or resource receipt was claimed, and no training
was queued. Y14 below is a pending design only.
