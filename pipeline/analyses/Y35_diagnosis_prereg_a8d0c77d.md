# Y35 Prospective Diagnosis

Stable ID AdaRES-Y35; hypothesis final-mask-evidence-to-language-token-attention.
Foreground lease a8d0c77d45d142c0a6c46b163b956253, after executed Y34 CPU rejection.

## Hypothesis And Distinction

Final confidence can compare the language evidence evoked by the candidate's
foreground and background visual summaries. Each summary queries ALL unpadded
final contextualized language tokens, then their attended-language difference
and the final query predict a confidence residual.
Existing decoder cross-attention is acknowledged. The claim is a direct
final predicted-mask -> visual support -> individual words -> confidence path,
not absent upstream language conditioning or proven semantic necessity.
Y33 instead queries individual visual tokens from the final query.
Y28's rejected realization pools visual features directly; Y35 is discriminated
by equal language first moments but distinct per-token language distributions.
Y30 temporal and Y31 spatial-covariance routes are not reopened.

## Frozen Mechanism

Point-mass-weighted sigmoid-mask FG/BG visual means, padding excluded.
Functional LayerNorm then a shared 256-dimensional query projection; words
use shared key/value projections, sqrt(256) scale, native language padding.
FG/BG attention over words, their value difference concatenated with normalized
query, Linear512->256/GELU/Linear256->2 final residual. No trained scalar,
temperature/width/threshold/loss/feature search. Parent d_model fixes dimension.
No GT, test labels, word subsets or validation choices in the forward path.

## First Tests And Native Cohort

Actual-source CPU32 repeat/padding/support+word+query permutations, detach
forward identity and equal-language-mean different-distribution control.
Prediction-blind first64 distinct TRAIN scenes in native order, <=4 descriptions
per scene: selected/oracle point-IoU gap, positive foreground/background visual
contrast, language token variation, wrong-word and wrong-mask interventions.
Expected intended response nonzero, cached repeat exact, joint permutations
equivariant; not a required task-gain threshold or tiny-cohort significance.
Follow actual cached TRAIN8 and released-init4 disabled/enabled/repeat,
mask-detach/wrong-word/wrong-mask matched controls and native-loss gradients.
Every new trainable tensor must receive finite nonzero gradients. Only final
indicator loss may change with the residual enabled. Preserve all adverse
losses; investigate unexplained harm before admission, never weaken a gate.
Then physical native train two steps, checkpoint model/AdamW/PolyLR resume,
stress/matched parameter/time/memory and independent source verification.

## Protocol And Post-Training Freeze

Independent physical W2 package; released SPFormer backbone; native128 queries;
seed1999, fixed30 epochs, same optimizer/losses/all masks/auxiliary outputs and
one standard argmax. Only GPU0 after live idle/exclusive checks, Y33/GPU1 intact.
If admitted: full9508 expressions/141 scenes Y35,W2,B1, two passes each,
firstpass primary mIoU plus Acc25/Acc50/active queries/latency/peak memory.
Whole-scene bootstrap10000 seed1999; TRAIN rawpointcuts1487/3325.
Material subgroup harm = paired CIupper<0 against either W2 or B1.
CIcrossing0 is inconclusive, not equality or no-harm. Quantify repeat noise
empirically and trained paired cost; target.515 unchanged.
Six gates, smell review, efficacy/necessity/novelty/Pareto remain unknown now.
