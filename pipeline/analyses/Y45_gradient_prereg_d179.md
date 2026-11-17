# SAME Y45 Native Gradient Attribution

- Continue stable AdaRES-Y45; do not modify model/config/checkpoint or gates.
- Freeze the same prediction-blind TRAIN8 indices0,44,60,101,115,126,137,171;
  released-init first4, seed1999; capture actual decoder/criterion inputs/RNG.
- Differentiate legacy total without the new likelihood, and new categorical
  likelihood, through all four new tensors and native three-category logits.
- Correct, same-scene cyclic-wrong truth and collapsed FP/FN targets use the
  same cached masks/features/matching labels. Repeat the correct case.
- Record signed inner products/norms at parameter and query-logit levels;
  matched/unmatched groups, actual target IoU, and every adverse scene remain.
  No minimum cosine/effect-size threshold or outcome-selected cohort.
- Verify the analytic matched-label derivative along quality log odds
  [1,-softmax(zFP,zFN)] and categorical derivative(p-q)/(B*Q). Floating
  comparison uses an a-priori operation-count8*float32eps error envelope;
  no fitted tolerance. Report algebraic errors, not global determinism.
- Three-category likelihood equals binary IoU likelihood plus the conditional
  FP/FN likelihood weighted by1-qTP. Verify values and gradients in native
  captures. This is a factorization test, not a new weighted model.
- Opposing gradients explain a local multi-objective tradeoff, not efficacy,
  harm or an automatic rejection/PASS. Six-gate review must retain limitations.
- No balancing-weight tuning, validation search, scalar-ranking rename,
  diagnostic weight deployment or new training is permitted by this test.
