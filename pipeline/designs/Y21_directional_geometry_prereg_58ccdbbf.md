# Y21 Directional Geometry Counterfactual

Frozen before outputs2026-10-02; stable candidate AdaRES-Y21.

- Hypothesis: native training universally applies XYflip/arbitrary yaw but leaves ScanRefer direction-bearing text unchanged. This can disproportionately invalidate directional grounding; orientation-preserving augmentation conditioned on TRAIN text is a candidate only if a native counterfactual separates this mechanism from generic geometric sensitivity.
- No validation used. Frozen W2 best, seed1999, first64distinct train scan IDs, first32expressionchunk, no new object-target selection. Inference-only no optimizer/model updates.
- Word membership frozen case-insensitive exact tokens left/right/front/behind/back, with same grouping applied to all outputs. This is a noisy proxy; view dependence is unknown and camera-relative language need not be tied to global axes.
- Controls: native raw input/repeat; fixed yaw180degrees about Z; puretranslation[1,1,0]. Point identities, GT masks, superpoints, token/annotation identities must remain equal. Each state built from separately loaded raw arrays; transform_test receives copied xyz.
- No transformation or threshold scan. Rotation is a prospective geometric counterfactual, not the exact full training augmentation distribution.
- Native metrics: selected mIoU baseline/rotation/translation, selected-query flips, probabilities repeated numerical noise, per-expression absolute and signed mIoU difference. Scene is aggregation unit.
- Primary: per-scene paired rotational degradation direction-minus-nondirection in scenes containing both groups, and the same translation contrast. Require a reproducible positive primary direction contrast beyond the corresponding translation contrast before proposing dataset intervention. If mixed/negative contrast, reject this specific disproportionate-direction hypothesis; don't subset/tune words.
- Even positive diagnostic is not trained efficacy or a license to remove arbitrary augmentation. Independent production dataset/native loss/backward/resources/admission/frozen30epochs required; orientation/camera relation is an explicit causal uncertainty.
