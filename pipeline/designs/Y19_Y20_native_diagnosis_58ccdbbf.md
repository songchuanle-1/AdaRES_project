# Frozen Training-Only Diagnosis - AdaRES-Y19 / AdaRES-Y20

Frozen before new diagnostic outputs, owned tick58ccdbbf48104660911c2a0c0fc97d3e,2026-10-02Asia/Shanghai.

## Questions And Controls
- Y19 proposed all-query exact-point-IoU indicator supervision. G1 already trained all-query mask-IoU soft targets and failed30epochs at.4980; Y5/Y6/Y7 rejections preserved. Execute G1's actual extracted loss_indi on synthetic equal-point-per-superpoint and unequal-point cases versus independently point-expanded targets. Record whether the change is only target measure/weight calibration; no relaunch or tuning of G1 under a new name.
- Y20 distinct hypothesis: W2's QueryCompetitionIndicator has a differentiable mask-overlap path into mask generation. Indicator classification supervision can oppose the native segmentation objective through this path. Detaching only the masks supplied to the competition indicator changes backward routing, not any native forward/inference values, query count, masks, thresholds or indicator architecture.
- Y20 matched controls: same frozen W2 checkpoint, same native train batch/matches/loss weights, native route versus masks.detach() at the indicator input; original masks remain differentiable for their native segmentation and other paths.

## Prediction-Blind Cohort
- Train split only, seed1999, first64 distinct scan IDs encountered in native dataset order, firstchunk per scan, four expressions maximum; original training augmentations/elastic configuration retained.
- Native training loss is actually executed with model.train(), no optimizer updates. Repeat native forward with identical Python/NumPy/CPU/CUDA RNG and BN buffers restored; report numerical differences rather than inventing bitwise repeatability.
- Checkpoint: existing frozen W2 best20260926_145041; diagnostics do not change its weights or any active training.
- Observe final indicator input mask logits, final criterion indicator loss and final mask BCE+Dice; compute their autograd gradients with respect to those logits. Use native same weights; no value scan.

## Readouts And Disposition
- Y20 record per-scene normalized gradient cosine, fraction of mutually active mask coordinates with opposite signs, gradient norms/finite status, native repeat values, and detached-route indicator-to-mask gradient (must be absent/zero) with indicator-head gradient preserved.
- Population support requires reproducible nonzero native path and scene-mean negative cosine versus zero routing under the detached control. No validation mIoU or arbitrary effect threshold used for admission. Adverse scenes remain in evidence.
- If all scene means are nonnegative, reject the global conflicting-route hypothesis; do not redefine a favorable subgroup posthoc. If mixed/mean negative, retain all variation, proceed to native backward/resource/independence tests, not an efficacy claim.
- Y19 source-equivalence/calibration-only finding rejects renamed implementation, not the universal usefulness of metric targets.
- Before any Y20 training admission, freeze fullScanRefer/9508expressions/141scenes, W2/B1 comparators, two independent evaluation passes, scan-clustered intervals and train-size1487/3325 group material-harm rule. Params/no-extra-inference path expected identicalW2; measured latency/memory required, not assumed. Fixed30epochs/released SPFormer/one GPU.
- No query expansion, teacher path, Y18 mask BCE/Dice replacement, threshold/temperature/weight tuning or validation-selected repair. Teacher and prior trained failures remain untouched.
