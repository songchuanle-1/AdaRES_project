# Y17 Same-Target Description Consistency Preregistration

Frozen2026-09-30 20:07 Asia/Shanghai, before new native predictions.
Stable ID AdaRES-Y17, hypothesis same-target-description-mask-consistency.

## 1. Motivation

Goal: improve standard-indicator ScanRefer segmentation through metric-aligned
evidence, mIoU>=.515. Different genuine annotations for the same object
provide target-equivalent supervision without masking semantic content.
This differs from rejected Y11 disjoint-object exclusivity and O1 token removal.
No claim that all same-target descriptions are linguistic paraphrases.

## 2. Current Best Results

W2 .5100, B1 .5020, Y15 singleGPU30-epoch native training. Y16 global
superpoint-resolution deficit screen negative. No new validation result.

## 3. Analysis And Direction

Can native predictions disagree across target-equivalent annotations beyond
same-expression numerical repeat noise? If so, a scene-local paired training
path could learn consistent spatial support without teacher inference.
For soft spatial predictions p_a,p_b use scene-point-weighted symmetric
Bernoulli Jensen-Shannon divergence, m=(p_a+p_b)/2,
JS=mean_n[.5 KL(Ber(p_a)||Ber(m))+.5 KL(Ber(p_b)||Ber(m))].
Zero for identical distributions; gradients act on both descriptions.
Matching positive labels alone is not proof consistency improves accuracy;
wrong but consistent masks and loss tradeoffs remain risks.

## 4. Experiment Plan

Freeze first16 lexicographic actual TRAIN scans having two different object
tuples, each with two different exact annotation token strings.
Select first two object tuples, first two distinct token annotations each.
Four expressions/scene, no augmentation, training=False, native W2
best checkpoint, seed1999. Save manifest BEFORE any model outcomes.
Run independent repeated native forwards; match identities exactly.
Record selected-mask IoU for each same-object pair and disjoint-object
controls, per-description selected-vs-oracle GT gap, cross-description
candidate-pool oracle gain and repeated-forward mask/selection flips.
No query-index correspondence across descriptions is assumed.
Reuse continuous probabilities and point mass for native JS; also compare
cyclic different-target pairs in the exact same scene. Count repeats and
scene means; no tiny-cohort significance or fixed effect-size admission rule.
Positive diagnosis requires actual same-target discrepancy exceeding
measured repeats; if absent reject, if direction ambiguous preserve pending.

Only then design physical paired-training package: matched native GT losses,
same-target pairing versus different-target/permuted pairing, all new paths
finite backward, full-scene time/memory and goal constraints. The study does
NOT license forced consistency at test time or full training before six gates.

## 5. GPU Allocation Plan

Only live-idle server-2 GPU1, exclusive existing diagnostic flock.
GPU0 Y15 untouched. Bounded240second diagnostic and current real deadline.

## 6. Implementation Checklist

AST and source hash; frozen exact cohort; repeat/control alignment;
point-weighted probabilities; retain all rows/errors; no validation loading.

## 7. Risk Assessment

Train checkpoint may already be consistent. Disjoint pairs are specificity
controls, not equivalent-target controls. Both same-target annotations may
be wrong; candidate oracle uses labels for diagnosis only. Paired training
cost/real objective integration and literature novelty unverified.

## 8. Smell Check

The prospective graph connects real descriptions of one object during
training, not a scalar or score mixture. No coefficient or feature scan.
