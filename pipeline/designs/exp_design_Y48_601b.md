# Experiment Design - AdaRES-Y48

## 1. Motivation
AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515.
SAME Y48 asks whether final confidence supervision should agree on identical
nonempty output masks when the literal point TP/FP/FN is identical.

## 2. Current Best Results
W2 remains scientific comparator. Y45 fixed30 completed with FIRST .50765678
versus W2 .51010516/B1 .50221439; all frozen overall/subgroup intervals cross0.
This is not proof of failure of every error-evidence mechanism.

## 3. Analysis And Direction
The initial one-winner premise was disproved: native matching is one-to-many.
SAME-forward TRAIN8/32 expressions/128queries nevertheless observed31 mixed
native-label nonempty duplicate groups out of171. Equal hard masks have equal
point confusion, not necessarily equal soft masks, query states or usefulness.
The actual-label incidence supports a falsifiable target intervention, not
novelty, necessity, learnability or generalization.

## 4. Experiment Plan
Independent six-file W2-derived adares_y48_mask_equivalence package.
Only final indicator indices receive the OR of existing positive labels in
exactly identical nonempty hard-mask groups. Empty masks and existing positives
are preserved. Six auxiliary calls, matching for every other loss, eos=.1,
native num_boxes, loss weights, masks, model dimensions and inference unchanged.
Hard-mask >0 is inherited, not a searched threshold. No added parameters.
Config is byte-identical native W2 under a unique basename.

Prospective production validation: actual package CPU32 equality to original
native CE on derived indices and omitted-mask parent CE; whole-parent AST
stripping audit; released SPFormer actual tools.train two steps on frozen first4
TRAIN scenes; indicator parameter updates; exact model/optimizer/scheduler state
restore; actual cached-decoder physical-parent equality; native predict2warmups/
3measurements on those batches; original largest-pair native stress2warmups/
3measurements, finite loss/gradients, peak reserved<24GiB. No diagnostic weights
deployed. Preserve any failure, investigate concrete causes, never fabricate PASS.

Post-training freeze before dispatch: released SPFormer, seed1999, fixed30,
one GPU. Y48/W2/B1 full9508/141 twice EACH, FIRST primary; scene-bootstrap10000,
seed1999, TRAIN size cuts1487/3325, CI upper<0 versus either comparator triggers
meaningful-subgroup harm review. Intervals crossing0 mean inconclusive. Report
repeat noise, official unique/multiple unknowns, queries128/singleargmax and
trained paired costs. No epoch/threshold/feature/weight selection on validation.

## 5. GPU Allocation Plan
Only server2 GPU0/1, one experiment/card. Preserve external GPU0 PID19723.
GPU1 naturally freed after verified Y45fixed30. Every diagnostic/dispatch still
requires live idleness and exclusive lease. At most one admission under this
tick's unchanged budget; reserve budget0 does not permit pool expansion.

## 6. Implementation Checklist
Native diagnosis and proposed-target CPU/native controls observed; independent
package built with only helper/optional final-loss arguments/final call changed.
Actual remote parent bytes captured; local/native six-file AST equality verified.
The original byte-hash guard failure is retained, with no files written by v1.
Production/released-init/resource tests, exact tested-byte commit, six-gate
review and hash-bound receipt remain required before canonical dispatch.

## 7. Risk Assessment
OR can overweight poor masks. Hard-mask equivalence need not imply equivalent
soft state or causal utility. The intervention may trade matched-label identity
for mask consistency and hurt size groups. Native gradient activity is not
trained benefit. No validation-based repair or automatic historical reopening.

## 8. Smell Check
This is a same-ID supervision-consistency realization grounded in actual joint
labels, not a new ranking margin, coefficient scan or renamed closed family.
Small code size alone is neither novelty nor admission evidence.
