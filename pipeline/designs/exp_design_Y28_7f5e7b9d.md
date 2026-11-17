# Experiment Design: AdaRES-Y28
2026-10-05, owned7f5e7b9d138646e1a4458341d88f7124.
Created after prospective diagnostic prereg and motivation review.

## 1. Motivation
Goal: "AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned evidence,
targeting ScanRefer mIoU >= 0.515."
Capacitygap1. Test explicit final-mask visual readback for confidence, not
fixed cosine reranking or a geometry-peer rename. Upstream semantic information
already exists; readback benefit/necessity is unproved.

## 2. Current Best Results
| Experiment ID | Config | State | Best mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | done_complete | .510107995 | .562999579 | .473719922 | 30 |
| B1 | adares_b1_standard_res | done_complete | .502119335 | .552797644 | .466681174 | 30 |
| Y27 | adares_y27_relative_peer_res | done_complete | .509987856 | .562999579 | .475034731 | 30 |
Y27-W2allCI[-.008415036,.008403480], not efficacy-positive.
Values are preceding frozen full comparison, not new validation tuning.

## 3. Analysis & Direction
W2confidence reads overlap and query features, but not final mask-pooled
scene content directly. Native TRAIN64/256 diagnosis has163oracle gaps with
distinct selected/oracle pooled features. This is a discriminating diagnosis,
not accuracy proof. Old J1 rawcosine and Y27geometry results remain unchanged.
Readback uses ALL256 parent feature channels, original mass and predicted masks.
No hand-selected word, feature, threshold, loss weight or temperature.

## 4. Experiment Plan
| # | ID | Config Path | Purpose / Hypothesis | Expected Result | Priority | Depends On |
|---|---|---|---|---|---|---|---|
| 1 | AdaRES-Y28 | configs_song/model_paper5/adares_y28_mask_readback_res.yaml | Final predicted-mask visual-content aggregation into confidence | Reproducible native mechanism; trained effect unknown | P0 | Six real gates |
For each stage, foreground=sum(sigmoid(mask)*valid*mass*scene)/sum(weights),
background uses(1-sigmoid(mask))*valid*mass. A shared two-layer MLP receives
normalizedquery, foreground, foreground-background (3*256 ->256 ->256),
then adds readback to the original overlap context before the unchanged head.
No new loss or GT inference inputs; exactly128queries/1selected.
New4trainabletensors/expected262656parameters. Initialization uses CPU RNG fork
so all shared initial tensors remain exactly identical to parent.
Native original fixed30/seed1999/released SPFormer/AdamW/PolyLR unchanged.

Frozen controls: correct/rolled mask weights, uniform point-mass scene pooling,
disabled path, detach path, repeat; joint superpoint/query permutation and
padding counterfactuals. Native TRAIN8 and released-init4 kept regardless of
sign. Isolate same encoded decoder/criterion inputs for mask-gradient readback.
All four new tensors must receive finite/nonzero gradients and optimizer updates.
Resume model/optimizer/scheduler exactly before continuation; retain numerical
differences afterward, no whole-loader/RNG or bitwise trajectory claim.

Frozen post-training plan BEFORE launch:
- Full ScanRefer val9508expressions/141scenes, native best checkpoint; W2/B1
  EACH two independent eval passes, source/checkpoint/identity hashes preserved.
- Primary mIoU plus Acc25/Acc50/spmIoU/query128/selected1. No alternate subset
  winner selection; no validation threshold/temperature/weight/schedule tuning.
- Paired expression-weighted whole-scene bootstrap10000/seed1999/95%CI;
  training-seed uncertainty not identified by this interval.
- Original TRAINtargetsizecuts1487/3325 define small/medium/large. Material
  harm if subgroup difference-to-W2 upper95%CI<0. All/subgroup intervals
  crossing0 are inconclusive; no rounding negative endpoints into gains.
- Empirical independent repeat differences, not invented universal noise floor.
- Trained serialbatch1 TRAIN8+frozenlarge728/6780, two warmups/three readings;
  median/p95 and allocated/reserved memory versusW2. Descriptivep95 limitation.
- Lightweight Pareto claim ONLY with trained task/cost evidence; feasibility
  permits<24GiB and<1%parameteraddition, not a promise of Pareto improvement.

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 1 | Exclusive diagnostics, then at most1admission | Y28 | Only authorized slot, liveguards required |
GPU0reserved outsideproject; neverquery/refill. No server-6/background.

## 6. Implementation Checklist
- [x] StableID/preregistered TRAINdiagnosis with actualsourcehash/nativecapture.
- [x] Independent six-file W2copy/configexact; onlydecoderreadbackchanges.
- [x] Matched cached/native controls and every new tensor gradient.
- [x] Native tools.train two steps/checkpoint resume/cached parentidentity.
- [x] Serial native stress/latency/memory and exact shared initialization.
- [x] Structural smell, independence and six-gate admission review.
- [ ] Scoped tested7sourcecommit/sync/admissionreceipt/canonicaldispatch.
- [ ] Increasing native steps in unique traininglog before trainingclaim.

## 7. Risk Assessment
Predicted-mask weights can propagate harmful confidence gradients into mask
formation; detach-control must measure this causal path, not presume helpfulness.
Random new weights may increase initial loss; preserve correct/wrong/disabled
comparisons, no outcome-selected cohort. Additional repeated pooling may raise
memory/time. Original masks unchanged immediately, but trained masks may change.
Sparse/FPS numerical instability persists; cachedidentity is not global stability.
Scientific novelty/causal trained benefit unknown. Never broaden diagnostic
permission to GPU0 or relaunch old completedY23/Y27.

## 8. Smell Check
Prospective review: new aggregation connects final predicted masks, scene
content and confidence; not norm-only/linear-only numeric retuning. Four tensors
262656params expected, dimension inherited256/no search. Signals1/3/4/5/6 not
expected; signal2 superficial MLP resemblance is defended by NEW mask-weighted
scene aggregation and foreground/background correspondence. ExactAST and
actual native controls required before treating this structural check asPASS.
ActualAST/nativecontracts/native2steps/resume/resources completed; review
Y28_admission_review_7f5e7b9d.md preserves initial mixed/control losses and fixed
2step diagnostic learnability without claiming necessity/efficacy.
Status: IMPLEMENTED; sourcecommit/receipt/launchverification stillPENDING.
