# Y17 Target-Equivalent Paired Training Design

2026-09-30 20:25 Asia/Shanghai. Prospective implementation, NOT admission.
Follow motivation_review_20260930-201800.md; retain stable ID.

## 1. Motivation

Improve faithful standard-indicator confidence using metric-aligned evidence,
target ScanRefer mIoU>=.515. Same-object real descriptions give paired spatial
supervision without deleting semantics or requiring another inference model.

## 2. Current Best Results

| ID | State | mIoU | Epochs |
|---|---|---|---|
| B1 | complete | .5020 | 30 |
| W2 | complete | .5100 | 30 |
| Y15 | native training GPU0 | unknown | 1/30 in progress |
| Y17 | diagnosis | unknown | no full training |

## 3. Analysis And Direction

Train16 selected same-target overlap .91084 versus repeat1.00000 and
wrong-target.09392. Same/repeat JS.0030789/6.24e-15.
W2 checkpoint native augmented four-batch gradients: same-JS confidence
gradients align with native total in every batch; wrong-target directions
separate, including three opposite confidence-head directions.
Released-backbone initialization has three slightly opposite confidence
directions: preserve this conflict, no assertion of convergence.

Native training readout p_e(s)=sum_q a_eq b_eqs,
a_eq=softmax_q(positive_margin_eq), b_eqs=sigmoid(mask_logit_eqs).
No query-index matching across descriptions. For pair(e,f),
J=sum_s n_s JS(Ber(p_es),Ber(p_fs))/sum_s n_s.
Let m=(p+q)/2; dJS/dp=.5(logit(p)-logit(m)).
For margin u_ek:
dp_es/du_ek=a_ek(b_eks-p_es),
dJ/du_ek=sum_s n_s/total*.5(logit(p_es)-logit(m_s))*
a_ek(b_eks-p_es).
For mask logit z_eks:
dp_es/dz_eks=a_ek b_eks(1-b_eks).
Thus paired evidence can reach both confidence and spatial support.
These identities are exact in the probability interior; numerical clamp
edges have a different derivative. JS unit checks double64 passed.

## 4. Experiment Plan

Implement a PHYSICAL copy of W2 under adares_y17_target_equivalent_pairing,
no imports from other variants. Register unique config/launcher mapping.
Integrate deterministic prospective manifest ordering into a project-local
training dataset class; preserve full annotation multiset, original per-scene
chunk4 sizes, scene ordering, batch2 and no augment differences between pairs.
Add native loss component mean_same_target_pair_JS, averaged over valid
scene pairs. No paired examples => differentiable zero; never fabricate
different-target positive pairs. Proposed fixed coefficient1 is a unit-scale
bounded divergence choice (0<=JS<=ln2), NOT a validated necessity or strength;
review total reduction and native gradient scale before admission.
Keep native GT BCE/dice/indicator/score/semantic/aux losses unchanged.
No validation coefficient scan or inference probability mixture replacement.

Matched controls: identical pairing manifest/native GT objective with JS off;
same implementation JS with prospectively permuted target pair assignment,
for diagnosis only, not a second independently admitted full run.
First native diagnostic uses released SPFormer initialization on the frozen
eight scenes, original augmented batch2; capture objective component and
all affected parameter groups, full finite backward/optimizer step, unchanged
inference when training pair loss disabled. Check target pairing identity and
missing-pair behavior. Then full frozen Y15 stress manifest large native
batches for time/peak and student-only inference latency versus W2.
Commit tested source only after physical/import/hash checks; six-gate receipt
and canonical monitor dispatch only. Exactly30epochs/seed1999.

Posttraining official fullScanRefer val mIoU versus W2/B1,target.515,
Acc25/50,spmIoU,128available queries/1selected, student latency/peak.
Repeat noise/scene-clustered paired intervals, official unique/multiple,
train-defined size cuts1487/3325 with no val tuning. Material harm/Pareto
rule follows frozen Y15 plan, no arbitrary noise constant. Historical
references alone cannot prove a matched causal benefit.

## 5. GPU Allocation Plan

Y15 GPU0 untouched. Y17 diagnostics use server-2 live-idle GPU1 exclusive
lease; training only after gates; one task/card. No server-6 or scheduler.

## 6. Implementation Checklist

Pending physical package, sampler native integration, fixed objective scale
review, native optimizer/resource/inference independence tests. Completed
population preservation audit and analytic derivative, native partial
gradients at initialization/checkpoint. These are not all-new-path PASS.

## 7. Risk Assessment

Repeated objects dominate pairing; rare singleton objects receive no JS.
Pair grouping alters chunk composition despite preserving identities/counts;
pair-off control must use exact same regrouping. JS can encourage
consistently wrong masks or compete with native losses. Smooth mixture
training readout differs from native argmax inference. No fresh full30epoch
control or novelty proof. Checkpoint gradients cannot replace production
released-backbone initialization evidence.

## 8. Smell Check

Adds a paired cross-description spatial training objective and data path,
not a scalar confidence gate. No new inference parameters are necessary.
Fixed coefficient is explicitly prospective, not validated by these probes.
