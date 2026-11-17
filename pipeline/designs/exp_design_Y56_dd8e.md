# Experiment Design - AdaRES-Y56 Unique Competitors

## 1. Motivation
Goal description: AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
Test whether competitor representation multiplicity is a removable nuisance
in the confidence evidence graph; this is not a tuning search.

## 2. Hypothesis And Scope
For fixed unique query+valid-softmask prototypes, changing their exact
multiplicities should not change prototype confidence. Exclude the entire
focal equivalence class, including all focal copies, from peers. This extends
the earlier singleton-focal diagnostic explicitly, before new outcomes.
No claim of approximate-duplicate invariance or population harm is made.

## 3. Architecture
Independent package adares_y56_unique_competitors, same W2 parameter shapes
and all 769 indicator inputs. Class keys concatenate detached raw query and
raw logits on valid support. Let c(j) count exact copies; zero same-class
overlap, divide remaining column j by c(j), row-normalize with the unchanged
1e-6 denominator floor. Max-overlap excludes same-class copies but is NOT
divided by counts. Single-class inputs have zero context/max-overlap.
No-copy inputs take the unchanged overlap graph. No thresholds, trainable
scalars, extra loss, dropped features, query changes or state transfer.

## 4. Matched Controls
CPU float32/64: seed1999, d256/Q128; unique prototype counts [1,63,64],
[1,64,63], [2,62,64]. Each replicated row is compared with the same native
head on three unique prototypes, including duplicated focal. Repeat exact;
joint query permutation within 128*eps*max(1,maxabs(logits)); same bound for
unique-set error. Negative controls: no copies equals W2 bitwise; identical
query with different raw masks remains distinct; different raw queries with
same masks remain distinct; one-coordinate perturbation is not rounded away;
padding, empty support, single class, Q1, zero mass, mixed batches; finite
nonzero component gradients. Parent must witness non-focal multiplicity
signal above the same fixed arithmetic bound; retain any failed assertion.

## 5. Native Backward And Resources
Released SPFormer initialization is the only formal init. Use original fixed
eight TRAIN scenes plus the discovery pair from the broader TRAIN cohort,
seed1999, native batch2, all seven indicator sites and unchanged task loss.
Every indicator parameter and each site's logits must have finite nonzero
task gradients. Capture actual exact class counts and parent-vs-candidate
counterfactuals on the SAME inputs/weights; no-copy rows must be bitwise equal.
Run full parent/candidate native loss on discovery pair with independent
reset state/RNG. Preserve signed loss deltas without a pretraining gain gate.
Separate trained-W2 case diagnostics are explicitly not formal initialization.

Resources: parameter delta must be zero. Run two warmups and three measured
native forward/backward iterations on the largest point-count pair in this
fixed cohort, peak reserved below physical 24GiB. Profile parent/candidate
same weights on each single TRAIN scene, two warmups/three timings, report
paired latency and peak memory. No arbitrary relative-latency cutoff is
invented; assess lightweight Pareto compatibility and uncertainty from these
measurements, do not claim global worst-case or global Pareto guarantee.

## 6. Training And Frozen Evaluation
Same released SPFormer pretrain, Q128, single argmax, seed1999/fixed30,
batch2, original optimizer/scheduler/loss and seven auxiliary attachments.
No validation tuning. After training, full frozen ScanRefer val9508/141,
two independent repeats; FIRST primary, repeat is noise audit. Compare
Y36 champion, W2 parent and B1 baseline; scene-cluster bootstrap10000/1999,
scene is sampling unit, not query/layer. Use existing TRAIN-frozen target
size cuts1487/3325 and material-harm trigger subgroup difference CI upper<0
vsY36/W2. Report mIoU, Acc@.25/.5, active queries, latency, peak memory;
require lightweight Pareto, no metric-free promotion. One seed cannot resolve
seed uncertainty; repeats quantify inference noise only. Official unique/
multiple subgroup labels are unknown unless source metadata verifies them.

## 7. Risks
Copies were rare and absent at the final layer in the prior trained cohort;
training may erase the signal, so overall benefit is unknown. Exact grouping
is discontinuous at equality, detached and potentially expensive; formal
native activity and measured cost are required. Branches without copies are
deliberately no-op. No novelty claim without a literature audit. No automatic
commit/admission from CPU tests, gradient activity or an improved local loss.

## 8. Smell Check
Zero parameter delta is intentional. This changes the peer equivalence and
aggregation graph by a fully specified set measure; it is not temperature,
threshold, feature-subset, loss-weight or normalization-constant tuning.
No configurations are swept; negative controls can falsify this realization.
