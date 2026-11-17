# Y17 Implementation And Native Validation Freeze

2026-09-30 20:45 Asia/Shanghai, before this tick's implementation tests.
Stable AdaRES-Y17, prior paired_training_design remains intact.

Physical copy W2 model modules; shared native dataset/ops/train utility use
allowed, no imports from another model variant. Local dataset subclass
reorders TRAIN annotations using the prior deterministic pair-unit algorithm.
Preserve all annotation identities and original per-scene chunk sizes.
Native training entry gains one optional build_train_dataset hook; other model
classes fall back to existing build_dataset. No global dataset monkey patch.
Validation loader, prediction and existing models remain unchanged.

Training collate returns per-scene, one-use same-object pairs whose actual
encoded token rows differ. Annotation tokens can differ while truncation makes
encoded inputs identical: exclude such pairs explicitly, not from annotations.
Never pair across batch scenes, never infer correspondence by query index.
Criterion extension uses final native masks and indicator margins only.
Point-mass weighted Bernoulli JS, mean pairs within scene, then mean eligible
scenes. No eligible pair returns graph-connected zero. Fixed unit coefficient1
from bounded divergence scale, no hyperparameter search; native GT losses
remain exactly as the paired-off arm on the same prediction tensors.
Prediction uses original argmax indicator and no pairing metadata.

Validation first: pure sampler exhaustive small cases and full native train
dataset annotation preservation; no-pair/mixed-object/cross-scene/padding
fail-closed tests; analytic gradients and differentiable zero. Then released
SPFormer initialized native paired batches, same graph pair-off and wrong
pair control, JS reaching confidence/spatial parameters, full backward and
AdamW/PolyLR update through tools.train. Record all finite/nonzero gradients
and updates, original loss values, objective ratios, parameters/memory/time.
Checkpoint training state still diagnostic only.

Next fixed large train stress manifest with isolated W2/Y17 optimizer steps,
warm student-only inference, exact identical-state predictions W2/Y17
with runtime labels/pair IDs counterfactual, native checkpoint round trip
and actual DataLoader worker check. Use available GPU1/exclusive flock,
GPU0 Y15 untouched. Each test bounded by actual21:41:14 local tick deadline.
Any error leaves separate failure evidence before scoped repair/rerun.
Only after all six gates, source hash audit, candidate code commit, frozen
full-val harm/Pareto plan and receipt can canonical dispatcher start training.
No teacher, new inference queries/parameters or protocol changes.
