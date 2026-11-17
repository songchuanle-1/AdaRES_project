# Y15 Stress, Resume And Teacher-Specificity Preregistration

Frozen 2026-09-30 15:43 Asia/Shanghai, before this tick's new outcomes.
Stable candidate AdaRES-Y15; validation, not an admission.

## 1. Motivation

Goal description: "AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
Continue the existing complementary-mask query, not a new candidate family.
Teacher usefulness and deployment robustness remain unknown.

## 2. Current Best Results

| ID | State | ScanRefer mIoU | Epochs |
|---|---|---|---|
| W2 | done_complete | 0.5100 | 30 |
| B1 | done_complete | 0.5020 | 30 |
| Y15 | validation | unknown | no full training |

The previous four-scene native probes established connectivity only.

## 3. Analysis & Direction

Teacher/GT total new-parameter gradients were nearly collinear. Inspect the
extra-query BCE itself, not an arbitrarily amplified total objective.
Dataset items are expression chunks; sampling unit is actual scan_id.

## 4. Experiment Plan

1. CPU-only metadata pass through each distinct training scan's raw point cloud.
   Freeze the union of the top two scans by raw point count and top two by
   unique superpoint count, descending count and ascending scan_id tie break.
   Pair selected scans in that deterministic order, at the native batch size
   two (last singleton if odd). Use the existing four-expression train chunks,
   augmentation and seed1999. Save manifest before any model output.
2. Independent clean processes for W2, Y15 teacher and physical GT package,
   released SPFormer initialization. Two warmup steps, three measured native
   optimizer steps per batch. Record actual points/superpoints, parameters,
   per-step time/peak memory, finite loss, OOM/failure without changing config.
3. Native gorilla checkpoint/save/resume with AdamW and PolyLR. After one
   native update and scheduler step, save; compare uninterrupted next step
   versus a new instance resumed from that checkpoint on the exact same batch.
   Compare model/optimizer/scheduler before continuation exactly, check teacher
   excluded from student state, and report all continuation differences and
   independent-repeat noise rather than inventing an equality tolerance.
   Preserve RNG state for the paired diagnostic; separately audit lazy teacher
   construction effects. No altered training schedule.
4. On the original first64 distinct train scenes, use the first eight scenes
   in scan order having a prior material teacher event. Freeze these identities
   from the existing train-only diagnosis before execution. On augmented
   current batches, compare extra-query BCE gradients for correct teacher,
   GT, and same-scene different-object teacher-target control. If an expression
   lacks a different-object peer, mark unavailable, never substitute same
   object. Log scan_id/object_ids/ann_id, material incidence, weighted mask
   differences, cosine/norm/max gradient differences and repeated outcomes.
   Material teacher events must actually occur in this probe to establish the
   claimed substitution; no fixed efficacy or tiny-cohort significance gate.
5. Fixed native prediction repeat, raw probability noise, binary/selection
   repeat flips and label-independence against independent repeat noise.

## 5. GPU Allocation Plan

Only server-2 GPU0/1; one diagnostic process per card with live idle checks and
exclusive per-card flock. No queue/training bypass or server-6 access.
CPU metadata may run separately. Bound each launch to the actual tick deadline.

## 6. Implementation Checklist

- Create additive stress/resume/specificity tools; preserve older evidence.
- Bind the teacher file to a recorded SHA256 before any eventual training.
- Audit external teacher device and RNG lifecycle without changing objectives.
- Persist failures and successful reruns independently.
- Admit only after six-gate review, tested-code commit and hashed receipt.

## 7. Risk Assessment

Full-cloud stress may OOM despite smaller scenes passing. That is evidence,
not permission to reduce batch size or points. Sparse reductions may prevent
bitwise continuation identity; report measured noise. Native gorilla does not
save RNG, so paired RNG replay here is diagnostic, not a claim of exact
whole-dataloader resume. Teacher necessity is not established by a nonzero
gradient alone. No validation tuning or BCE coefficient change.

## Frozen Post-Training Evaluation

Exactly30 epochs, released SPFormer backbone, official full ScanRefer val,
existing best-checkpoint rule with no post-hoc epoch extension. Primary mIoU,
target0.515 and W2/B1 references; secondary Acc25/50, spmIoU, active queries.
Evaluate checkpoints with identical input ordering/seed and repeated
inference; retain expression identities and cluster uncertainty by scan.
Use scene-clustered paired bootstrap intervals when sufficient independent
scenes exist. Otherwise, or if intervals cross a decision boundary, mark
inconclusive. Noise floor must come from repeats, never a chosen constant.

Predeclared subgroups: official unique/multiple and target-point-size tertiles
whose cutpoints are computed on training annotations before val evaluation.
Material harm means a subgroup's paired mIoU deficit exceeds the measured
repeat noise with its entire95% scene-clustered interval below zero. A single
adverse scene is a diagnostic flag; unquantified noise makes harm inconclusive.
Do not tune architecture, loss, thresholds or features on any subgroup result.

Lightweight evaluation uses student-only warm inference, same hardware/batch,
parameters, peak memory and latency versus W2 and GT control. Report the
accuracy/cost frontier, not accuracy alone: a variant dominated in accuracy,
latency and memory has no lightweight improvement claim. An accuracy increase
with higher cost is a tradeoff, not automatic Pareto superiority. Pretraining
resource executability requires native batch2 fitting the authorized24GiB
hardware, one student optimizer, one extra query (K128->129) and no teacher
inference dependency. Teacher adds only its frozen reference; record full
training cost and decide whether the specificity justifies this route. This
protocol does not turn hardware fit into a trained lightweight efficacy PASS.
