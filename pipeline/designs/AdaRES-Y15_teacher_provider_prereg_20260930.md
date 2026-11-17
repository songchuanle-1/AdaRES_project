# Y15 Current-Batch Teacher Provider

Registered before provider outcomes, 2026-09-30 Asia/Shanghai.
Native backward with external targets already ran; full augmented target supply
has not run. This amendment does not erase v1/v2 label-repeat warnings.

## Provider

Use a physical local copy of frozen B4 within the Y15 package. No imports of
another model variant in production. Lazily load the frozen checkpoint on the
first native loss call only. Exclude the external reference from student
optimizer/state_dict; its immutable checkpoint path/hash is training provenance.
The reference's raw masks come from the current batch's voxelization,
superpoints and language, with eval/no_grad. Process teacher before the student
autograd graph. No offline projected query or stale superpoint target is used.
No teacher is constructed in predict mode.

Use the unchanged K student masks to compute detached point-mass oracle IoU
on training ground truth. If frozen teacher oracle - current student oracle
exceeds the existing registered 0.05 material definition, use teacher binary
mask as extra-query BCE target; otherwise use the native superpoint GT union.
Original K masks, auxiliary targets and GT dice/score/semantic/indicator are
unchanged. The teacher BCE substitutes one existing query-row budget rather
than adding a coefficient or extra term.

## Tests

- Two independent real train batches, same augmented geometry for teacher and
  student; verify expression-to-scene boundaries, target lengths, detached
  masks and finite/nonzero gradients on every new tensor.
- Same provider targets with frozen shared weights in matched GT-only,
  language-permuted and slot-only arms. Report teacher/GT equivalence cases.
- Measure peak reserved/allocated with lazy teacher resident and real native
  optimizer steps. Frozen teacher tensors must not occur in student checkpoint.
- Isolate inference using cached decoder inputs from one forward, exact labels
  changes with identical decoder outputs, and repeated raw predictions.
- An unchanged-input repeat noise floor is diagnostic only; no silent conversion
  of v1/v2 strict numerical independence failure to PASS.
- Released SPFormer initialization must be tested as well as W2-warm probes.
  W2-warm probes do not authorize W2 checkpoint initialization for training.
- Full admission still requires explicit lightweight/Pareto and harm plan,
  plus valid independent config/code commit and all six real receipts.
