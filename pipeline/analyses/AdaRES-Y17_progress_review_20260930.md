# Y17 Actual Progress And Open Gates

2026-09-30 20:25 Asia/Shanghai, this owned design tick.
Diagnosis advanced, no new full-training entry or admission receipt.

Frozen train16,32same-target pairs/64expressions:
selected overlap .91084261, different-object.09391677, repeated1.0;
same/different/repeat JS.00307886/.04927221/6.24493e-15.
19/32 same-target binary pairs differ, repeat selection flips0.
Selected/oracle target IoU means.85234006/.93861115.
Same-target candidate pool oracle gain.00517597, diagnostic labels only.

Native augmented released-init4batches/8actual scenes:
finite gradients reach10indicator tensors and4input projection tensors.
Same-v-native indicator-head cosine[-.03406,.03281,-.05312,-.01988].
Input projection same/wrong cosines~.86-.95 suggest generic smoothing
at initialization; do not discard or amplify it away.

Diagnostic W2 checkpoint on exact same native cohort:
same-v-native confidence cosines[.04671,.29954,.44452,.21644].
same-v-wrong[.14948,-.52660,-.75876,-.96569].
Exact-self32readouts JS0. No optimizer step, physical variant or new
parameter backward claim. Checkpoint initialization is diagnostic ONLY.

Population sampler audit562scans/36665expressions/9372chunks:
30258paired and6407unpaired expressions; all identities and original
per-scene chunk-size sequences preserved. Native/proposed same-target
pair comparisons3214/38708; incidence gain is not improved accuracy.
Actual native data-class integration remains pending.
Double64JS closed-form derivative max errors1.30e-18/1.52e-18,
selfJS0, self-gradient roundoff1.67e-18, symmetry0. No exact-zero
gradient claim despite algebraic zero.

Next: physical independent W2 package/config, local paired dataset sampler
and justified frozen objective with matched pair-off/wrong-target controls;
native full backward/optimizer, no-pair edge cases, stress/resource and
inference identity. Run on live-idleGPU1 next owned research cycle.
Do not count diagnosis+math as scientific admission or claim novelty.
