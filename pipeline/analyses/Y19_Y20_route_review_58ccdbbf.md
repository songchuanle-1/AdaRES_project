# Training Diagnosis Review - AdaRES-Y19 / AdaRES-Y20

Owned tick58ccdbbf48104660911c2a0c0fc97d3e; snapshot2026-10-02T07:48:36.785511+08:00.

## Actual Tests
- Y19 executed the actual G1 loss_indi function extracted by AST, versus separately point-expanded reference. Equal masses[1,1,1]: loss/gradient differences0/0. Unequal masses[1,4,2]: middle quality.5->.2, loss difference.10553765/gradientmax.02161419.
- This confirms the proposed all-query soft-target structure is G1's old mechanism; point measure changes targets/weights, not an independent new confidence interaction. G1's fixed30epoch.4980 failure and Y5/Y6 prior rejections are not relabeled.
- Reject Y19 realized proposal as renamed metric-target calibration. This is an independence/constraint failure, not evidence that all future quality objectives fail. Query-permuted new training controls were not run because the actual source-equivalence test resolved the specific duplication question.
- Y20 ran actual W2 model.train native loss on64distinct training scans/256expressions, all native losses/matches unchanged, same RNG/BN restoration, no optimizer steps; native/repeat/detached controls.
- Final indicator-input mask gradients versus native final maskBCE+Dice: scene-mean cosine+.0277698147,21negative/43positive; mean opposite coordinate fraction.1604790. All finite. Detached route zeros its indicator-mask gradient and keeps indicator-head gradient nonzero.
- Native repeat mean cosine+.0277677000, maximum cosine discrepancy.0002361611, zero sign changes. Preregistered global opposing-direction hypothesis is falsified. Reject Y20; never subset the21negative scenes or rebrand N1's rejected detach-only ablation for admission.

## Repeat Audit
- Original end-to-end capture raw logit max discrepancy8.6640701 retained. Native loss differences near numeric noise do not prove identical tensor/query order; cause remains unknown.
- Independent first8TRAINscan cached exact final indicator query/mask/padding/mass inputs: native/repeat/detached logits all exactlyequal; native mask gradient nonzero, detached mask gradient absent, detached head gradient nonzero, all finite.
- Cached-input identity proves only local forward intervention identity. It does not erase upstream native nondeterminism/order discrepancy, establish full-model determinism, or estimate trained benefit.
- Original64peak reserved14.673828GiB in diagnostic; no resource/admission/Pareto PASS.

## Independence And Limits
- Y19 objective-calibration hypothesis versus Y20 optimization-backward-routing hypothesis are distinct diagnostic questions. Neither is independently admissible production code; no candidate package/config/receipt/training created.
- One frozen W2 checkpoint, train-only bounded cohorts. No validation search, no effect-size retuning, no new claims of literature novelty.
- Continue third independent stableY21 augmentation mismatch diagnostic under its own preregistration; do not claim all routes exhausted from these failures.
- Evidence: y19_y20_diagnosis_58ccdbbf.json; y20_cached_input_58ccdbbf.json; Y19_Y20_native_diagnosis_58ccdbbf.md; Y20_cached_input_repeat_58ccdbbf.md. Logs/helpers retained and separately hashed on ledger.
