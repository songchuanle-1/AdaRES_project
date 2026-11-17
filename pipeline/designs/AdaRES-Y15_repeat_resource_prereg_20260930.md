# Y15 Repeat And Native Loop Follow-Up

Registered 2026-09-30 11:40 Asia/Shanghai, after v1 and before follow-up.
The v1 label-invariance assertion was false; no inference independence PASS.
Do not erase v1 or change its binary-equality criterion to hide this result.

1. Compare original prediction with an unchanged repeat and a GT-zeroed repeat.
   Report raw probabilities, binary differences and the repeat noise floor.
   If differences exceed the repeat behavior, investigate dependence, not tune.
2. Warm three forwards and measure ten synchronized student-only forwards for
   enabled/disabled completion on two fixed train scans, same model and weights.
   Report all times, medians and allocated/reserved peaks. Cold first backward
   in v1 was 23 seconds versus warmed 0.35-0.39 seconds and is not a fair ratio.
3. Execute tools/train.py train() on exactly two batches, GT configuration,
   released native optimizer, scheduler, grad clipping; inspect every new tensor
   update. This is bounded integration, not a queued full-model experiment.
4. Resource admission remains pending: no pre-existing numerical Pareto budget
   is assumed from the two-batch measurements. Do not retroactively fit a budget.
5. Teacher native loss connectivity uses supplied frozen targets in the probe.
   Full training needs a teacher provider aligned to augmented current batches;
   cached superpoint indices from nonaugmented diagnosis are not reusable.
