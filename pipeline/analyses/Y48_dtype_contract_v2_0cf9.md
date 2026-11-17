# Y48 Native Float32 Loss Contract Correction

- V1 stopped before numerical checks: native loss_indi constructs a Float32
  pos_mask and assigns it into zeros_like(logits), so Double logits raise a
  dtype error. Original v1 source and raw failure log remain immutable.
- V2 executes the SAME extracted native loss/matcher with Float32 logits,
  denominator, literal gradient and shared scalar. Native eos_coef remains
  unchanged. No source function, label, mask, model, optimizer or gate changes.
- Before v2 execution, use the already specified reduction-length envelope
  with the actual loss tensor's epsilon: 8 * float32 epsilon *128*scale.
  Actual indicator/source equivariance stays Double with its original
  float64 envelope. This is a dtype harness correction, not a failed
  effect-size criterion being relaxed, and it does not admit a candidate.
- Record literal errors and optimum residuals, not only PASS labels.
  Synthetic ambiguity and cached duplicate incidence still cannot establish
  native mixed-label conflict or population benefit.
