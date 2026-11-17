# AdaRES-Y14 design: cross-fitted alignment reliability gate

Date: 2026-09-30 Asia/Shanghai

## Motivation and frozen hypothesis

Y13 established that mask-space correspondence is real: Hungarian matched-query
similarity was 0.21915 versus 0.04485 for cyclic controls. A fixed equal-weight
committee nevertheless failed its significance rule. Y14 asks whether a small
gate trained only on training scenes can learn when aligned B4 confidence is
reliable.

For expression `e`, `s_i = IoU(b(p_i^W2), b(p_{pi(i)}^B4))`,
`d_i = |zhat_i^W2 - zhat_{pi(i)}^B4|`, and mask entropy `h_i^m`. Define

`phi_i = [s_i, 1-s_i, d_i, h_i^W2, h_i^B4, 1]`,
`g_i = sigmoid(w^T phi_i)`,
`c_i = zhat_i^W2 + g_i * zhat_{pi(i)}^B4`.

`w` is trained only on the first 64 non-augmented training scenes with
scene-level 4-fold cross-fitting. The query-level objective is the oracle-mask
ranking loss

`L = sum_{i,j: IoU_i > IoU_j} log(1 + exp(-(c_i-c_j)/tau))`,

with fixed `tau=0.25`; no validation label, threshold, or epoch is used for
optimization.

## Controls

1. State-shuffled gate: identical architecture and training budget with
   `phi` shuffled within each expression.
2. Hungarian shuffled: cyclic B4 assignment with the same gate.
3. W2-only and Y13 fixed committee.

## Frozen experiment

Train the gate on train64 with cross-fitting, freeze the four fold models, and
evaluate once on the first 32 validation scenes. The primary comparison is
scene-mean query IoU of gated committee minus W2; the second is minus the
state-shuffled control. Pass requires both deltas positive, gated-minus-W2
scene sign test `p<0.05`, zero repeated-forward selection mismatch, and no
validation-derived gate parameter.

## Six gates and status

| Gate | Required evidence | Status |
|---|---|---|
| Diagnosis | Y13 match-specific direction and Y14 train-only gate improvement | Y13 passed direction but not significance; Y14 unrun |
| Controls | state-shuffled and assignment-shuffled gates | designed, unrun |
| Native backward | finite gradients for `w` and no frozen-backbone update | pending |
| Resources | single-GPU memory and latency profile | pending |
| Independence | not a Y12/Y13 coefficient scan; no positive-set loss | designed, pending implementation audit |
| Goal constraints | 30-epoch protocol, server-2 only, no threshold/loss-weight tuning | designed |

No admission or queue entry is authorized until all six gates pass.
