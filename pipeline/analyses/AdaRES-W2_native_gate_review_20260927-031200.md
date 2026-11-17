# AdaRES-W2 native admission review

2026-09-27 03:12 Asia/Shanghai. W2 is `adares_w2_pointmass_competition`,
physically copied from B4; the only model-path change is a point-count-aware
overlap matrix and input propagation. Unique config is byte-identical to B4
except basename. No validation hyperparameter search. Native tests used an
isolated package under the server-2 project `AdaRES/tools/w2_probe_packages/`;
production model tree was not changed by the probes.

## Mathematical mechanism

For query mask probabilities p_qs and observed point counts n_s (independent of
object labels), B4 computes overlap with n_s=1. W2 uses

    I_qr = sum_s n_s p_qs p_rs
    U_qr = sum_s n_s (p_qs + p_rs - p_qs p_rs)
    A_qr = (I_qr/U_qr) / sum_{k != q} (I_qk/U_qk)
    c_q = H(LN([z_q, sum_r A_qr z_r,
                z_q - sum_r A_qr z_r, max_{r != q} I_qr/U_qr]))

where H and z_q are unchanged B4 indicator head and normalized query.
The relevant equality is `IoU_points(mask_q,mask_r) = I_qr/U_qr` for hard
masks and superpoint-to-point broadcasting. Soft masks provide a differentiable
surrogate. The path adds point-mass evidence to context before the existing
indicator loss; it does not change the target labels, loss weights, backbone or
mask output. It is not the R2 peer-confidence propagation rule.

## Six gates

| Gate | Observation | State |
|---|---|---|
| Diagnosis | train64 point counts vary (mean CV 3.3245), and W2 changes oracle margin in 205/388 errors versus 166/388 shifted-count control. Frozen mIoU 0.8225565 versus B4 0.8225620, no claim of gain. | Mechanism exists, modest task response. |
| Controls | Same input masks/query/head; true n_s versus cyclically rotated n_s and B4 uniform mass, fixed ahead of data. Direct W2 versus shifted oracle-margin wins 208/388, losses 180/388. W2 flips only 6/1660, zero beneficial. | Controlled effect is mixed; no validation claim. |
| Native backward | Native batch `[0,1]`: W2 loss 13.6225185, indicator-head grad norm 2.476598, out_norm 0.684150. B4 loss 13.6225128, grads 2.476640/0.684147. Batch `[256,257]`: W2 loss 11.0877094, grads 1.658293/0.911834; B4 loss 11.0876675, grads 1.658399/0.911820. True versus rotated mass produced nonzero logit differences in all seven layers; `[256,257]` max first/final differences 0.03268/0.00435 and head-weight gradient max difference 0.000693. Finite nonzero gradients, no optimizer step. | Passed for tested batches. |
| Resources | `[0,1]` W2/B4 reserved 4.789/4.750 GiB, seconds 20.38/20.43; `[256,257]` 6.434/6.396 GiB, seconds 24.86/24.92. Additional parameters zero. | Fits one 24 GiB RTX3090 on tested batches; not full-run maximum. |
| Independence | Unique model package and YAML basename; no imports from another variant. Synthetic multiscene expansion `[2,1,2,1,3,2,1]`, zero-padding and global point-count checks passed. Two native predict scenes matched the separately derived reweighting function exactly. | Passed in isolated staging. |
| Constraints | ScanRefer, released SPFormer sparse-UNet, project decoder, 30 epochs, seed 1999, one GPU on server-2 [0,1]; no server-6, no validation tuning. | Satisfied if deployed unchanged. |

## Prospective training interpretation

Primary: ScanRefer validation point mIoU versus B4-E30 0.5094 and B1 0.5020,
using the fixed 30-epoch protocol. Report Acc@0.25, Acc@0.5, spmIoU, active
queries, latency, peak memory. Compare scene-paired mIoU differences with
scene-clustered intervals when enough scenes exist; estimate noise by a
separate paired repeat or mark it unknown. Material harm: a preregistered
subgroup with >=20 scenes and a scene-cluster interval wholly below zero
requires investigation, not automatic cherry-picked threshold changes. A
single adverse scene is diagnostic only. The B4-E30 champion resumed from an
epoch-20 optimizer state, so its historical 0.5094 is not a fresh matched
30-epoch control. If W2 surpasses it, repeat with a fresh matched B4 before
claiming causal lift or novelty. If W2 loses, report the failure and close
this exact mechanism without weight/temperature/feature scans.

Literature novelty has not been established. The frozen train64 signal is
small and could disappear or reverse in full training; the decision to
train is a falsifiable test, not a forecast of >0.515.
