# Experiment Result Analysis - AdaRES_project
Experiment: AdaRES-C2 | State: done_complete

## 1. Basic Info
- ID: AdaRES-C2
- Config: configs_song/model_paper5/adares_c2_hard_negative_margin_res.yaml
- Server: server-2 | GPU: 1
- Launched: 2026-09-22 22:31:46 +08:00.
- Terminal reconciliation: 2026-09-23 06:50:59 +08:00, after three absent-process checks.
- Launch-to-reconciliation interval: 8h 19m 13s; actual runtime is not established.
- State: done_complete; all 10 validation results and final last.pth save verified.
- Protocol: B4-E30 best-checkpoint warm start, fixed 10 epochs, seed 1999,
  batch size 2, learning rate 2e-5; hard-negative IoU-margin objective.
- Sources: pipeline/experiments/AdaRES-C2.json;
  pipeline/designs/exp_design_20260922-223000.md;
  remote log /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_c2_hard_negative_margin_res_20260922_223146.log.

## 2. Best Metrics
| Metric | Value | @ Epoch |
|---|---:|---:|
| mIoU | 0.4980 | 8 |
| Acc@0.50 | 0.544 | 8 |
| Acc@0.25 | 0.606 | 8 |
| spmIoU | 0.464 | 8 |
| spAcc@0.50 | 0.509 | 8 |
| spAcc@0.25 | 0.598 | 8 |

Best mIoU is reported as 0.4980 by the terminal record; stdout prints 0.498.
Epoch 8 saved best.pth; epoch 9 also rounds to 0.498 but did not save a new best.
Auxiliary metrics above are aligned to epoch 8, not independently maximized.
Active queries, inference latency, and peak memory: not measured in the
available terminal evidence; the efficiency-reporting constraint remains open.

Best checkpoint (save confirmed by log, not loaded this tick):
`/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_c2_hard_negative_margin_res/20260922_100039/best.pth`.

## 3. Convergence Summary
- Validation mIoU, epochs 1-10 (stdout precision):
  0.478, 0.485, 0.470, 0.483, 0.437, 0.479, 0.480, 0.498, 0.498, 0.495.
- Best checkpoint first saved at epoch 8; two subsequent non-improving epochs.
- Final epoch: mIoU 0.495, Acc@0.50 0.541, Acc@0.25 0.598, spmIoU 0.461.
- The run recovered from a substantial epoch-5 dip but never surpassed the
  inherited B4-E30 reference. The final log confirms Val[10/10] and last.pth.
- The log's internal start timestamp and checkpoint directory use
  2026-09-22 10:00:39, inconsistent with the launch record. Do not derive
  wall-clock runtime from this timestamp without resolving the remote clock.

## 4. Comparison
- vs baseline AdaRES-B1: -0.0040 mIoU (0.4980 vs 0.5020).
- vs direct reference/champion AdaRES-B4-E30: -0.0114 (0.4980 vs 0.5094).
- vs AdaRES-B2-E30: -0.0078 (0.4980 vs 0.5058).
- vs target: -0.0170 (0.4980 vs 0.5150).
- These are checkpoint-level comparisons, not a matched causal estimate:
  C2 adds 10 adaptation epochs and no B4-only continuation under the identical
  adaptation schedule is available. A single seed does not establish significance.

## 5. Conclusion
- Conclusion: unsuccessful improvement; retain B4-E30 as champion.
- The tested hard-negative margin adaptation did not improve the aggregate
  task metric. Evidence does not isolate ranking, masks, or adaptation as cause.
- Close this tested route. Do not enqueue margin/weight/epoch tuning,
  a seed repeat, or a renamed equivalent as a successor.
- C1 remains running under its admitted fixed protocol. Evaluate it on terminal
  reconciliation; investigate ranking-versus-mask degradation and matched-control
  adequacy in the next eligible design, without asserting a root cause now.
- No crash repair, rerun, model edit, or queue mutation is warranted this tick.
