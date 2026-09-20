# AdaRES_project Research Memory

## Goal And Phase
- Phase 1: ScanRefer mIoU >= 0.515; not reached. Phase 2 awaits Phase 1.
- Root: D:/project/AdaRES_project; model code: AdaRES/.
- Remote code: /bak_home/songcl/Project/AdaRES_project/AdaRES.
- Use server-2 GPUs 0 and 1 only; server-6 is prohibited.
- Train from released SPFormer sparse-UNet with project-authored decoder.
  Never attribute released b005/CFNet results to AdaRES-B0.
- No validation tuning of thresholds, temperatures, loss weights, feature
  subsets, epochs, ranking margin, or occupancy loss weight.
- Report mIoU, Acc@0.25, Acc@0.5, active queries, latency, and peak memory.

## Champion And Baseline
- AdaRES-B1 is the faithful standard-indicator baseline and champion:
  mIoU 0.502, Acc@0.25 0.610, Acc@0.5 0.553, spmIoU 0.467;
  best epoch 18 of 20. Target gap: 0.013.
- B1 fixed-checkpoint pilot: 4 scenes / 108 expressions, 22.277 effective
  active queries, 25.773 ms/expression mean latency, 2161.027 MiB peak memory.
  Pilot subset mIoU does not replace full-validation mIoU.
- Evidence: pipeline/analyses/AdaRES-B1_profile_pilot_20260920-095000.json
  and pipeline/analyses/AdaRES-B1_done_complete_20260920-091444.md.

## Active Hypotheses
- B2 tests true-IoU supervision of the standard query-only indicator, keeping
  the mask generator. If unsuccessful, do not tune its coefficient.
- B3 tests mask-conditioned confidence evidence: query latent state plus
  foreground and complementary background scene features pooled by its mask.
  It changes the confidence information path, not a numeric setting.
- Existing admission evidence: finite nonzero module gradients and real-scene
  smoke; B3 loss 21.7446, gradient norm 246.0453.
- Review/design: pipeline/reviews/motivation_review_20260920-093708.md and
  pipeline/designs/exp_design_20260920-093708.md; admission decision:
  pipeline/decisions/20260920-093708_next_batch.md.

## Closed Or Failed Directions
- B0 closed after 20 epochs at mIoU 0.333. Strong oracle masks (0.847 on
  124 validation expressions) but poor ranking; not the official baseline.
- Applied fixes: shape-compatible pretrain loading, scalar loss reshape before
  DDP reduction, and non-main-rank logger guard.
- B2's original smoke harness incorrectly required B0-only loss keys.
  Model-agnostic repair passed real-scene smoke: loss 22.3169, gradient
  norm 249.0111. This was a harness failure, not a rejected B2 hypothesis.

## Running And Queue
- Current preflight identifies B2 running on server-2 GPUs 0,1.
- Log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_b2_20260919_204142.log.
- 2026-09-20 11:13 +0800 preflight: epoch 4/20, best mIoU 0.412,
  validation sequence 0.325, 0.380, 0.412; no_improve=0/10.
- Epoch-3 metrics: mIoU 0.412, Acc@0.25 0.523, Acc@0.5 0.452,
  spmIoU 0.384, spAcc@0.25 0.510, spAcc@0.5 0.422.
- B2 improved for three consecutive validations but remains 0.090 below B1
  at the same epoch; no early conclusion is drawn before the fixed 20 epochs.
- B3 remains admitted and waiting on B2. Watcher:
  /bak_home/songcl/Project/AdaRES_project/scripts/watch_b2_then_b3.sh;
  PID 85492 was verified alive at the prior 10:42 local tick, not this tick.
- Existing handoff requires all 20 B2 validations and final best mIoU below
  0.515. An early exit requires failure analysis, not automatic handoff.
- Current snapshot: active_count 2/2, running 1, waiting 1, claiming 0,
  candidate_budget 0. running_map assigns both allowed GPUs to B2.
- Previous monitor/accounting mismatch is not present in this snapshot.
  Two active experiments do not imply two concurrent GPU jobs or a free GPU.

## Open Questions And Risks
- Pending questions in snapshot manifest: none.
- Prior monitor process/accounting mismatch is no longer observed; root cause
  remains unverified. Do not claim a code fix or permanent resolution.
- Remote clock reports 2026-09-19 22:11:21 -0400 while local check occurred
  around 2026-09-20 10:42 +0800: roughly 31 minutes of skew after conversion.
  Use observed iteration advancement, not cross-host mtime alone, for liveness.
- B2 final trajectory and B3 effect remain unknown. B2/B3 efficiency metrics
  remain to be measured; do not substitute B1 pilot measurements.
- pipeline/closure_policy.json is absent. No due research-work IDs are listed;
  research_on_capacity_gap is not enabled in the supplied context/goal.
- Prior memory asserted an active native background goal; this headless tick
  did not verify that separate thread's goal state.

## Last Tick
- 2026-09-20 11:13 +0800; event/active_queue_changed.
- Used completed preflight without rerunning monitor or goal checks.
- Preflight shows B2 at epoch 4/20 with best 0.412, beyond the previous tick's
  epoch 3 progress. No additional live/GPU query; utilization was not sampled.
- Queue accounting now agrees with B2 running and B3 waiting.
- No terminal event, new candidate, queue mutation, or research decision.
- Report: pipeline/reports/summary_20260920-111342.md.
- No summaries older than 24 hours were found for cleanup.
