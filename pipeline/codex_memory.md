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
- AdaRES-B1 is the faithful standard-indicator baseline:
  mIoU 0.502, Acc@0.25 0.610, Acc@0.5 0.553, spmIoU 0.467;
  best epoch 18 of 20.
- AdaRES-B2 is the current champion: mIoU 0.505, Acc@0.25 0.607,
  Acc@0.5 0.555, spmIoU 0.471 at epoch 19. Target gap: 0.010.
- B1 fixed-checkpoint pilot: 4 scenes / 108 expressions, 22.277 effective
  active queries, 25.773 ms/expression mean latency, 2161.027 MiB peak memory.
  Pilot subset mIoU does not replace full-validation mIoU.
- Evidence: pipeline/analyses/AdaRES-B1_profile_pilot_20260920-095000.json
  and pipeline/analyses/AdaRES-B1_done_complete_20260920-091444.md.

## Active Hypotheses
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
- B2 completed all 20 validations. True-IoU indicator supervision improved
  B1 by 0.003 mIoU but remained 0.010 below target. The TensorBoardX EOFError
  occurred after final validation and checkpoint save and is non-fatal.

## Running And Queue
- B2 completed 20/20 validations; best epoch 19 mIoU 0.505. Best checkpoint:
  /bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b2_metric_indicator_res/20260919_204155/best.pth.
- B3 was automatically launched at 2026-09-20 09:18:19 -0400 because B2
  remained below 0.515. PID 122629 is live on server-2 GPUs 0,1.
- B3 log: /bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_b3_20260920_091819.log.
- Latest live check: epoch 1/20 at 1208/4686 (26%); no validation yet.
- GPU0 used 10216/24576 MiB at 47%; GPU1 used 10128/24576 MiB at 39%.
- Queue and running map now consistently identify B3 as the sole running task.

## Open Questions And Risks
- Pending questions in snapshot manifest: none.
- Remote clock reports 2026-09-19 22:11:21 -0400 while local check occurred
  around 2026-09-20 10:42 +0800: roughly 31 minutes of skew after conversion.
  Use observed iteration advancement, not cross-host mtime alone, for liveness.
- B3 effect remains unknown until validation. B2/B3 efficiency metrics remain
  to be measured; do not substitute B1 pilot measurements.
- pipeline/closure_policy.json is absent. No due research-work IDs are listed;
  research_on_capacity_gap is not enabled in the supplied context/goal.

## Last Tick
- 2026-09-20 21:59 +0800 manual reconciliation.
- Confirmed B2 normal completion and B3 automatic launch with live training.
- Updated queue, running map, experiment states, ledger, local model entry,
  and B2 terminal analysis.
- Next scheduled Windows tick: 2026-09-20 22:47 +0800.
