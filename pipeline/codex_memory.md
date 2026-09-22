# AdaRES_project Research Memory

## Goal And Phase
- Phase 1: ScanRefer mIoU >= 0.515; not reached. Phase 2 awaits Phase 1.
- Root: D:/project/AdaRES_project; model code: AdaRES/.
- Remote code root: /bak_home/songcl/Project/AdaRES_project/AdaRES.
- Use server-2 GPUs 0 and 1 only; never use server-6.
- New experiments use one GPU per task, one task per allowed GPU.
- The user-approved extension protocol is 30 total epochs. B2 and B4 resume
  from their epoch-20 last checkpoints on separate GPUs; optimizer and PolyLR
  state are preserved, and the epoch count must not be tuned.
- Train from the released SPFormer sparse-UNet backbone with the project decoder.
- Do not tune thresholds, temperatures, loss weights, feature subsets, epochs,
  ranking margin, or occupancy loss weight.

## Champion And Baseline
- AdaRES-B1 baseline: mIoU 0.502, Acc@0.25 0.610, Acc@0.5 0.553, spmIoU 0.467.
- AdaRES-B4 champion: mIoU 0.5070, Acc@0.25 0.607, Acc@0.5 0.556,
  spmIoU 0.472, spAcc@0.5 0.524, spAcc@0.25 0.600; target gap 0.0080.
- B4 checkpoint:
  /bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b4_query_competition_res/20260921_032553/best.pth.
- AdaRES-B2 former champion: mIoU 0.505.

## Active Hypotheses
- AdaRES-B5 token-grounding indicator: direct query-to-phrase token
  compatibility should add semantic evidence missing from confidence scoring.
- B5 preserves B1 indicator supervision and remains structurally independent
  from B4 query competition.

## Closed Or Failed Directions
- AdaRES-B0 closed at mIoU 0.333; oracle masks were strong but ranking failed.
- AdaRES-B1 completed as the faithful baseline at mIoU 0.502.
- AdaRES-B2 completed at mIoU 0.505.
- AdaRES-B3 closed at mIoU 0.5014; do not repeat mask-evidence pooling.
- AdaRES-B4 closed improved at mIoU 0.5070, best at epoch 19; it beat B2 by
  0.0020 but missed the target by 0.0080. Do not launch a duplicate or
  tuning-only B4 successor.
- B3 and B5 remain closed at their historical 20-epoch protocols. Do not
  compare a 30-epoch extension against them without labeling the mismatch.

## 30-Epoch Extension
- At 2026-09-22 the user approved training up to 30 epochs. B2 and B4 are the
  first paired extensions because they are the top two variants and remain
  below the target.
- Resume sources:
  B2 `20260919_204155/last.pth`;
  B4 `20260921_032553/last.pth`.
- Launch B2 on GPU1 and B4 on GPU0, one single-GPU process each. Do not modify
  the running extension's epoch setting or restart it from pretrain.

## Running And Queue
- AdaRES-B5 is running on server-2 GPU 1.
- At 2026-09-22 08:53 +08:00 B5 was in epoch 20/20 at iteration 1,686/4,686;
  one Python process used 17,184 MiB on GPU 1.
- Preflight monitor reported B5 best validation mIoU 0.5008.
- AdaRES-B4 is terminal and absent from the active queue; server-2 GPU 0 is idle.
- Active queue is below its two-GPU target, but this event tick is not a design
  tick.

## Open Questions And Risks
- Pending questions in the snapshot manifest: none.
- B5 has not beaten the B4 champion through its latest validation.
- B5 may be partly redundant with existing decoder language attention.
- B4 active-query count, latency, and peak memory remain unmeasured.
- `pipeline/closure_policy.json`, project-local `formats.md`,
  `references/autonomous_design.md`, and project-local `log_decision.sh` are
  absent; the shared skill provides the decision logger.

## Last Tick
- 2026-09-22 08:53 +08:00 event tick; monitor_rc=0, goal_rc=0.
- Closed B4 at mIoU 0.5070 after 20 validations and promoted it to champion.
- Verified B5 training progress, repaired the experiment ledger, updated
  research-work state, terminal analysis, and durable memory.
- Report: `pipeline/reports/summary_20260922-085300.md`.
- Next: monitor B5's final validation and terminal transition; if B5 stops
  improving and no design is pending, use the next design tick to propose one
  structurally distinct candidate for the idle GPU.
