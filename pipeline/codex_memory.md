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
- AdaRES-B4-E30 is running on server-2 GPU 0, PID 52617. Verified log:
  `resume_adares_b4_query_competition_res_20260922_093323.log`; resume began
  at epoch 21 with best mIoU 0.5072 and training is progressing.
- AdaRES-B2-E30 is running on server-2 GPU 1, PID 56854. Verified log:
  `resume_adares_b2_metric_indicator_res_20260922_093709.log`; resume began
  at epoch 21 with best mIoU 0.5046 and `Train[21/30]` is active.
- `queue.json` and `running.json` now model both extensions as the two active
  slots, preventing duplicate dispatch or unrelated design while they run.
- AdaRES-B5 is closed at mIoU 0.5030 after 20/20 validations.

## Open Questions And Risks
- Pending questions in the snapshot manifest: none.
- B5 has not beaten the B4 champion through its latest validation.
- B5 may be partly redundant with existing decoder language attention.
- B4 active-query count, latency, and peak memory remain unmeasured.
- `pipeline/closure_policy.json`, project-local `formats.md`,
  `references/autonomous_design.md`, and project-local `log_decision.sh` are
  absent; the shared skill provides the decision logger.

## Last Tick
- 2026-09-22 09:30-09:42 +08:00 manual protocol correction.
- Fixed B2 and B4 at 30 total epochs, resumed both from epoch-20 `last.pth`
  without resetting optimizer or PolyLR state, and launched one per GPU.
- Fixed B2's duplicate loss-registry error with `force=True` in commit
  `f23036b`; verified both processes beyond PID existence.
- Next: monitor validations 21-30, preserve the fixed protocol, and compare
  B2-E30 versus B4-E30 on the same training budget.
