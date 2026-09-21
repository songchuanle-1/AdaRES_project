# AdaRES_project Research Memory

## Goal And Phase
- Phase 1: ScanRefer mIoU >= 0.515; not reached. Phase 2 awaits Phase 1.
- Root: D:/project/AdaRES_project; model code: AdaRES/.
- Remote code: /bak_home/songcl/Project/AdaRES_project/AdaRES.
- Use server-2 GPUs 0 and 1 only; never use server-6.
- All experiments launched after B3 use exactly one GPU and at most one task per allowed GPU.
- Train from the released SPFormer sparse-UNet backbone with the project-authored decoder.
- Do not tune thresholds, temperatures, loss weights, feature subsets, epochs, ranking margin, or occupancy loss weight.
- Report mIoU, Acc@0.25, Acc@0.5, active queries, latency, and peak memory when measured.

## Champion And Baseline
- AdaRES-B1 faithful baseline: mIoU 0.502, Acc@0.25 0.610, Acc@0.5 0.553, spmIoU 0.467; best epoch 18.
- AdaRES-B2 current champion: mIoU 0.505, Acc@0.25 0.607, Acc@0.5 0.555, spmIoU 0.471; best epoch 19. Target gap: 0.010.
- B2 checkpoint:
  /bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b2_metric_indicator_res/20260919_204155/best.pth.
- B1 efficiency pilot only: 4 scenes / 108 expressions, 22.277 active queries,
  25.773 ms/expression, 2161.027 MiB peak memory.

## Active Hypotheses
- AdaRES-B4 query-competition indicator: relative soft-mask overlap among queries should improve top-1 ranking when candidates compete.
- AdaRES-B5 token-grounding indicator: direct query-to-phrase token compatibility should add semantic evidence missing from confidence scoring.
- B4 and B5 are structurally independent and preserve B1 indicator supervision.
- Both are waiting for monitor dispatch; no training has started yet.
- `pipeline/research_work.json` records both as queued with concrete post-dispatch checks.

## Closed Or Failed Directions
- AdaRES-B0 closed at mIoU 0.333; strong oracle masks but failed query ranking.
- Applied fixes: shape-compatible pretrain loading, scalar loss reshape before DDP reduction, non-main-rank logger guard, and isolated criterion registration.
- B2 closed after 20/20 validations at mIoU 0.505; post-validation TensorBoardX EOFError was non-fatal.
- B3 closed after 20/20 validations at mIoU 0.5014, Acc@0.25 0.609,
  Acc@0.5 0.554, spmIoU 0.464; best epoch 20. It did not beat B1 or B2.
- Do not repeat B3 mask-evidence pooling or add scalar variations of failed routes.

## Running And Queue
- No experiment is running.
- Queue has two waiting single-GPU tasks:
  - AdaRES-B4 on server-2 GPU 0, `adares_b4_query_competition_res.yaml`.
  - AdaRES-B5 on server-2 GPU 1, `adares_b5_token_grounding_res.yaml`.
- Both depend on terminal B3 and have satisfied unit/real-scene smoke gates.
- Committed source: `03ca391804aa2b347acae00090f656473b6ab1a6`.
- Remote `AdaRES/.codex_source_commit` matches local HEAD.

## Open Questions And Risks
- Pending questions in the snapshot manifest: none.
- B4 uses differentiable soft overlap, not exact mask intersection; zero-overlap samples supply no competition evidence.
- B5 may overlap with existing decoder language attention; a null result may indicate redundancy.
- Real-scene smoke does not measure full-run throughput, peak training memory, or convergence.
- B2 remains champion; no B3 efficiency measurement is available.
- `pipeline/closure_policy.json`, project-local `formats.md`,
  `references/autonomous_design.md`, and `log_decision.sh` are absent.

## Last Tick
- 2026-09-21 14:50 +08:00 design tick; monitor_rc=0, goal_rc=0.
- Candidate budget 2; both admissible candidates passed static, unit, and one-scene native smoke gates.
- Added B4/B5 packages, configs, tests, motivation review, design, decision, and queue entries.
- Local syntax checks passed; server-2 unit and real-scene smoke tests passed.
- Report: `pipeline/reports/summary_20260921-145043.md`.
- Next: monitor dispatch, verify each single-GPU process starts and progresses beyond PID-only evidence.
