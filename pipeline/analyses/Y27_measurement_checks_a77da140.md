# Y27 Measurement Checks
Owned event a77da1408e7d4f289f6dfcb5cc768332, 2026-10-05.

- New terminal/full-eval adapters reuse hash-pinned Y18 helpers through AST
  constant transformations, binding Y27 config/PID9901/source359a89d only.
  No candidate model, training, checkpoint, validation population, epoch count,
  threshold, loss weight or admission gate was edited.
- Four new measurement entries compiled successfully. The frozen evaluator AST
  bound the exact Y27 config, original PID and source marker. Actual terminal
  capture passed; actual Y27 and W2 two-pass captures passed. B1, clustered
  comparison and serial trained cost remain pending at this document's writing.
- `scripts/test_y27_frozen_analysis.py`: five synthetic arithmetic/contract
  checks passed in 1.361 seconds. They cover identity mismatch, negative-CI
  frozen harm rule, non-cancelling absolute repeat noise, population-shape
  failure and a zero interval. These tests are NOT scientific efficacy evidence
  or an equivalence claim. The analysis reuses whole-scene NumPy bootstrap,
  seed1999/10000 draws and frozen size cuts1487/3325.
- Historical analyzer noise fields remain present; the corrected mean absolute
  repeat difference is separate. The preregistered harm decision remains
  CI upper bound below zero, not a new threshold adjusted to observed noise.
- One direct owner.json read unexpectedly expanded embedded `work_before`
  history. It was not treated as a new scientific snapshot. Further ownership
  checks use selected token/thread/deadline fields, not the full lease object.
- Prior independent lock-absence probes checked `pipeline/.foreground.lockdir`,
  not the actual `pipeline/headless/.foreground.lockdir`. Historical successful
  finish/COMPLETE outputs are preserved, but a correct-path historical
  independent verification is not established by those probes. This tick must
  verify the actual headless lock after finish. Historical reports/evidence
  are not rewritten or retroactively certified.

## Source Hashes
| File | SHA256 |
|---|---|
| scripts/capture_y27_terminal.py | 9749c6d5f8344ed28f1c22de1841212d6482fa37fe08503b053591d172f5b1f7 |
| scripts/capture_y27_frozen_terminal.py | 4ac77d32eb7c8fa6abc6c28efc51cf1dac41dfe63654267a71f2c4572dee0882 |
| scripts/analyze_y27_frozen_terminal.py | 7360a2f6117af443db24459947e6304987d56a5f636fdacc6192d5b003f1f875 |
| scripts/probe_y27_trained_cost.py | 8792ea4d000458ffda87891cd1e81f6082e2be957ce0c1dd7467a191e855c86d |
| scripts/test_y27_frozen_analysis.py | 910eb17c82fe938da6a5135236e5f6a93e6072f95c99183ed74dba705739129a |
| AdaRES/tools/evaluate_adares_frozen_full.py | 7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13 |
| pipeline/designs/exp_design_Y27_11c9ed2a.md | 76a13eddf5b61e86a6c9f3d94549a2d4f285baffb9b4618de5895344bf2951f0 |
