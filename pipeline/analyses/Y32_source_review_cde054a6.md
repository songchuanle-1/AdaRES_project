# Y32 Source Discrimination

Date: 2026-10-08. Owned tick cde054a6b0254e37a0c0d71febbcc497.

## Actual Test

The preregistered CPU audit executed AST-extracted historical W2, G1, C1 and
B0/Y29-reference losses on 32 deterministic cached cases. Equal-mass exact
point-IoU matches actual C1 target quality within 1e-7. C1 and the proposed
normalized-quality target have identical order. A nonmatched-query quality
intervention changes both losses in 31/32 cases while W2 is unchanged.

## Disposition

Reject this realization's claimed new all-query quality information pathway:
the actual C1 source and executed interventions already implement that pathway.
The candidate is NOT numerically identical to C1. Its normalization, point
measure and replace-versus-add objective differences remain real; they do not
substantiate the preregistered independent information-path hypothesis and are
not sufficient grounds to launch a numeric recalibration-only successor here.
This is source-mechanism counterevidence, NOT a trained-harm result or a ban on
all quality-supervision research. No validation-based search occurred.

## Limitations And Failures

Unequal point measure, fallback behavior and trained efficacy were not isolated
in population tests. The first probe failed because native W2 allocates float32
targets while the harness supplied float64 logits. Preserve that log and v1
source; the v2 correction changes logits dtype only, not the frozen objective.
Neither failure is a production-training crash. All finite gradients in v2
establish mechanics only.

## Evidence

- Frozen prereg: pipeline/analyses/Y32_diagnosis_prereg_cde054a6.md.
- Executed result: pipeline/analyses/y32_source_discrimination_v2_cde054a6b0254e37a0c0d71febbcc497.json, SHA256 955ceb04c6d41d8e3f642ce89bfd62790ed793ed776739d1512cdaec49575041.
- Actual observation: 2026-10-08 17:59:29.079503 to 17:59:31.164838+08.
- No GPU queried/used by this CPU audit; GPU0 remains reserved.
