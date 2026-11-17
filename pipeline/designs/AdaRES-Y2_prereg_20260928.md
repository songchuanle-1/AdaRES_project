# AdaRES-Y2: Target-size moderation of point-mass competition

Frozen 2026-09-28 12:36 Asia/Shanghai before cohort extraction.
Distinct from Y1's ranking-versus-oracle decomposition: this tests whether
W2's point-count weighting preferentially harms small-target expressions.

Use the same first 64 non-augmented training scenes and first 32 validation
scenes, paired W2 and B4 prediction rows from Y1. Independently extract GT
positive point count and scene point count per expression from the native
dataset. Join by scene and expression order; fail if any expression count or
ann ID differs. Define target fraction as positive / total points. Freeze
quartile boundaries using train64 only (Q1 and Q3), then apply the same
boundaries to val32. The mechanism predicts the W2-B4 selected-IoU difference
is more negative in the small-target group than the large-target group,
while W2-B4 oracle-IoU differences are much less size-dependent.

Control: within-scene rotate the target fractions by one expression before
assigning groups, preserving scene size mix but breaking expression-level
alignment. Report group counts and paired mean selected/oracle differences,
as well as scene means. No threshold or subgroup selection from validation.
This is a diagnostic moderation analysis, not pretraining admission,
significance, or permission to tune the count transform.
