# Y15 Material-Subgroup Transfer Supplement

Created 2026-09-30 before this probe, after the all-expression native probe
found zero material instances among its first eight expressions.

Select the first two distinct scan IDs, sorted lexicographically, with at least
one material event in the frozen train64 Y15 diagnosis (b4_oracle-w2_oracle>0.05).
For each, use at most the first four material expressions in the frozen row
order, plus the first different-object expression in its original dataset item
for the language control. No validation labels or outcomes enter selection.
This is explicitly a train-diagnosis-selected subgroup, not an independent
efficacy estimate. Run the unchanged native gradient probe on this subgroup.

Additionally report the fraction of teacher mask-head gradient norm orthogonal
to the GT mask-head gradient: sqrt(max(0,1-cosine^2)). A small residual would
show redundancy; a residual does not alone show a useful new direction.
Report teacher gradient cosine to GT and wrong-language control separately.
Same scene/weights/IoU definition, no optimization or thresholds changed.
Do not infer "ordinary teacher BCE is innovative" from gradient reachability.
Negative or redundant gradients constrain the subsequent design.
Raw forward repeat and material status are reported anew; the frozen subgroup
is not filtered to remove instances whose new capture no longer qualifies.
