# AdaRES-Y5: Exact native indicator label coverage

Frozen 2026-09-28 14:48 Asia/Shanghai before native execution. The
predeclared question is whether the native **complete** indicator positives,
including in-object sampled seeds and the forced nearest query per target,
omit high-IoU oracles among W2's selection errors. Y4 only measured the
in-object subset and cannot settle this question.

Use the released W2 best checkpoint without weight changes on the first 64
non-augmented train scenes (1660 expressions), and first 32 validation scenes
(851 expressions). Cache the exact inference-expanded superpoint positions
from `expand_and_fps`. For each expression, construct target centroids with
the native criterion's superpoint instance labels and call its
`match_sampled_ins` with the same sampled seed IDs and positions. Compare
the returned positive index set with the oracle and selected query on all
expressions and the fixed material-error subgroup (oracle IoU minus selected
IoU > 0.05). Use a within-expression cyclic shift of the **entire native
positive set** as a non-semantic count-preserving control. Double-forward
each scene, report repeat mismatches and scene means; strict key alignment
with Y1/Y4 is required.

The omitted-oracle mechanism requires an error-oracle positive rate lower
than the selected-query rate, with the real-positive oracle rate above the
shifted control. If instead the native positives favor error oracles, reject
the proposed label-omission realization. Do not infer training benefit from
coverage counts, alter the training loss, tune validation choices, or relabel
Y4. A positive diagnosis would still require a distinct trainable remedy,
matched intervention, native backward, resource and independence checks.
