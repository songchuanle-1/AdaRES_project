"""Mechanically adapt previous canonical dispatcher checks to tested Y33."""

import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "scripts/admit_dispatch_y31.py"
output = root / "scripts/admit_dispatch_y33.py"
assert not output.exists()
replacements = {
    "adares_y31_mask_shape": "adares_y33_token_attention",
    "y31_shape_diagnosis": "y33_token_diagnosis",
    "626ebd89": "cde054a6",
    "Y31": "Y33", "y31": "y33",
    "missed_with_distinct_shape": "wrong_tokens_nonzero_scans",
    "wrong_shape_min": "wrong_masks_min",
    "uniform_shape_min": "wrong_tokens_min",
    "permutation_translation_max": "permutation_max",
    "parent_explicit_shape_input_absent": "equal_fg_bg_first_moments",
    "Y31 mask-supported spatial shape confidence": "Y33 final mask-conditioned token attention confidence",
    "Superpoint-centroid covariance approximation, adverse initial losses, wrong/uniform shape learnability, resume numerical differences and measured cost regressions retained. No trained efficacy, necessity, Pareto or novelty claim.":
        "Token attention direct path; upstream attention exists. Original adverse trained-parent loss, wrong-mask/token learnability, resume numerical differences and measured cost regressions retained. No trained efficacy, necessity, Pareto or novelty claim.",
}


class Adapt(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for a in sorted(replacements, key=len, reverse=True):
                node.value = node.value.replace(a, replacements[a])
        elif node.value == 133634:
            node.value = 328450
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        for v in node.comparators:
            if isinstance(v, ast.Constant) and v.value == 6:
                v.value = 7
        return node


tree = ast.fix_missing_locations(Adapt().visit(ast.parse(source.read_text())))
compile(tree, str(output), "exec")
output.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
print(str(output))
