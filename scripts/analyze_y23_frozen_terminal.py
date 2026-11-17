"""Reuse the frozen paired whole-scene bootstrap for Y23."""

import ast
import hashlib
import statistics
from pathlib import Path

from analyze_frozen_full import comparison as historical_comparison, keyed


def comparison(candidate, reference):
    results = historical_comparison(candidate, reference)
    first = [keyed(c["passes"][0]["rows"]) for c in (candidate, reference)]
    second = [keyed(c["passes"][1]["rows"]) for c in (candidate, reference)]
    assert all(a.keys() == b.keys() for a, b in zip(first, second))
    for group, result in results.items():
        if "scene_clustered_ci95" not in result:
            continue
        selected = [k for k, row in first[0].items() if group == "all" or row["size"] == group]
        noise = max(statistics.mean(abs(a[k]["piou"] - b[k]["piou"]) for k in selected)
                    for a, b in zip(first, second))
        result["historical_abs_mean_repeat_difference"] = result.pop("observed_group_mean_repeat_noise")
        result["observed_group_mean_absolute_repeat_difference"] = noise
        lower, upper = result["scene_clustered_ci95"]
        result["material_harm_within_observed_noise"] = upper < -noise
        result["positive_difference_beyond_observed_noise"] = lower > noise
    return results

source_path = Path(__file__).with_name("analyze_y18_frozen_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "966d40b34402f0ea779e357a4edad5aac5054fade627a3c975584b97e1d5e719"


class Y23Identity(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "analyze_frozen_full":
            return ast.copy_location(ast.Pass(), node)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y18", "Y23").replace("y18", "y23")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.parse(source, filename=str(source_path))
adapted = ast.fix_missing_locations(Y23Identity().visit(tree))
if __name__ == "__main__":
    exec(compile(adapted, str(source_path), "exec"),
         {"__name__": "__main__", "__file__": str(source_path), "comparison": comparison})
