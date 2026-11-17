"""Reuse the frozen two-pass evaluator with Y30 checkpoint/source identity."""

import ast
import hashlib
import json
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_frozen_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "8adbfad10fa4c6e77ee4e89e926a04ac25e6d6f17403b60c235a900c14ce409a"
bundle_path = source_path.parents[1] / "pipeline/analyses/y30_source_bundle_fp64_86766fcd883d46c1a9d60563f4213c2e.json"
bundle = bundle_path.read_bytes()
assert hashlib.sha256(bundle).hexdigest() == "b5aa308e27bd394ef5ade195c45e18855bd7c24cad3ff568b75a17c9738b1727"
files = json.loads(bundle)["files"]
assert len(files) == 7 and all(row["path"].startswith("AdaRES/") for row in files)


class Y30Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("adares_y18_point_measure_res.yaml", "adares_y30_query_trajectory_res.yaml")
            value = value.replace("Y18", "Y30").replace("y18", "y30").replace("7449", "39709")
            if "s=p['server'];root=pathlib.Path" in value:
                value = value.replace(
                    "assert not pathlib.Path('/proc/39709').exists()",
                    "assert not pathlib.Path('/proc/39709').exists()\n"
                    "assert (root/'.codex_source_commit').read_text().strip()"
                    " == '4f5c31f57994b8f9fb640d652c913863dc310a95'",
                )
                value = value.replace(
                    "assert digest(p['last']['path'])",
                    "assert all(digest(root / row['path'][7:]) == row['sha256'] for row in "
                    + repr(files) + ")\nassert digest(p['last']['path'])",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Y30Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
