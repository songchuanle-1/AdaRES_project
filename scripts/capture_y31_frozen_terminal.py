"""Reuse the frozen two-pass evaluator with Y31 checkpoint/source identity."""

import ast
import hashlib
import json
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_frozen_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "8adbfad10fa4c6e77ee4e89e926a04ac25e6d6f17403b60c235a900c14ce409a"
bundle_path = source_path.parents[1] / "pipeline/analyses/y31_source_bundle_lf_626ebd89c7094c31953d05dac9d0dc8d.json"
bundle = bundle_path.read_bytes()
assert hashlib.sha256(bundle).hexdigest() == "79badd8ec8d13504b1239b7260b182e8f1fedda4a0c65d1d54ed06e9a51cac7d"
files = json.loads(bundle)["files"]
assert len(files) == 7 and all(row["path"].startswith("AdaRES/") for row in files)


class Y31Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("adares_y18_point_measure_res.yaml", "adares_y31_mask_shape_res.yaml")
            value = value.replace("Y18", "Y31").replace("y18", "y31").replace("7449", "66677")
            if "s=p['server'];root=pathlib.Path" in value:
                value = value.replace(
                    "assert not pathlib.Path('/proc/66677').exists()",
                    "assert not pathlib.Path('/proc/66677').exists()\n"
                    "assert (root/'.codex_source_commit').read_text().strip()"
                    " == '5f2489de78196e8d89c004e452a665c0946f2a94'\n"
                    "assert not subprocess.check_output(['nvidia-smi','-i','1',"
                    "'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()",
                )
                value = value.replace(
                    "assert digest(p['last']['path'])",
                    "assert all(digest(root / row['path'][7:]) == row['sha256'] for row in "
                    + repr(files) + ")\nassert digest(p['last']['path'])",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Y31Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
