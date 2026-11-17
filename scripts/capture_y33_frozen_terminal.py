"""Use the same two-pass protocol with exact Y33 deployment identity."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / "scripts/capture_y18_frozen_terminal.py"
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "8adbfad10fa4c6e77ee4e89e926a04ac25e6d6f17403b60c235a900c14ce409a"
bundle = (ROOT / "pipeline/analyses/y33_source_bundle_lf_cde054a6b0254e37a0c0d71febbcc497.json").read_bytes()
assert hashlib.sha256(bundle).hexdigest() == "b08ee31f6a5ede73a2064f1ff2edfb52f5eb42ea5cf3cf5c868494c723b41d61"
files = json.loads(bundle)["files"]


class Identity(ast.NodeTransformer):
    def visit_Assert(self, node):
        if ast.unparse(node.test) == "server['allowed_gpus'] == [1]":
            return ast.copy_location(ast.parse("assert server['allowed_gpus'] == [0, 1]").body[0], node)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("adares_y18_point_measure_res.yaml", "adares_y33_token_attention_res.yaml")
            value = value.replace("Y18", "Y33").replace("y18", "y33").replace("7449", "64797")
            if "s=p['server'];root=pathlib.Path" in value:
                value = value.replace(
                    "assert digest(p['last']['path'])",
                    "assert (root/'.codex_source_commit').read_text().strip() == "
                    "'b49b744dc2943a38ab4c3aeb788acc6a2cb90011'\n"
                    "assert not subprocess.check_output(['nvidia-smi','-i','1',"
                    "'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()\n"
                    "assert all(digest(root / row['path'][7:]) == row['sha256'] for row in "
                    + repr(files) + ")\nassert digest(p['last']['path'])",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path)})
