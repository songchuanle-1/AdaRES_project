"""Bind the unchanged frozen two-pass evaluator to Y36 and preserve Y37."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / "scripts/capture_y18_frozen_terminal.py"
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "8adbfad10fa4c6e77ee4e89e926a04ac25e6d6f17403b60c235a900c14ce409a"
files = []
for name, expected in (
    ("y36_source_bundle_lf_8fd5e469dee241da8f21da7c870c0d3f.json",
     "96a2e9621d96847b8621438813dc3a1cbc07e3ac883a61f163a526db04cf5f39"),
    ("y37_source_bundle_lf_e636ce91a12e45488093b07fa2fc092b.json",
     "b7706f769091980c07769f988a9bd7312071c6c48163c220a805f80c4a2ea282"),
):
    raw = (ROOT / "pipeline/analyses" / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    files.extend(json.loads(raw)["files"])


class Identity(ast.NodeTransformer):
    def visit_Assert(self, node):
        if ast.unparse(node.test) == "server['allowed_gpus'] == [1]":
            return ast.copy_location(
                ast.parse("assert server['allowed_gpus'] == [0, 1]").body[0], node
            )
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace(
                "adares_y18_point_measure_res.yaml",
                "adares_y36_local_adjacency_res.yaml",
            )
            value = value.replace("Y18", "Y36").replace("y18", "y36")
            value = value.replace("7449", "100018")
            if "s=p['server'];root=pathlib.Path" in value:
                guards = (
                    "assert (root/'.codex_source_commit').read_text().strip() == "
                    "'67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773'\n"
                    "assert not subprocess.check_output(['nvidia-smi','-i','1',"
                    "'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()\n"
                    "old=pathlib.Path('/proc/20348/cmdline').read_bytes().split(b'\\0')\n"
                    "assert b'tools/train.py' in old and "
                    "b'configs_song/model_paper5/adares_y37_pair_distribution_res.yaml' in old\n"
                    "assert old[old.index(b'--gpu_ids')+1] == b'0'\n"
                    "assert all(digest(root / row['path'][7:]) == row['sha256'] for row in "
                    + repr(files) + ")\n"
                )
                value = value.replace(
                    "assert digest(p['last']['path'])",
                    guards + "assert digest(p['last']['path'])",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(
    compile(tree, str(source_path), "exec"),
    {"__name__": "__main__", "__file__": str(source_path)},
)
