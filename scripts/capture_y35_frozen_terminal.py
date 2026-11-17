"""Run the frozen two-pass evaluation with exact Y35 identity on GPU0."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / "scripts/capture_y18_frozen_terminal.py"
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "8adbfad10fa4c6e77ee4e89e926a04ac25e6d6f17403b60c235a900c14ce409a"
bundle_path = ROOT / "pipeline/analyses/y35_source_bundle_lf_a8d0c77d45d142c0a6c46b163b956253.json"
bundle = bundle_path.read_bytes()
assert hashlib.sha256(bundle).hexdigest() == "7dc4bbfd4fbca46cbda803818c77b76e08b2b01b9201ed956bd7cabcffd3bc0a"
files = json.loads(bundle)["files"]


class Identity(ast.NodeTransformer):
    def visit_Assert(self, node):
        if ast.unparse(node.test) == "server['allowed_gpus'] == [1]":
            return ast.copy_location(ast.parse("assert server['allowed_gpus'] == [0, 1]").body[0], node)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("adares_y18_point_measure_res.yaml", "adares_y35_language_evidence_res.yaml")
            value = value.replace("Y18", "Y35").replace("y18", "y35").replace("7449", "108331")
            value = value.replace("'--gpu-id','1'", "'--gpu-id','0'")
            if "s=p['server'];root=pathlib.Path" in value:
                value = value.replace(
                    "assert digest(p['last']['path'])",
                    "assert (root/'.codex_source_commit').read_text().strip() == "
                    "'6725f9bd6a8f9b47addae6a4b8336d004f73f1ae'\n"
                    "assert not subprocess.check_output(['nvidia-smi','-i','0',"
                    "'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()\n"
                    "assert all(digest(root / row['path'][7:]) == row['sha256'] for row in "
                    + repr(files) + ")\nassert digest(p['last']['path'])",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path)})
