"""Compare relevant native functions across Python AST versions."""

import ast
import hashlib
import json
from pathlib import Path
import sys

NAMES = {"get_iou", "dice_loss", "loss_masks", "loss_indi", "_get_src_permutation_idx"}


def canonical(node):
    if isinstance(node, ast.AST):
        if type(node).__name__ == "Index":
            return canonical(node.value)
        return [type(node).__name__, {key: canonical(value) for key, value in ast.iter_fields(node)
                                     if key not in ("type_params", "type_comment")}]
    if isinstance(node, list):
        return [canonical(value) for value in node]
    return node


path = (Path("ipdn/model_paper5/adares_w2_pointmass_competition/loss.py") if "P" in globals()
        else Path(__file__).resolve().parents[1] / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/loss.py")
source = path.read_bytes()
nodes = {n.name: canonical(n) for n in ast.walk(ast.parse(source))
         if isinstance(n, ast.FunctionDef) and n.name in NAMES}
assert nodes.keys() == NAMES
digest = hashlib.sha256(json.dumps(nodes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
result = dict(complete=True, raw_sha256=hashlib.sha256(source).hexdigest(),
              canonical_functions_sha256=digest, functions=sorted(nodes),
              python=sys.version.split()[0], scope="Only these five AST function bodies; not entire package identity.")
if "P" in globals():
    Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
else:
    if len(sys.argv) == 2:
        remote_source = Path(sys.argv[1]).read_bytes()
        remote_nodes = {n.name: canonical(n) for n in ast.walk(ast.parse(remote_source))
                        if isinstance(n, ast.FunctionDef) and n.name in NAMES}
        result["same_interpreter_remote_sha256"] = hashlib.sha256(remote_source).hexdigest()
        result["same_interpreter_functions_equal"] = nodes == remote_nodes
        result["same_interpreter_module_equal"] = canonical(ast.parse(source)) == canonical(ast.parse(remote_source))
        assert result["same_interpreter_functions_equal"], "Relevant deployed functions differ"
    print(json.dumps(result))
