"""Same-interpreter whole-module equivalence after removing only the new branch."""

import ast
import hashlib
from pathlib import Path
import yaml

base = Path("ipdn/model_paper5")
parent_path = base / "adares_w2_pointmass_competition/dec.py"
candidate_path = base / "adares_y41_boundary_probability/dec.py"
parent = ast.parse(parent_path.read_text(encoding="utf-8"))
candidate = ast.parse(candidate_path.read_text(encoding="utf-8"))
removed = {"class": 0, "constructor": 0, "final_addition": 0}
body = []
for node in candidate.body:
    if isinstance(node, ast.ClassDef) and node.name == "MaskBoundaryProbabilityReadout":
        removed["class"] += 1
    else:
        body.append(node)
candidate.body = body
decoder = next(n for n in candidate.body if isinstance(n, ast.ClassDef) and n.name == "DEC")
for method in decoder.body:
    if not isinstance(method, ast.FunctionDef):
        continue
    body = []
    for statement in method.body:
        constructor = (
            isinstance(statement, ast.With) and any(
                isinstance(n, ast.Attribute) and n.attr == "boundary_readout"
                for n in ast.walk(statement))
        )
        addition = (
            isinstance(statement, ast.Assign) and any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "boundary_readout"
                for n in ast.walk(statement))
        )
        if constructor or addition:
            removed["constructor" if constructor else "final_addition"] += 1
        else:
            body.append(statement)
    method.body = body
assert removed == {"class": 1, "constructor": 1, "final_addition": 1}
assert ast.dump(parent) == ast.dump(candidate), "Parent module differs beyond new branch"
parent_cfg = yaml.safe_load(Path("configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml").read_text())
candidate_cfg = yaml.safe_load(Path("configs_song/model_paper5/adares_y41_boundary_probability_res.yaml").read_text())
differences = []


def compare(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            compare(left.get(key), right.get(key), prefix + "." + str(key))
    elif left != right:
        differences.append({"key": prefix, "parent": left, "candidate": right})


compare(parent_cfg, candidate_cfg)
result = {
    "complete": True, "id": "AdaRES-Y41", "removed_exactly": removed,
    "whole_decoder_module_parent_identity": True,
    "parent_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
    "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    "config_differences": differences,
    "parser_version": sys.version,
    "limitations": ["Cross-Python-version ast.dump hashes are not comparable proof.",
                    "Earlier source-bundle field was based on non-DEC class comparison; "
                    "this fresh whole-module test supplies the missing stronger assertion."],
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
