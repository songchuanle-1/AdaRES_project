"""Build only an independent final-indicator target realization of SAME Y48."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 600
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
target = parent.parent / "adares_y48_mask_equivalence"
config = ROOT / "AdaRES/configs_song/model_paper5/adares_y48_mask_equivalence_res.yaml"
assert not target.exists() and not config.exists()
assert hashlib.sha256((parent / "loss.py").read_bytes()).hexdigest() == (
    "836ff790a0a8b9c55cf465a472c9462699f0753a7e87d0ef74a8946edf31565a")
tree = ast.parse((parent / "loss.py").read_text(encoding="utf-8"))
original_ast = ast.dump(tree)
helper = ast.parse((ROOT / "scripts/y48_equivalence_targets_601b.py").read_text()).body[-1]
assert isinstance(helper, ast.FunctionDef) and helper.name == "close_nonempty_mask_indices"
criterion = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "Criterion")
loss = next(node for node in criterion.body
            if isinstance(node, ast.FunctionDef) and node.name == "loss_indi")
forward = next(node for node in criterion.body
               if isinstance(node, ast.FunctionDef) and node.name == "forward")
assert len(loss.args.args) == 5 and not loss.args.defaults
loss.args.args.extend([ast.arg(arg="mask_pred"), ast.arg(arg="mask_padding")])
loss.args.defaults.extend([ast.Constant(value=None), ast.Constant(value=None)])
guard = ast.parse(
    "if mask_pred is not None:\n"
    "    indices = close_nonempty_mask_indices(mask_pred, mask_padding, indices)\n").body[0]
loss.body.insert(0, guard)
calls = [node for node in ast.walk(forward)
         if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
         and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
         and node.func.attr == "loss_indi"]
assert len(calls) == 1 and len(calls[0].args) == 4
calls[0].args.extend(ast.parse("f(pred['masks'], pred['batch_mask'])").body[0].value.args)
tree.body.insert(tree.body.index(criterion), helper)
ast.fix_missing_locations(tree)
compile(tree, "<independent-y48>", "exec")
audit = ast.parse(ast.unparse(tree))
audit.body = [node for node in audit.body
              if not (isinstance(node, ast.FunctionDef) and node.name == helper.name)]
audit_criterion = next(node for node in audit.body
                       if isinstance(node, ast.ClassDef) and node.name == "Criterion")
for node in audit_criterion.body:
    if isinstance(node, ast.FunctionDef) and node.name == "loss_indi":
        node.args.args = node.args.args[:-2]
        node.args.defaults = node.args.defaults[:-2]
        node.body = node.body[1:]
    if isinstance(node, ast.FunctionDef) and node.name == "forward":
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "loss_indi":
                call.args = call.args[:-2]
assert ast.dump(audit) == original_ast
target.mkdir()
for source in parent.glob("*.py"):
    shutil.copyfile(source, target / source.name)
(target / "loss.py").write_text(ast.unparse(tree) + "\n", encoding="utf-8", newline="\n")
shutil.copyfile(config.parent / "adares_w2_pointmass_competition_res.yaml", config)
files = sorted(target.glob("*.py")) + [config]
assert len(files) == 7
result = dict(
    complete=True, id="AdaRES-Y48",
    files=[dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files],
    parent_source_hashes={path.relative_to(ROOT / "AdaRES").as_posix():
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in parent.glob("*.py")},
    stripped_intervention_AST_exact_parent=True, unchanged_inference_files=5,
    final_only=True, new_parameters=0, configuration_byte_identical_parent=True,
    limitations="Local independent files only; production tests, receipt and admission still required.")
output = ROOT / ("pipeline/analyses/y48_local_bundle_601b_" + args.token + ".json")
with output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
