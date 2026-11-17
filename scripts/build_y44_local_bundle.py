"""Independent W2 copy with native coordinate-feature final confidence input."""
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
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
target = parent.parent / "adares_y44_coordinate_feature"
config = ROOT / "AdaRES/configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml"
assert not target.exists() and not config.exists()


def statements(text):
    return ast.parse(text).body


def methods(tree, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return cls, {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


dec = ast.parse((parent / "dec.py").read_text(encoding="utf-8"))
cls, method = methods(dec, "DEC")
readout = ast.parse((ROOT / "scripts/y44_cross_readout.py").read_text())
readout_class = next(n for n in readout.body if isinstance(n, ast.ClassDef))
dec.body.insert(dec.body.index(cls), readout_class)
method["__init__"].body.extend(statements(
    "with torch.random.fork_rng(devices=[]):\n"
    "    self.point_readout = MaskCoordinateFeatureReadout(d_model, in_channel)\n"))
for name in ("forward_iter_pred", "forward"):
    method[name].args.args.append(ast.arg(arg="point_cross"))
    method[name].args.defaults.append(ast.Constant(value=None))
forward = method["forward_iter_pred"]
mass_nodes = [n for n in forward.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "point_mass" for t in n.targets)
              and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "squeeze"]
assert len(mass_nodes) == 1
position = forward.body.index(mass_nodes[0]) + 1
forward.body[position:position] = statements(
    "if point_cross is None:\n"
    "    raise ValueError('Native coordinate-feature cross moments are required')\n"
    "point_cross, _ = self.get_batches(point_cross, batch_offsets)\n")
returns = [n for n in forward.body if isinstance(n, ast.Return)]
assert len(returns) == 1 and isinstance(returns[0].value, ast.Dict)
position = forward.body.index(returns[0])
forward.body[position:position] = statements(
    "pred_indis = pred_indis + self.point_readout("
    "query, pred_masks, batch_mask, point_mass, point_cross)\n")
calls = [n for n in ast.walk(method["forward"]) if isinstance(n, ast.Call)
         and isinstance(n.func, ast.Attribute) and n.func.attr == "forward_iter_pred"]
assert len(calls) == 1
calls[0].args.append(ast.Name(id="point_cross", ctx=ast.Load()))

ipdn = ast.parse((parent / "ipdn.py").read_text(encoding="utf-8"))
_, method = methods(ipdn, "IPDN")
for name in ("loss", "predict"):
    function = method[name]
    extraction = [n for n in function.body if isinstance(n, ast.Assign)
                  and isinstance(n.value, ast.Call)
                  and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "extract_feat"]
    assert len(extraction) == 1
    extraction[0].targets = [ast.Tuple(
        elts=[ast.Name(id="sp_feats", ctx=ast.Store()), ast.Name(id="point_cross", ctx=ast.Store())],
        ctx=ast.Store())]
    extraction[0].value.args.append(ast.Name(id="coords_float", ctx=ast.Load()))
    position = function.body.index(extraction[0]) + 1
    function.body[position:position] = statements(
        "point_cross = torch.cat([\n"
        "    point_cross[int(batch_offsets[i]):int(batch_offsets[i+1])].repeat(int(scenes_len[i]), 1)\n"
        "    for i in range(len(scenes_len))\n"
        "])\n")
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "dec"]
    assert len(calls) == 1
    calls[0].args.append(ast.Name(id="point_cross", ctx=ast.Load()))
function = method["extract_feat"]
function.args.args.append(ast.arg(arg="coords_float"))
feature_nodes = [n for n in function.body if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Subscript)
                 and isinstance(n.value.value, ast.Attribute) and n.value.value.attr == "features"]
assert len(feature_nodes) == 1
position = function.body.index(feature_nodes[0]) + 1
function.body[position:position] = statements(
    "mean_features = scatter_mean(x, superpoints, dim=0)\n"
    "mean_coords = scatter_mean(coords_float, superpoints, dim=0)\n"
    "point_cross = scatter_mean(\n"
    "    ((coords_float - mean_coords[superpoints]).unsqueeze(-1) *\n"
    "     (x - mean_features[superpoints]).unsqueeze(1)).flatten(1), superpoints, dim=0)\n")
returns = [n for n in function.body if isinstance(n, ast.Return)]
assert len(returns) == 1
returns[0].value = ast.Tuple(
    elts=[ast.Name(id="x", ctx=ast.Load()), ast.Name(id="point_cross", ctx=ast.Load())],
    ctx=ast.Load())
for tree in (dec, ipdn):
    ast.fix_missing_locations(tree)
    compile(tree, "<new-y44>", "exec")
target.mkdir()
for path in parent.glob("*.py"):
    shutil.copyfile(path, target / path.name)
for name, tree in (("dec.py", dec), ("ipdn.py", ipdn)):
    (target / name).write_text(ast.unparse(tree) + "\n", encoding="utf-8", newline="\n")
shutil.copyfile(config.parent / "adares_w2_pointmass_competition_res.yaml", config)
files = sorted(target.glob("*.py")) + [config]
assert len(files) == 7
result = {
    "complete": True, "id": "AdaRES-Y44", "final_only": True,
    "coordinate_axes": 3, "feature_channels": 32, "cross_inputs": 96,
    "files": [{"path": p.relative_to(ROOT).as_posix(),
               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
    "intervention": "Raw all3x32 centered native coordinate-feature cross moments, point-mass FG/BG, query final-indicator only.",
    "limits": ["Local independent implementation, not native admission or efficacy.",
               "No normalization, component selection, numeric sweep or diagnostic weight deployment."],
}
output = ROOT / ("pipeline/analyses/y44_local_bundle_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"path": str(output), "files": len(files)}))
