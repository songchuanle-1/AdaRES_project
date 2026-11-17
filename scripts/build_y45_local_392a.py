"""Mechanically independent W2 package with a final decomposition path."""

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
target = parent.parent / "adares_y45_metric_decomposition"
config = ROOT / "AdaRES/configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml"
assert not target.exists() and not config.exists()


def statements(text):
    return ast.parse(text).body


def methods(tree, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return cls, {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


dec = ast.parse((parent / "dec.py").read_text(encoding="utf-8"))
cls, method = methods(dec, "DEC")
readout_tree = ast.parse((ROOT / "scripts/y45_decomposition_readout.py").read_text())
readout = next(n for n in readout_tree.body if isinstance(n, ast.ClassDef))
dec.body.insert(dec.body.index(cls), readout)
method["__init__"].body.extend(statements(
    "with torch.random.fork_rng(devices=[]):\n"
    "    self.metric_readout = MetricDecompositionReadout(d_model)\n"))
forward = method["forward_iter_pred"]
returns = [n for n in forward.body if isinstance(n, ast.Return)]
assert len(returns) == 1 and isinstance(returns[0].value, ast.Dict)
position = forward.body.index(returns[0])
forward.body[position:position] = statements(
    "error_logits = None\n"
    "if self.metric_readout.enabled:\n"
    "    error_logits = self.metric_readout(query, pred_masks, batch_mask, point_mass, inst_feats)\n"
    "    quality_odds = error_logits[..., 0] - torch.logsumexp(error_logits[..., 1:], dim=-1)\n"
    "    pred_indis = pred_indis + torch.stack((torch.zeros_like(quality_odds), quality_odds), dim=-1)\n")
returns[0].value.keys.extend([ast.Constant(value="error_logits"), ast.Constant(value="point_mass")])
returns[0].value.values.extend([ast.Name(id="error_logits", ctx=ast.Load()),
                               ast.Name(id="point_mass", ctx=ast.Load())])
ipdn = ast.parse((parent / "ipdn.py").read_text(encoding="utf-8"))
cls, method = methods(ipdn, "IPDN")
cls.body.extend(statements('''
def point_positives(self, superpoints, batch_offsets, scenes_len, gt_pmasks):
    result=[]
    expression=0
    with torch.no_grad():
        for scene in range(len(scenes_len)):
            left,right=int(batch_offsets[scene]),int(batch_offsets[scene+1])
            labels=superpoints[(superpoints>=left)&(superpoints<right)].long()-left
            for _ in range(int(scenes_len[scene])):
                truth=gt_pmasks[expression].reshape(-1).gt(.5)
                if len(truth)!=len(labels):
                    raise ValueError("Point truth must align with its original scene")
                result.append(torch.bincount(labels[truth],minlength=right-left).float())
                expression+=1
    if expression!=len(gt_pmasks):
        raise ValueError("Expression truth count mismatch")
    return torch.cat(result)
'''))
native_loss = method["loss"]
mass = next(n for n in native_loss.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "point_mass" for t in n.targets))
native_loss.body[native_loss.body.index(mass) + 1:native_loss.body.index(mass) + 1] = statements(
    "point_positive = self.point_positives(superpoints, batch_offsets, scenes_len, gt_pmasks)\n")
out = next(n for n in native_loss.body if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "out" for t in n.targets))
native_loss.body[native_loss.body.index(out) + 1:native_loss.body.index(out) + 1] = statements(
    "out['point_positive'] = self.dec.get_batches(point_positive.unsqueeze(-1), batch_offsets)[0].squeeze(-1)\n")
loss_tree = ast.parse((parent / "loss.py").read_text(encoding="utf-8"))
cls, method = methods(loss_tree, "Criterion")
cls.body.extend(statements('''
def loss_decomposition(self, pred):
    with torch.no_grad():
        masks=pred["masks"].detach().gt(0).float()
        valid=(~pred["batch_mask"]).float()
        positive=pred["point_positive"]*valid
        negative=(pred["point_mass"]-positive)*valid
        tp=torch.einsum("bqn,bn->bq",masks,positive)
        fp=torch.einsum("bqn,bn->bq",masks,negative)
        fn=positive.sum(-1,keepdim=True)-tp
        targets=torch.stack((tp,fp,fn),dim=-1)
        targets=targets/targets.sum(-1,keepdim=True).clamp_min(1)
    return -(targets*pred["error_logits"].log_softmax(-1)).sum(-1).mean()
'''))
forward = method["forward"]
loss_record = next(n for n in forward.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Subscript)
                   and isinstance(n.targets[0].slice, ast.Constant)
                   and n.targets[0].slice.value == "loss")
position = forward.body.index(loss_record)
forward.body[position:position] = statements(
    "if pred.get('error_logits') is not None:\n"
    "    decomposition_loss = self.loss_decomposition(pred)\n"
    "    loss = loss + decomposition_loss\n"
    "    loss_out['decomposition_loss'] = decomposition_loss\n")
for tree in (dec, ipdn, loss_tree):
    ast.fix_missing_locations(tree)
    compile(tree, "<independent-y45>", "exec")
target.mkdir()
for path in parent.glob("*.py"):
    shutil.copyfile(path, target / path.name)
for name, tree in (("dec.py", dec), ("ipdn.py", ipdn), ("loss.py", loss_tree)):
    (target / name).write_text(ast.unparse(tree) + "\n", encoding="utf-8", newline="\n")
shutil.copyfile(config.parent / "adares_w2_pointmass_competition_res.yaml", config)
files = sorted(target.glob("*.py")) + [config]
assert len(files) == 7
manifest = dict(
    complete=True, id="AdaRES-Y45", final_only=True,
    files=[dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
    intervention="Query/FG/BG point-mass readout predicts TP/FP/FN fractions; final positive odds get metric odds residual; TRAIN-only categorical likelihood targets.",
    limitations=["Independent local implementation, not admission or efficacy.",
                 "Equal unit coefficient is joint categorical likelihood, not a searched loss weight.",
                 "Multiplicative matching/quality odds assumes complementary evidence; usefulness remains falsifiable."])
output = ROOT / ("pipeline/analyses/y45_local_bundle_392a_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(dict(path=str(output), files=len(files))))
