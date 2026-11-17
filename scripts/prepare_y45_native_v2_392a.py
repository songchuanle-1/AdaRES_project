"""Cache actual native inputs/RNG for matched controls; retain full repeat failure."""

import argparse
import ast
from datetime import datetime, timezone
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
source = ROOT / "scripts/probe_y45_native_path_392a.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
extra = ast.parse('''
from probe_adares_y15_resume import rng,restore
def clone(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,(list,tuple)):return type(value)(clone(v) for v in value)
    if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
    return value
def before_decoder(module,inputs):
    capture["dec"]=clone(inputs)
    capture["dec_rng"]=rng()
''').body
tree.body[0:0] = extra
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "before_criterion")
function.body.extend(ast.parse('capture["criterion_args"]=clone(inputs[1:])').body)
hooks = next(n for n in tree.body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "hooks")
hooks.value.elts.append(ast.parse("model.dec.register_forward_pre_hook(before_decoder)", mode="eval").body)
outer = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "index")
delete = next(n for n in outer.body if isinstance(n, ast.Delete) and any(
    isinstance(t, ast.Name) and t.id == "target" for t in n.targets))
position = outer.body.index(delete)
outer.body[position:position] = ast.parse('''
point_positive=pred["point_positive"].detach().clone()
replay_args=clone(capture["criterion_args"])
decoder_inputs=clone(capture["dec"])
decoder_rng=capture["dec_rng"]
''').body
controls = next(n for n in outer.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "mode")
context = next(n for n in controls.body if isinstance(n, ast.With))
assert len(context.body) == 1 and context.body[0].value.func.id == "model"
context.body = ast.parse('''
restore(decoder_rng)
cached_pred=model.dec(*decoder_inputs)
cached_pred["point_positive"]=point_positive
loss,losses=model.criterion(cached_pred,*replay_args)
''').body
contexts = [n for n in tree.body if isinstance(n, ast.With)]
assert len(contexts) == 2
contexts[0].body = ast.parse('''
outputs=[]
def collect_inputs(module,inputs):
    capture["physical_args"]=clone(inputs)
hook=model.dec.register_forward_pre_hook(collect_inputs)
for batch in batches:
    gorilla.set_random_seed(1999)
    result=model(batch,mode="predict")
    inputs=clone(capture["physical_args"])
    disabled=model.dec(*inputs)
    outputs.append((inputs,{k:disabled[k].detach().cpu().clone() for k in ("masks","indis","sample_inds")}))
hook.remove()
''').body
contexts[1].body = ast.parse('''
for inputs,expected in outputs:
    result=parent.dec(*inputs)
    for key,value in expected.items():
        assert torch.equal(value,result[key].cpu()),key
''').body
result_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "result")
result_node.value.keywords.append(ast.keyword(
    arg="control_scope", value=ast.Constant("Actual native cached decoder inputs and pre-decoder RNG; full-input repeat failure retained. No whole-pipeline bitwise claim.")))
output = ROOT / "scripts/probe_y45_native_path_392a_v2.py"
assert not output.exists()
output.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")
print(str(output))
