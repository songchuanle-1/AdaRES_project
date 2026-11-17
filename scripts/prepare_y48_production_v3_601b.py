"""Verify two known nonpersistent HF buffers without weakening state equality."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
token = sys.argv[sys.argv.index("--token") + 1]
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 480
source = ROOT / "scripts/probe_y48_production_601b_v2.py"
log = ROOT / ("pipeline/analyses/y48_production_v2_601b_" + token + ".log")
text = log.read_text(encoding="utf-8")
assert 'Unexpected key(s) in state_dict: "text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"' in text
tree = ast.parse(source.read_text())
helpers = ast.parse('''
def complete_diagnostic_state(model, reference):
    result = cpu_state(model)
    allowed = {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
    for key in reference.keys() - result.keys():
        assert key in allowed
        value = model
        for part in key.split("."):
            value = getattr(value, part)
        assert torch.equal(value.detach().cpu(), reference[key].cpu())
        result[key] = value.detach().cpu().clone()
    return result

def load_complete_diagnostic_state(model, state):
    loaded = model.load_state_dict(state, strict=False)
    assert not loaded.missing_keys
    assert not set(loaded.unexpected_keys) - {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
    assert differences(state, complete_diagnostic_state(model, state))["equal"]
''').body
counts = dict(load=0, snapshot=0)


class BufferCompatibility(ast.NodeTransformer):
    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict"
                and any(key.arg == "strict" and isinstance(key.value, ast.Constant)
                        and key.value.value is True for key in node.keywords)):
            assert isinstance(node.func.value, ast.Name) and node.func.value.id in ("restored", "parent")
            counts["load"] += 1
            return ast.copy_location(ast.Call(
                func=ast.Name(id="load_complete_diagnostic_state", ctx=ast.Load()),
                args=[node.func.value, node.args[0]], keywords=[]), node)
        expected = ast.parse("cpu_state(restored)", mode="eval").body
        if ast.dump(node) == ast.dump(expected):
            counts["snapshot"] += 1
            return ast.copy_location(ast.parse(
                "complete_diagnostic_state(restored, saved_state)", mode="eval").body, node)
        return self.generic_visit(node)


tree = BufferCompatibility().visit(tree)
assert counts == dict(load=2, snapshot=1)
position = next(i for i, node in enumerate(tree.body)
                if isinstance(node, ast.Assert))
tree.body[position:position] = helpers
destination = ROOT / "scripts/probe_y48_production_601b_v3.py"
code = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
compile(code, str(destination), "exec")
with destination.open("x", encoding="utf-8") as stream:
    stream.write(code)
manifest = dict(
    complete=True, id="AdaRES-Y48", preregistered_at=datetime.now().astimezone().isoformat(),
    failed_source=dict(path=source.relative_to(ROOT).as_posix(),
                       sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
    failed_log=dict(path=log.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(log.read_bytes()).hexdigest()),
    source=dict(path=destination.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),
    change="Only diagnostic restore compatibility for the exact two known HF metadata buffers; compare their actual tensor values and require full augmented model state, optimizer and scheduler equality.",
    prediction="No missing model keys or other unexpected keys; physical metadata buffers numerically equal saved values; all original controls/stress rerun.",
    model_changes=0, result="pending")
output = ROOT / ("pipeline/analyses/y48_production_buffer_repair_601b_" + token + ".json")
with output.open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
print(json.dumps(manifest, indent=2))
