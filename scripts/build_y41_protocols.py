"""Mechanically specialize immutable native protocols, with explicit identities."""

from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "5e172bd47ee3450dbee928bdecaede7f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
specs = [
    ("probe_y37_native_controls_v3.py", "probe_y41_native_controls.py"),
    ("probe_y37_physical_native.py", "probe_y41_physical_native.py"),
    ("probe_y37_resources.py", "probe_y41_resources.py"),
]
for source_name, target_name in specs:
    source = ROOT / "scripts" / source_name
    text = source.read_text(encoding="utf-8")
    text = text.replace("Y37", "Y41").replace(
        "adares_y37_pair_distribution", "adares_y41_boundary_probability"
    ).replace("pair_readout", "boundary_readout").replace("pair_gradients", "boundary_gradients")
    if target_name != "probe_y41_resources.py":
        tree = ast.parse(text)
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                for part in ast.walk(node.test):
                    if isinstance(part, ast.Constant) and part.value == 8:
                        part.value = 4
                        count += 1
            if isinstance(node, ast.keyword) and node.arg == "added_trainable_tensors":
                assert node.value.value == 8
                node.value.value = 4
        assert count >= 2
        text = ast.unparse(tree) + "\n"
    target = ROOT / "scripts" / target_name
    assert not target.exists()
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"path": target_name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "protocol_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))

source = ROOT / "scripts/probe_y36_trained_cost_fixed.py"
original = source.read_text(encoding="utf-8")
imports = original[:original.index("root = Path(P['root'])")]
body = original[original.index("cfg = gorilla.Config.fromfile"):]
old_assert = "assert parameters == {'candidate': 165089671, 'parent': 164629637}"
assert body.count(old_assert) == 1
body = body.replace(old_assert, "assert parameters == {name: capture['parameters'] for name, capture in captures.items()}")
body = body.replace("AdaRES-Y36", "AdaRES-Y37")
prefix = """
root = Path(P['root'])
cmd = Path('/proc/102267/cmdline').read_bytes().split(b'\\0')
assert b'configs_song/model_paper5/adares_y39_point_distribution_res.yaml' in cmd
assert cmd[cmd.index(b'--gpu_ids')+1] == b'1'
combined = json.loads(Path('/tmp/adares_y37_frozen_full_retry_' + P['token'] + '.json').read_text())
captures = {'candidate': combined['rows']['y37'], 'parent': combined['rows']['w2']}
assert all(c['complete'] and c['completed_passes'] == 2 for c in captures.values())
source_manifest = [(p, h) for p, h in json.loads(Path('/tmp/adares_y37_terminal_current_' + P['token'] + '.json').read_text())['source_hashes'].items()]
assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p,h in source_manifest)
"""
text = imports + prefix + body
text += """
assert Path('/proc/102267/cmdline').read_bytes().split(b'\\0') == cmd
"""
target = ROOT / "scripts/probe_y37_trained_cost_current.py"
assert not target.exists()
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps({"path": target.name, "protocol_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))

source = ROOT / "scripts/probe_y41_boundary_cpu.py"
readout = ROOT / "scripts/y41_boundary_readout.py"
text = ("from pathlib import Path\n"
        "P['readout_path']='/tmp/adares_y41_readout_' + P['token'] + '.py'\n"
        "assert not Path(P['readout_path']).exists()\n"
        "Path(P['readout_path']).write_text(" + repr(readout.read_text()) +
        ", encoding='utf-8')\n" + source.read_text())
target = ROOT / "scripts/probe_y41_boundary_cpu_embedded.py"
assert not target.exists()
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps({"path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
