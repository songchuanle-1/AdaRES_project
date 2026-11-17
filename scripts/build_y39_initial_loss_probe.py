"""Specialize the fixed adverse-loss protocol without selecting settings."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
source = ROOT / "scripts/probe_y37_initial_loss_investigation.py"
text = source.read_text(encoding="utf-8")
text = text.replace("Y37", "Y39").replace("adares_y37_pair_distribution", "adares_y39_point_distribution")
text = text.replace("pair_readout", "point_readout").replace("wrong_pairs", "wrong_variance")
tree = ast.parse(text)
node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "shape_input")
old = ast.get_source_segment(text, node)
new = """def shape_input(module, args):
    q, masks, pad, mass, variance = args
    if mode == 'rolled':
        return q, masks.roll(1, 1), pad, mass, variance
    if mode == 'wrong_variance':
        changed = variance.clone()
        for b in range(variance.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = variance[b, :count].roll(1, 0)
        return q, masks, pad, mass, changed"""
assert text.count(old) == 1
text = text.replace(old, new)
text = "assert __import__('pathlib').Path('.codex_source_commit').read_text().strip()=='67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773'\n" + text
target = ROOT / "scripts/probe_y39_initial_loss_investigation.py"
assert not target.exists()
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
