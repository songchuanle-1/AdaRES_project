"""Reuse current-tick frozen comparator captures with explicit provenance."""

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
source = ROOT / "scripts/probe_frozen_full_gpu1_392a.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
loop = next(n for n in tree.body if isinstance(n, ast.For)
            and isinstance(n.target, ast.Tuple) and n.target.elts[0].id == "identity")
reuse = ast.parse('''
if identity in ("w2","b1"):
    reuse_path=Path("/tmp/adares_y43_frozen_392a_"+P["token"]+".json")
    reuse=json.loads(reuse_path.read_text())
    assert reuse["complete"] and reuse["physical_gpu"]==1
    assert reuse["evaluator_sha256"]==expected_evaluator
    capture=reuse["rows"][identity]
    assert capture["config_sha256"]==config_sha
    assert capture["checkpoint_sha256"]==checkpoint_sha
    assert capture["complete"] and capture["completed_passes"]==2
    assert all(p["summary"]["expressions"]==9508 and p["summary"]["scenes"]==141 for p in capture["passes"])
    result["rows"][identity]=capture
    result["summaries"][identity]=capture["passes"][0]["summary"]
    result.setdefault("reused_comparators",{})[identity]=dict(
        path=str(reuse_path),sha256=hashlib.sha256(reuse_path.read_bytes()).hexdigest(),
        newly_executed_passes=0)
    result["preserved_after_"+identity]=preserved()
    Path(P["output"]).write_text(json.dumps(result),encoding="utf-8")
    continue
''').body
assert loop.body[0].targets[0].id == "config"
assert isinstance(loop.body[1], ast.Assert)
loop.body[2:2] = reuse
tree.body.insert(1, ast.parse('assert "y44_frozen" in P["output"]').body[0])
output = ROOT / "scripts/probe_y44_frozen_reuse_392a.py"
assert not output.exists()
output.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")
print(json.dumps(dict(path=output.relative_to(ROOT).as_posix(),
                      sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                      original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      fresh_candidate_passes=2, reused_reference_passes_per_method=2)))
