"""Preserve failed programs; repair only probe bindings, never model gates."""

import argparse
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
records = []
for name in ("probe_y42_parent_semantics", "probe_y42_native_controls",
             "probe_y42_initial_loss_investigation"):
    source = ROOT / "scripts" / (name + ".py")
    text = source.read_text(encoding="utf-8")
    if name == "probe_y42_parent_semantics":
        anchor = "if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple):"
        assert text.count(anchor) == 1
        text = text.replace(anchor, "if method.name == 'extract_feat' and isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple):")
    else:
        assert "point_cocovariance" in text
        text = text.replace("point_cocovariance", "point_covariance")
    target = ROOT / "scripts" / (name + "_v2.py")
    assert not target.exists()
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8", newline="\n")
    records.append({
        "original": source.relative_to(ROOT).as_posix(),
        "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "repaired": target.relative_to(ROOT).as_posix(),
        "repaired_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    })
logs = []
for name in ("y42_parent_semantics", "y42_native_controls"):
    path = ROOT / ("pipeline/analyses/" + name + "_" + args.token + ".log")
    assert path.exists()
    logs.append({"path": path.relative_to(ROOT).as_posix(),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
output = ROOT / ("pipeline/analyses/y42_probe_binding_repair_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps({
    "complete": True, "id": "AdaRES-Y42", "original_programs_preserved": records,
    "failed_logs": logs, "model_source_or_protocol_changed": False,
    "reason": "Config name doubled by mechanical variance rename; AST adapter "
              "mistakenly asserted on unchanged expand_and_fps tuple return.",
    "scope": "Test-binding repair only. No failed gate relabeled or tolerance relaxed.",
}, indent=2), encoding="utf-8")
print(json.dumps({"repair_record": str(output), "programs": len(records)}))
