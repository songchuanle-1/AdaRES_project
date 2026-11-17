"""Resolve the new cost adapter without rewriting a historical protocol."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
original = root / "scripts/probe_y31_trained_cost.py"
bundle = root / "pipeline/analyses/y33_source_bundle_lf_cde054a6b0254e37a0c0d71febbcc497.json"
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == "b08ee31f6a5ede73a2064f1ff2edfb52f5eb42ea5cf3cf5c868494c723b41d61"
path = root / "scripts/probe_y33_trained_cost.py"
source = path.read_text(encoding="utf-8")
assert source.count("ORIGINAL_COST_SOURCE") == source.count("Y33_FILES") == 1
source = source.replace("ORIGINAL_COST_SOURCE", repr(original.read_text(encoding="utf-8")))
source = source.replace("Y33_FILES", repr(json.loads(bundle.read_text())["files"]))
compile(source, str(path), "exec")
path.write_text(source, encoding="utf-8")
print(json.dumps(dict(complete=True, original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                     adapted_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     scope="IDs/deployment marker/source manifest/expected parameters only; timing/optimizer/data unchanged.")))
