"""Validate the published handoff files without importing ML libraries."""

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / "docs/handoff_20261117/publication_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures, syntax_failures = [], []
    for row in manifest["files"]:
        name = PurePosixPath(row["path"])
        if name.is_absolute() or ".." in name.parts:
            failures.append({"path": row["path"], "reason": "unsafe manifest path"})
            continue
        source = root.joinpath(*name.parts)
        if not source.is_file():
            failures.append({"path": row["path"], "reason": "missing"})
            continue
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            failures.append({"path": row["path"], "reason": "sha256 mismatch"})
        if source.suffix == ".py":
            try:
                ast.parse(data, filename=row["path"])
            except (SyntaxError, ValueError) as exc:
                syntax_failures.append({"path": row["path"], "reason": str(exc)})
    goal = root / "pipeline/goal.json"
    if hashlib.sha256(goal.read_bytes()).hexdigest() != manifest["scientific_goal_sha256"]:
        failures.append({"path": "pipeline/goal.json", "reason": "scientific goal changed"})
    print(json.dumps({
        "files_checked": len(manifest["files"]), "file_failures": failures,
        "syntax_failures": syntax_failures,
        "gpu_used": False, "ml_libraries_imported": False,
        "datasets_or_checkpoints_loaded": False,
        "limitation": "This is source verification, not native/runtime/scientific acceptance.",
    }, ensure_ascii=False, indent=2))
    return 1 if failures or syntax_failures else 0


if __name__ == "__main__":
    sys.exit(main())
