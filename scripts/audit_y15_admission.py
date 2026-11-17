"""Verify physical candidate ownership, frozen config and remote byte identity."""

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package_paths = [
        root / "AdaRES/ipdn/model_paper5/adares_y15_query_completion",
        root / "AdaRES/ipdn/model_paper5/adares_y15_query_completion_gt",
    ]
    paths = sorted({path for package in package_paths
                    for path in package.rglob("*.py")})
    configs = [
        root / "AdaRES/configs_song/model_paper5/adares_y15_query_completion_res.yaml",
        root / "AdaRES/configs_song/model_paper5/adares_y15_query_completion_gt_res.yaml",
    ]
    records = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        compile(tree, str(path), "exec")
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.level:
                    base = path.parent
                    for _ in range(node.level-1):
                        base = base.parent
                    target = base.joinpath(*(node.module or "").split("."))
                    if not (target.with_suffix(".py").is_file()
                            or (target / "__init__.py").is_file()):
                        raise ValueError("Unresolved relative import: " + str(path))
                    names = ["."*node.level + name for name in names]
            else:
                continue
            if any("model_paper5" in name for name in names):
                raise ValueError("Cross-variant production import: " + str(path))
            imports.extend(names)
        records.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "imports": imports,
        })
    for path in configs:
        records.append({"path": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    reference = yaml.safe_load((root / (
        "AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
    )).read_text(encoding="utf-8"))
    for path in configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["train"]["epochs"] == 30
        assert cfg["dataloader"]["train"]["batch_size"] == 2
        assert cfg["train"]["pretrain"].endswith("/sp_unet_backbone.pth")
        cfg["model"].pop("teacher_checkpoint", None)
        cfg["model"].pop("teacher_checkpoint_sha256", None)
        cfg["model"]["dec"].pop("completion_mode")
        cfg["model"]["dec"].pop("completion_enabled")
        cfg["model"]["criterion"].pop("completion_target")
        if cfg != reference:
            raise ValueError("Unregistered config difference: " + str(path))
    for path in root.joinpath("AdaRES/ipdn/model_paper5/"
                              "adares_y15_query_completion/teacher_reference").glob("*.py"):
        reference_path = root / "AdaRES/ipdn/model_paper5/adares_b4_query_competition" / path.name
        if path.read_bytes() != reference_path.read_bytes():
            raise ValueError("Teacher source changed relative to frozen B4: " + path.name)
    work = json.loads((root / "pipeline/research_work.json").read_text(encoding="utf-8"))
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y15")
    for evidence in candidate["evidence"]:
        actual = hashlib.sha256((root / evidence["path"]).read_bytes()).hexdigest()
        if actual != evidence["sha256"]:
            raise ValueError("Candidate evidence mismatch: " + evidence["path"])
    program = """
import hashlib,json,pathlib,sys
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
records=json.load(sys.stdin)
out=[]
for row in records:
    path=root/row['path']
    out.append({'path':row['path'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
print(json.dumps(out))
"""
    result = subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "songcl@192.168.1.163",
        "/home/songcl/.conda/envs/adares_project/bin/python -c '"
        + program.replace("'", "'\"'\"'") + "'",
    ], input=json.dumps(records), text=True, capture_output=True, timeout=60,
        check=True)
    remote = json.loads(result.stdout)
    assert len(remote) == len(records)
    assert all(a["path"] == b["path"] and a["sha256"] == b["sha256"]
               for a, b in zip(records, remote))
    output = {
        "at": datetime.now(timezone.utc).isoformat(),
        "candidate": "AdaRES-Y15", "files": records,
        "python_source_count": len(paths), "config_count": len(configs),
        "ast_compile_pass": True, "no_cross_variant_imports": True,
        "teacher_physical_b4_identity": True,
        "frozen_parent_config_except_registered_intervention": True,
        "candidate_evidence_hashes_verified": len(candidate["evidence"]),
        "remote_byte_identity": True,
        "limitations": "Shared dataset/native ops/utilities remain dependencies; "
        "this source audit is not a performance or novelty test.",
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "files"}))


if __name__ == "__main__":
    main()
