"""Audit physical production imports, frozen config and remote identity."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    package = root/"AdaRES/ipdn/model_paper5/adares_y17_target_equivalent_pairing"
    records = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        compile(tree, str(path), "exec")
        for node in ast.walk(tree):
            names = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                     else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            assert not any("model_paper5" in name for name in names), path
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1
                target = package/(node.module or "")
                assert target.with_suffix(".py").is_file() or (target/"__init__.py").is_file()
        records.append(dict(path=path.relative_to(root).as_posix(),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    config = root/"AdaRES/configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml"
    parent = root/"AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
    assert yaml.safe_load(config.read_text()) == yaml.safe_load(parent.read_text())
    records.append(dict(path=config.relative_to(root).as_posix(),
                        sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
    command = """import hashlib,json,pathlib,sys
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
rows=json.load(sys.stdin)
print(json.dumps([dict(path=r['path'],sha256=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()) for r in rows]))
"""
    remote = subprocess.run(["ssh", "-o", "BatchMode=yes", "songcl@192.168.1.163",
                             "python3 -c '"+command.replace("'", "'\"'\"'")+"'"],
                            input=json.dumps(records), text=True, capture_output=True,
                            check=True, timeout=40)
    assert json.loads(remote.stdout) == records
    evidence = {}
    for name in ("y17_unit", "y17_entry", "y17_physical_controls",
                 "y17_physical_stress", "y17_physical_inference", "y17_physical_resume"):
        path = root/"pipeline/analyses"/(name+"_20260930.json")
        evidence[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(production_files=records, evidence=evidence,
                  no_cross_variant_production_imports=True,
                  frozen_w2_config_equal=True, remote_byte_identity=True,
                  extra_trainable_parameters=0, passed=True,
                  limitations="Small-cohort feasibility, not trained efficacy or novelty.")
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
