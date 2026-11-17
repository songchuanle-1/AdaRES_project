"""Y18 scoped source identity and frozen-configuration audit."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidate = root / "AdaRES/ipdn/model_paper5/adares_y18_point_measure"
    parent = root / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
    records = []
    for path in sorted(candidate.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        compile(tree, str(path), "exec")
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.level:
                    target = path.parent.joinpath(*(node.module or "").split("."))
                    if not (target.with_suffix(".py").is_file()
                            or (target / "__init__.py").is_file()):
                        raise RuntimeError("Missing package-relative dependency")
                    names = ["." * node.level + n for n in names]
            else:
                continue
            if any("model_paper5" in name for name in names):
                raise RuntimeError("Cross-variant production import")
            imports.extend(names)
        records.append({"path": path.relative_to(root).as_posix(),
                        "sha256": sha(path), "imports": imports})
    unchanged = []
    for filename in ("backbone.py", "dec.py", "sample_model.py"):
        a = ast.parse((candidate / filename).read_text(encoding="utf-8"))
        b = ast.parse((parent / filename).read_text(encoding="utf-8"))
        if ast.dump(a) != ast.dump(b):
            raise RuntimeError("Unexpected production source change: " + filename)
        unchanged.append(filename)
    def methods(path):
        cls = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                   if isinstance(n, ast.ClassDef) and n.name == "IPDN")
        return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    old, new = methods(parent / "ipdn.py"), methods(candidate / "ipdn.py")
    if set(old) != set(new):
        raise RuntimeError("Unexpected model method additions")
    modified = [key for key in old if old[key] != new[key]]
    if modified != ["loss"]:
        raise RuntimeError("Model intervention exceeded loss path: " + str(modified))
    cfg_path = root / "AdaRES/configs_song/model_paper5/adares_y18_point_measure_res.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    parent_cfg = yaml.safe_load((root / (
        "AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
    )).read_text(encoding="utf-8"))
    if cfg != parent_cfg or cfg["train"]["epochs"] != 30 or cfg["train"]["seed"] != 1999:
        raise RuntimeError("Frozen parent configuration changed")
    records.append({"path": cfg_path.relative_to(root).as_posix(), "sha256": sha(cfg_path)})
    program = """
import hashlib,json,pathlib,sys
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
rows=json.load(sys.stdin)
print(json.dumps([dict(path=r['path'],sha256=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()) for r in rows]))
"""
    command = "/home/songcl/.conda/envs/adares_project/bin/python -c '" + program.replace("'", "'\"'\"'") + "'"
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                             "songcl@192.168.1.163", command],
                            input=json.dumps(records), capture_output=True,
                            text=True, check=True, timeout=60)
    remote = json.loads(result.stdout)
    if any(a["path"] != b["path"] or a["sha256"] != b["sha256"]
           for a, b in zip(records, remote)) or len(records) != len(remote):
        raise RuntimeError("Remote candidate byte mismatch")
    output = {"candidate": "AdaRES-Y18", "ast_compile_pass": True,
              "no_cross_variant_imports": True, "remote_byte_identity": True,
              "config_exact_parent_equivalence": True,
              "unchanged_model_methods_except_loss": True,
              "unchanged_modules": unchanged, "files": records,
              "parameters_added": 0,
              "scope": "Static source/config identity, not trained efficacy",
              "limitations": "Point label, mass and Dice smoothing effects coupled "
              "in intervention; confidence and matching remain native."}
    path = Path(args.output)
    if path.exists():
        raise RuntimeError("Refusing overwrite audit evidence")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "files"}))


if __name__ == "__main__":
    main()
