"""Bind the unchanged builder to actual native parent bytes after AST audit."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
token = sys.argv[sys.argv.index("--token") + 1]
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
capture = ROOT / ("pipeline/analyses/w2_native_source_601b_" + token)
local = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
assert len(list(capture.glob("*.py"))) == 6
identity = {}
for source in local.glob("*.py"):
    native = capture / source.name
    assert ast.dump(ast.parse(source.read_text(encoding="utf-8"))) == ast.dump(
        ast.parse(native.read_text(encoding="utf-8")))
    identity[source.name] = dict(
        local_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        AST_equal=True)
assert identity["loss.py"]["native_sha256"] == (
    "836ff790a0a8b9c55cf465a472c9462699f0753a7e87d0ef74a8946edf31565a")
native_config = capture / "config.yaml"
local_config = ROOT / "AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
assert yaml.safe_load(native_config.read_text()) == yaml.safe_load(local_config.read_text())
source = ROOT / "scripts/build_y48_local_601b.py"
failure = dict(
    complete=True, id="AdaRES-Y48",
    observed_at_local=datetime.now().astimezone().isoformat(),
    failed_source=dict(path=source.relative_to(ROOT).as_posix(),
                       sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
    observed_failure="AssertionError at native byte-hash precondition; no candidate package/config written.",
    parent_sources=identity, configuration_semantically_equal=True,
    repair="Use captured actual native parent bytes; same target helper and intervention; no scientific threshold/gate change.")
manifest = ROOT / ("pipeline/analyses/y48_build_byte_identity_repair_601b_" + token + ".json")
with manifest.open("x", encoding="utf-8") as stream:
    json.dump(failure, stream, indent=2)
replacements = dict(parent=0, target=0, config=0, manifest=0)


class BindNativeParent(ast.NodeTransformer):
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "parent":
                node.value = ast.parse(repr(str(capture)), mode="eval").body
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                      args=[node.value], keywords=[])
                replacements["parent"] += 1
            elif name == "target":
                node.value = ast.parse(
                    "ROOT / 'AdaRES/ipdn/model_paper5/adares_y48_mask_equivalence'",
                    mode="eval").body
                replacements["target"] += 1
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "shutil" and node.func.attr == "copyfile"
                and len(node.args) == 2
                and ast.dump(node.args[0]) == ast.dump(ast.parse(
                    "config.parent / 'adares_w2_pointmass_competition_res.yaml'", mode="eval").body)):
            node.args[0] = ast.parse("parent / 'config.yaml'", mode="eval").body
            replacements["config"] += 1
        return self.generic_visit(node)

    def visit_DictComp(self, node):
        expected = ast.parse("path.relative_to(ROOT / 'AdaRES').as_posix()", mode="eval").body
        if ast.dump(node.key) == ast.dump(expected):
            node.key = ast.parse(
                "'ipdn/model_paper5/adares_w2_pointmass_competition/' + path.name",
                mode="eval").body
            replacements["manifest"] += 1
        return self.generic_visit(node)


tree = ast.fix_missing_locations(BindNativeParent().visit(ast.parse(source.read_text())))
assert replacements == dict(parent=1, target=1, config=1, manifest=1)
exec(compile(tree, str(source), "exec"), {"__name__": "__main__", "__file__": str(source)})
