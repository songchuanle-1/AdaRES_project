"""Same-interpreter AST proof after stripping exactly the covariance path."""

import ast
import hashlib
from pathlib import Path
import yaml

base = Path("ipdn/model_paper5")
parent = base / "adares_w2_pointmass_competition"
candidate = base / "adares_y42_point_covariance"
counts = {}


def record(key):
    counts[key] = counts.get(key, 0) + 1


class StripNewCallArgument(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if node.args and isinstance(node.args[-1], ast.Name) and node.args[-1].id == "point_covariance":
            assert isinstance(node.func, ast.Attribute) and node.func.attr in ("res", "dec")
            node.args.pop()
            record("call_argument")
        return node


for name in ("dec.py", "ipdn.py"):
    original = ast.parse((parent / name).read_text(encoding="utf-8"))
    changed = ast.parse((candidate / name).read_text(encoding="utf-8"))
    changed.body = [
        node for node in changed.body
        if not (isinstance(node, ast.ClassDef) and node.name == "MaskPointCovarianceReadout")
    ]
    for cls in changed.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in ("DEC", "IPDN"):
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.args.args and method.args.args[-1].arg == "point_covariance":
                method.args.args.pop()
                assert isinstance(method.args.defaults[-1], ast.Constant)
                assert method.args.defaults[-1].value is None
                method.args.defaults.pop()
                record("signature")
            kept = []
            for statement in method.body:
                if isinstance(statement, ast.With) and any(
                    isinstance(n, ast.Attribute) and n.attr == "point_readout"
                    for n in ast.walk(statement)
                ):
                    record("constructor")
                    continue
                if isinstance(statement, ast.If) and any(
                    isinstance(n, ast.Name) and n.id == "point_covariance"
                    for n in ast.walk(statement.test)
                ):
                    record("required_input")
                    continue
                if isinstance(statement, ast.Assign):
                    targets = {n.id for target in statement.targets
                               for n in ast.walk(target) if isinstance(n, ast.Name)}
                    if any(
                        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "point_readout" for n in ast.walk(statement.value)
                    ):
                        record("final_addition")
                        continue
                    if "sp_feats" in targets and "point_covariance" in targets:
                        assert isinstance(statement.value, ast.Call)
                        assert isinstance(statement.value.func, ast.Attribute)
                        assert statement.value.func.attr == "extract_feat"
                        statement.targets = [ast.Name(id="sp_feats", ctx=ast.Store())]
                        record("extract_call")
                    elif "point_covariance" in targets:
                        record("covariance_assignment")
                        continue
                    elif name == "ipdn.py" and method.name == "extract_feat" and targets in (
                        {"mean_features"}, {"centered"}, {"joint_rows"}
                    ):
                        record("extract_statistic")
                        continue
                if method.name == 'extract_feat' and isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple):
                    assert [n.id for n in statement.value.elts] == ["x", "point_covariance"]
                    statement.value = ast.Name(id="x", ctx=ast.Load())
                    record("extract_return")
                kept.append(statement)
            method.body = kept
    changed = StripNewCallArgument().visit(changed)
    assert ast.dump(original) == ast.dump(changed), name + " changed outside new branch"
assert counts == {
    "constructor": 1, "signature": 2, "required_input": 1,
    "covariance_assignment": 4, "final_addition": 1, "call_argument": 3,
    "extract_call": 2, "extract_statistic": 3, "extract_return": 1,
}, counts
configs = Path("configs_song/model_paper5")
left = yaml.safe_load((configs / "adares_w2_pointmass_competition_res.yaml").read_text())
right = yaml.safe_load((configs / "adares_y42_point_covariance_res.yaml").read_text())
assert left == right
result = {
    "complete": True, "id": "AdaRES-Y42", "whole_modules_parent_identity": True,
    "removed_exactly": counts, "config_differences": [], "parser_version": sys.version,
    "source_hashes": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (parent, candidate) for path in folder.glob("*.py")
    },
    "limits": ["AST identity after removing the explicit new path, not original raw-byte equality.",
               "Same interpreter; independent files are still required for admission."],
}
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
