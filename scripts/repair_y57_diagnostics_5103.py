"""Repair test-driver API compatibility; preserve frozen programs and model bytes."""

import argparse
import ast
from datetime import datetime
import json

from research_5103 import ROOT, TOKEN, lease, ref, save, bind, tick, monitor


class Compatibility(ast.NodeTransformer):
    train_changes = 0
    scalar_changes = 0

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "zero_grad" and isinstance(call.func.value, ast.Call)
                and isinstance(call.func.value.func, ast.Attribute)
                and call.func.value.func.attr == "train"):
            inner = call.func.value
            self.train_changes += 1
            call.func.value = inner.func.value
            return [ast.copy_location(ast.Expr(value=inner), node), node]
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "numel"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "v"):
            self.scalar_changes += 1
            node.func.value = ast.Call(func=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()),
                                                        attr="as_tensor", ctx=ast.Load()),
                                       args=[node.func.value], keywords=[])
        return self.generic_visit(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    lease()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y57")
        bundle = tick.load(ROOT / candidate["tested_bundle"]["path"])
        assert all(ref(r["path"]) == r for r in bundle["files"])
        failure = ref("pipeline/analyses/y57_native_" + TOKEN + ".log")
        new = []
        for kind in ("native", "resources"):
            old = "pipeline/analyses/y57_" + kind + "_" + TOKEN + "_program.py"
            transform = Compatibility()
            tree = transform.visit(ast.parse((ROOT / old).read_text(encoding="utf-8")))
            assert transform.train_changes == 1
            assert transform.scalar_changes == (1 if kind == "resources" else 0)
            source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
            ast.parse(source, feature_version=(3, 8))
            path = old.replace("y57_" + kind + "_", "y57_" + kind + "_v2_")
            assert not (ROOT / path).exists()
            (ROOT / path).write_text(source, encoding="utf-8", newline="\n")
            plan = dict(complete=True, id="AdaRES-Y57", token=TOKEN,
                preregistered_at=datetime.now().astimezone().isoformat(),
                previous_program=ref(old), failure_log=failure, program=ref(path),
                model_bundle=candidate["tested_bundle"], model_unchanged=True,
                scope="Diagnostic driver only: project IPDN.train returns None, so separate train() "
                      "and zero_grad(). Resource output count accepts tensor/numpy via as_tensor; "
                      "scientific controls/backward/stress/timings unchanged.", result="pending")
            plan_ref = save(path.replace("_program.py", "_prereg.json"), plan)
            new.extend([ref(path), plan_ref])
        bind(candidate, failure, ref("pipeline/analyses/y57_cpu_" + TOKEN + ".json"), *new, ref(__file__))
        candidate.setdefault("diagnostic_revision_history", []).append(dict(
            scope="driver_train_return_and_output_scalar_compatibility",
            preregistered_at=datetime.now().astimezone().isoformat(), result="pending",
            failure_evidence=[failure], replacement=new))
        candidate["next_action"] = "Run same-model frozen Y57 native_v2 then resources_v2; " \
                                   "driver compatibility only, not a model revision or new scientific PASS."
        lease()
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(programs=new, model_unchanged=True), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
