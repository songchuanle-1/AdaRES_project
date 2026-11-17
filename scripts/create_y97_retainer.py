"""Specialize current-parent retention and runtime tests; leave signed sources intact."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


class Identity(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        node.module = node.module.replace("prepare_y96_seed_query_skip", "prepare_y97_language_quadratic_input")
        node.module = node.module.replace("y96", "y97")
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for old, new in (
                ("adares-y96-seed-query-skip", "adares-y97-language-quadratic-input"),
                ("sampled-seed initial-query generator bypass", "full-token quadratic input before language norm"),
                ("sampled-seed skip/original full generator/initial-coarse/shared-head/hard-gate consumers",
                 "full-token quadratic/raw-projection/language-norm-sampling-self/shared-head/hard-gate consumers"),
                ("original generated query and undetached sampled seed, original sampled SCG seeds, five original generator layers, float64 projections, initial and first coarse consumers",
                 "same raw768 tokens and original projection, float64 quadratic references, same original language norm, sampling and first language-self consumers"),
                ("scalar/slot-local seed causality", "scalar/token-local quadratic causality"),
                ("AdaRES-Y96", "AdaRES-Y97"),
                ("Y96", "Y97"),
                ("y96", "y97"),
                ("earlier Y94 and Y95 have priority", "earlier Y94, Y95 and Y96 have priority"),
                ("earlier Y94 and Y95.", "earlier Y94, Y95 and Y96."),
            ):
                node.value = node.value.replace(old, new)
        return node


def create(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y97")
        assert candidate["stage"] == "implementation" and not candidate.get("code_commit")
        originals, outputs = [], []
        for old, new in (("retain_y96_cpu_native.py", "retain_y97_cpu_native.py"),
                         ("test_y96_runtime_plan.py", "test_y97_runtime_plan.py")):
            source, target = ROOT / "scripts" / old, ROOT / "scripts" / new
            assert not target.exists()
            originals.append(ref(source))
            parsed = Identity().visit(ast.parse(source.read_text(encoding="utf-8")))
            body = ast.unparse(ast.fix_missing_locations(parsed)) + "\n"
            ast.parse(body, feature_version=(3, 8))
            target.write_text(body, encoding="utf-8", newline="\n")
            outputs.append(ref(target))
        assert all(ref(entry["path"]) == entry for entry in originals)
        output = ROOT / ("pipeline/checks/y97_retainer_support_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, id=candidate["id"],
            source_specialization_only=True, originals_unchanged=True, original_templates=originals,
            new_helpers=outputs, native_executed=False, admitted=False, production_sync=False,
            evidence=originals + outputs + [ref(__file__)]))
        candidate["evidence"].extend([ref(output)] + originals + outputs + [ref(__file__)])
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(support=ref(output), helpers=outputs)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
