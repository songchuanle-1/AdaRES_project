"""Specialize existing current-owner retention/runtime tools without touching them."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


class Identity(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        node.module = node.module.replace("prepare_y95_query_set_mean", "prepare_y96_seed_query_skip")
        node.module = node.module.replace("y95", "y96")
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for old, new in (
                ("adares-y95-query-set-mean", "adares-y96-seed-query-skip"),
                ("all-query mean refinement bypass", "sampled-seed initial-query generator bypass"),
                ("all-query-mean bypass/original norm-FFN/shared head/hard-gate consumers",
                 "sampled-seed skip/original full generator/initial-coarse/shared-head/hard-gate consumers"),
                ("full all-query means, float64 projection references, same original norm and FFN consumers",
                 "original sampled SCG seeds, five original generator layers, float64 projections, initial and first coarse consumers"),
                ("six original self/language tensors", "original generated query and undetached sampled seed"),
                ("scalar/set-peer causality/seven-stage", "scalar/slot-local seed causality/seven-stage"),
                ("AdaRES-Y95", "AdaRES-Y96"),
                ("Y95", "Y96"),
                ("y95", "y96"),
                ("earlier Y94 has priority", "earlier Y94 and Y95 have priority"),
                ("after earlier Y94.", "after earlier Y94 and Y95."),
                ("preserve earlier Y94.", "preserve earlier Y94 and Y95."),
            ):
                node.value = node.value.replace(old, new)
        return node


def create(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y96")
        assert candidate["stage"] == "implementation" and not candidate.get("code_commit")
        originals, outputs = [], []
        for old, new in (("retain_y95_cpu_native.py", "retain_y96_cpu_native.py"),
                         ("test_y95_runtime_plan.py", "test_y96_runtime_plan.py")):
            source, target = ROOT / "scripts" / old, ROOT / "scripts" / new
            assert not target.exists()
            originals.append(ref(source))
            parsed = Identity().visit(ast.parse(source.read_text(encoding="utf-8")))
            body = ast.unparse(ast.fix_missing_locations(parsed)) + "\n"
            ast.parse(body, feature_version=(3, 8))
            target.write_text(body, encoding="utf-8", newline="\n")
            outputs.append(ref(target))
        assert all(ref(entry["path"]) == entry for entry in originals)
        output = ROOT / ("pipeline/checks/y96_retainer_support_" + token + ".json")
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
