"""Retain generated v1 files and correct only pre-execution metadata in v2."""
import argparse
import ast
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


class Metadata(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("run_y93_bound_native.py", "run_y93_bound_native_v2.py")
            value = value.replace("rebind_y93_on_y92.py", "rebind_y93_on_y92_v2.py")
            value = value.replace("original_pre_y93_binding", "original_pre_y92_binding")
            value = value.replace("files on Y91.", "files on Y92.")
            value = value.replace("Run identical Y92 native inputs", "Run identical Y93 native inputs")
            node.value = value
        return node

    def visit_FunctionDef(self, node):
        if node.name == "test_current_Y90_Y91_sources_preserved":
            node.name = "test_current_Y91_Y92_sources_preserved"
        return self.generic_visit(node)


def prepare(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/research_work.json")}
        originals, generated = [], []
        for source, target in (
            ("run_y93_bound_native.py", "run_y93_bound_native_v2.py"),
            ("rebind_y93_on_y92.py", "rebind_y93_on_y92_v2.py"),
        ):
            old, new = ROOT / "scripts" / source, ROOT / "scripts" / target
            assert not new.exists()
            originals.append(ref(old))
            tree = Metadata().visit(ast.parse(old.read_text(encoding="utf-8")))
            ast.fix_missing_locations(tree)
            body = ast.unparse(tree) + "\n"
            ast.parse(body, feature_version=(3, 8))
            new.write_text(body, encoding="utf-8", newline="\n")
            generated.append(ref(new))
        assert all(ref(path) == entry for path, entry in before.items())
        assert [ref(ROOT / entry["path"]) for entry in originals] == originals
        output = ROOT / ("pipeline/checks/y93_binding_preexecution_metadata_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, recorded_at=datetime.now().astimezone().isoformat(),
            diagnosis="Inspection found stale parent prose, a colliding historical metadata key and old test/doc names.",
            originals_preserved=originals, corrected_v2=generated, helper=ref(__file__),
            original_helpers_executed=False, original_guards_executed=False,
            scientific_files_changed=False, native_bodies_changed=False,
            original_guard_logic_changed=False, scientific_revision=False))
        print(json.dumps(dict(metadata_repair=ref(output), generated=generated,
                              original_helpers_executed=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    prepare(parser.parse_args().token)
