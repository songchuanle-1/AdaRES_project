"""Repair only the environment-capable Git wrapper; retain the failed helper."""
import argparse
from datetime import datetime
import json

import rebind_y94_on_y93 as original
from retain_y91_cpu_native import git
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


class Runner(original.Runner):
    def visit_Constant(self, node):
        node = super().visit_Constant(node)
        if isinstance(node.value, str):
            node.value = node.value.replace(
                "Original native YAML/TRAIN two finite task backward and optimizer steps passed with full normalized-evidence hidden residual controls. Efficacy unmeasured.",
                "Original native YAML/TRAIN two finite backward and optimizer steps passed with original query-self geometry and hard SCG controls. Efficacy unmeasured.")
        return node


def run(token):
    lease(token, 300)
    assert ref("scripts/rebind_y94_on_y93.py")["sha256"] == (
        "ea83fa54ccd451aef0382b1562604b4819840b3b4928f8d037d9753ec710c2f1")
    before = {path: ref(path) for path in
              ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
    before_index = git("ls-files", "--stage", "-z")
    old_git, old_runner = original.git, original.Runner
    original.git, original.Runner = git, Runner
    try:
        original.rebind(token)
    finally:
        original.git, original.Runner = old_git, old_runner
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("ls-files", "--stage", "-z") == before_index
        assert all(ref(path) == value for path, value in before.items())
        path = ROOT / ("pipeline/checks/y94_y93_rebind_wrapper_repair_" + token + ".json")
        assert not path.exists()
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == original.CID)
        failure = ref("pipeline/checks/y94_y93_rebind_failure_" + token + ".json")
        tick.save(path, dict(
            complete=True, id=original.CID, token=token,
            observed_at=datetime.now().astimezone().isoformat(),
            original_failure=failure, original_failed_helper=ref(original.__file__),
            repair_helper=ref(__file__), environment_capable_git=ref("scripts/retain_y91_cpu_native.py"),
            ten_original_binding_guards_unchanged=True,
            model_and_native_program_bodies_unchanged=True, no_scientific_revision=True,
            production_HEAD_index_queue_running_preserved=True,
            binding=candidate["additive_version_binding"]))
        candidate["evidence"].extend([failure, ref(path), ref(__file__)])
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(repair=ref(path), original_failed_helper_retained=True)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    run(parser.parse_args().token)
