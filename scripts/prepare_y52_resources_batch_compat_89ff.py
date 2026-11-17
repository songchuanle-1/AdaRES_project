"""Freeze deployment-native inference batching; retain the original train stress."""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
from prepare_y52_native_backward_89ff import ref, TOKEN
import foreground_tick as tick
import pipeline_monitor as monitor


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
        assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
        assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 400
        for name, key in ((".tick_context.json", "context_sha256"), (".tick_context_state.next.json", "pending_sha256")):
            assert ref("pipeline/headless/" + name)["sha256"] == owner[key]
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y52")
        assert candidate["stage"] == "validation"
        for row in candidate["evidence"]:
            assert ref(row["path"]) == row
        original = "pipeline/analyses/y52_native_resources_program_" + TOKEN + ".py"
        failure = "pipeline/analyses/y52_native_resources_89ff_" + TOKEN + ".log"
        failure_text = (ROOT / failure).read_text(encoding="utf-8")
        assert "RuntimeError" in failure_text and "710" in failure_text and "667" in failure_text
        source = (ROOT / original).read_text(encoding="utf-8")
        replacements = [
            ("largest = max(range(4), key=lambda i: point_counts[i])",
             "largest = max(range(4), key=lambda i: point_counts[i])\n"
             "predict_batches = [dataset.collate_fn([copy.deepcopy(item)]) for item in items]\n"
             "predict_point_counts = [int(batch['coords_float'].shape[0]) for batch in predict_batches]"),
            ("for index, batch in enumerate(batches):", "for index, batch in enumerate(predict_batches):"),
            ("points=point_counts[index], two_warmups=True,",
             "points=predict_point_counts[index], two_warmups=True,"),
            ("for index in range(4):", "for index in range(8):"),
        ]
        for before, after in replacements:
            assert source.count(before) == 1, before
            source = source.replace(before, after)
        ast.parse(source, feature_version=(3, 8))
        program = "pipeline/analyses/y52_resources_batch_compat_program_" + TOKEN + ".py"
        plan_path = "pipeline/analyses/y52_resources_batch_compat_prereg_" + TOKEN + ".json"
        assert not (ROOT / program).exists() and not (ROOT / plan_path).exists()
        plan = dict(
            complete=True, id="AdaRES-Y52", token=TOKEN,
            preregistered_at=datetime.now().astimezone().isoformat(),
            original_program=ref(original), failure_evidence=ref(failure),
            scope="inference_batch_compatibility_not_model",
            change="Only predict uses eight single-scene batches, as native deployment requires. The SAME eight TRAIN scenes/32 expressions, exact initial states, two warmups/three measurements, all parity checks, and four original batchsize2 train pairs/largest-pair stress remain unchanged.",
            prediction="Parent native predict accepts supported batchsize1; parent/candidate outputs match within independently observed repeats. Native train batchsize2 remains finite and below the unchanged24GiB reserved-memory limit.",
            limitations="Original parent batchsize2 predict failure is preserved. No sample/model/threshold change, trained efficacy, global worst-case resource claim, model revision, commit, admission or training.",
            result="pending",
        )
        (ROOT / program).write_text(source, encoding="utf-8", newline="\n")
        plan["program"] = ref(program)
        tick.save(ROOT / plan_path, plan)
        for path in (failure, "scripts/prepare_y52_resources_batch_compat_89ff.py", program, plan_path):
            if path not in {e["path"] for e in candidate["evidence"]}:
                candidate["evidence"].append(ref(path))
        candidate.setdefault("diagnostic_revision_history", []).append(dict(
            scope=plan["scope"], preregistered_at=plan["preregistered_at"],
            change=plan["change"], prediction=plan["prediction"],
            failure_evidence=[ref(failure)], evidence=[ref(plan_path)], result="pending"))
        candidate["next_action"] = "Execute frozen native batchsize1 paired predict and unchanged batchsize2 largest-pair train stress. Preserve parent unsupported-batch failure. No model revision or automatic commit/admission."
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(program=ref(program), plan=ref(plan_path)), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
