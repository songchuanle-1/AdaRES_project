"""Correct cross-tick command provenance without rewriting original receipts."""
import argparse
import copy
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from advance_y96 import prepared, validate_result
import closure
import experiment_first

NATIVE_TOKEN = "411304d6e92840b8844bb9ed860554fb"


def corrected_smoke(old, actual_command, activation, rawref):
    result = copy.deepcopy(old)
    result.update(command=actual_command, native_execution_token=NATIVE_TOKEN,
        provenance_note="Actual native execution was in the earlier tick; this admission reuses the "
                        "unchanged result and performs no new native execution.",
        original_admission_command_label=old["command"])
    result["evidence"] = old["evidence"] + [activation, rawref]
    assert {key: value for key, value in old.items() if key not in ("command", "evidence")} == {
        key: result[key] for key in old if key not in ("command", "evidence")}
    assert old["command"] != result["command"]
    return result


def correct(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate, _, _, frozen = prepared(work)
        assert candidate["stage"] == "training"
        queue = monitor.load_queue(str(ROOT))
        before_queue = copy.deepcopy(queue)
        running = monitor.load_running(str(ROOT))
        task = next(row for row in queue["gpu_queues"]["server-1"] if row["id"] == "AdaRES-Y96")
        assert task["status"] == "running" and task["pid"] == 3893460 and task["launch_verified"]
        original_ref = ref(task["admission_receipt"])
        original = tick.load(closure.artifact(ROOT, original_ref))
        assert task["admission_receipt"] == candidate["admission_receipt"] == "pipeline/admissions/AdaRES-Y96.json"
        original_smoke_ref = original["native_smoke"]
        original_smoke = tick.load(closure.artifact(ROOT, original_smoke_ref))
        rawref = candidate["native_smoke_result"]
        assert rawref["path"] == "pipeline/analyses/y96_server1_smoke_" + NATIVE_TOKEN + ".json"
        validate_result(tick.load(closure.artifact(ROOT, rawref)), candidate, frozen)
        activation_ref = ref("pipeline/checks/y96_server1_smoke_" + NATIVE_TOKEN + "_activation.json")
        activation = tick.load(closure.artifact(ROOT, activation_ref))
        assert activation["token"] == NATIVE_TOKEN and activation["server"] == "server-1"
        assert activation["native_plan"] == candidate["native_smoke_plan"]
        command = "python -B scripts/run_y96_bound_native_y95.py --server server-1 --token " + NATIVE_TOKEN
        smoke = corrected_smoke(original_smoke, command, activation_ref, rawref)
        assert original_smoke["command"].endswith(token) and smoke["command"].endswith(NATIVE_TOKEN)
        validate_result(smoke, candidate, frozen)
        for key in ("losses", "gradient_checks", "initialization_paths", "data_source", "batches"):
            assert smoke[key] == original_smoke[key]
        base = "pipeline/checks/y96_native_provenance_" + token
        smoke_path = ROOT / (base + "_smoke.json")
        note_path = ROOT / (base + ".json")
        receipt_path = ROOT / ("pipeline/admissions/AdaRES-Y96_provenance_" + token + ".json")
        assert not any(path.exists() for path in (smoke_path, note_path, receipt_path))
        tick.save(smoke_path, smoke)
        receipt = copy.deepcopy(original)
        receipt.update(native_smoke=ref(smoke_path), supersedes_provenance=original_ref,
            provenance_corrected_at=datetime.now().astimezone().isoformat())
        for gate in receipt["gates"].values():
            gate["evidence"] = [ref(smoke_path) if entry == original_smoke_ref else entry
                                for entry in gate["evidence"]]
        experiment_first.validate_smoke(ROOT, receipt)
        tick.save(receipt_path, receipt)
        task["admission_receipt"] = ref(receipt_path)["path"]
        closure.admission(ROOT, task)
        expected = copy.deepcopy(before_queue)
        next(row for row in expected["gpu_queues"]["server-1"]
             if row["id"] == "AdaRES-Y96")["admission_receipt"] = task["admission_receipt"]
        assert queue == expected and monitor.load_running(str(ROOT)) == running
        tick.save(note_path, dict(complete=True, token=token, id="AdaRES-Y96",
            original_receipt=original_ref, original_smoke=original_smoke_ref,
            actual_native_execution_token=NATIVE_TOKEN, actual_native_result=rawref,
            activation=activation_ref, corrected_smoke=ref(smoke_path), corrected_receipt=ref(receipt_path),
            scientific_values_unchanged=True, original_files_unchanged=True,
            native_reexecuted=False, runtime_mutations=False, PID_unchanged=3893460,
            helper=ref(__file__)))
        candidate.update(admission_receipt=task["admission_receipt"], native_smoke=ref(smoke_path),
                         provenance_correction=ref(note_path))
        candidate["evidence"].extend([ref(note_path), ref(smoke_path), ref(receipt_path), ref(__file__)])
        if work.get("tick_attempt", {}).get("candidate_id") == candidate["id"]:
            work["tick_attempt"].update(kind="implementation_test", result="passed",
                command="python -B scripts/correct_y96_reused_native_provenance.py --token " + token,
                observation="Actual earlier native result reused; command provenance corrected additively. "
                            "Original receipts, scientific fields, code and live PID preserved; no native replay.",
                evidence=[ref(note_path), ref(smoke_path), ref(receipt_path)])
        assert ref(original_ref["path"]) == original_ref
        assert ref(original_smoke_ref["path"]) == original_smoke_ref
        monitor.save_queue(str(ROOT), queue)
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(correction=ref(note_path), receipt=ref(receipt_path),
                              native_reexecuted=False, original_receipts_preserved=True, pid=3893460)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    correct(parser.parse_args().token)
