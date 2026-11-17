"""Preserve the exact failed CPU guard and measured diagnosis without relaxing it."""
import argparse
from datetime import datetime, timedelta
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y97_language_quadratic_input import CID, files
import closure


def record(token):
    lease(token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == CID)
        assert candidate["stage"] == "implementation" and not candidate.get("code_commit")
        assert not candidate.get("CPU_result") and not candidate.get("native_preparation")
        assert not candidate.get("native_smoke_result") and not candidate.get("admission_receipt")
        program = ROOT / ("pipeline/checks/y97_cpu_" + token + "_program.py")
        failed_log = ROOT / ("pipeline/analyses/y97_cpu_" + token + ".log")
        diagnosis = ROOT / ("pipeline/analyses/y97_batch_permutation_diagnosis_" + token + ".json")
        value = tick.load(diagnosis)
        assert not failed_log.with_suffix(".json").exists()
        assert value["complete"] is True and value["id"] == CID
        assert value["original_CPU_program_sha256"] == ref(program)["sha256"]
        assert value["original_log_sha256"] == ref(failed_log)["sha256"]
        assert value["original_model_source_sha256"] == ref(files()[0].parent / "dec.py")["sha256"]
        assert value["original_float32_failure_reproduced"] is True
        assert value["original_guard_unchanged"] is True and value["model_or_YAML_modified"] is False
        assert value["full_model_or_data_replayed"] is False and value["CUDA_initialized"] is False
        assert len(value["rows"]) == 2 and all(row["original_bitwise_guard_failed"] for row in value["rows"])
        assert value["rows"][0]["max_absolute_difference"] == 1.1920928955078125e-7
        assert value["rows"][1]["max_absolute_difference"] == 2.220446049250313e-16
        assert all(row["within_existing_scalar_tolerance"] for row in value["rows"])
        refs = [ref(program), ref(failed_log), ref(diagnosis), ref(diagnosis.with_suffix(".log")),
            ref("pipeline/checks/y97_batch_permutation_diagnosis_" + token + "_program.py"),
            ref(__file__), ref("scripts/diagnose_y97_batch_permutation.py")] + [ref(path) for path in files()]
        output = ROOT / ("pipeline/checks/y97_CPU_numerical_failure_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, id=CID, token=token, observed_at=datetime.now().astimezone().isoformat(),
            failure_scope="implementation", failed_guard="torch.equal(layer(projected.flip(0), raw.flip(0)), value.flip(0))",
            original_CPU_attempt=dict(program=ref(program), raw_log=ref(failed_log),
                full_model_constructed_before_failure=True, released_backbone_loaded_before_failure=True,
                TRAIN_data_loading_started_before_failure=True,
                full_CPU_control_suite_passed=False, decoder_backward_executed=False),
            exact_component_reproduction=ref(diagnosis), measured_rows=value["rows"],
            existing_scalar_tolerance_observation_is_NOT_a_replacement_PASS=True,
            original_guard_unchanged=True, original_model_and_YAML_unchanged=True,
            no_repair_or_scientific_revision_executed=True, mechanism_falsified=False,
            native_executed=False, admitted=False, synchronized=False, training_launched=False,
            efficacy_measured=False, evidence=refs))
        candidate.update(
            stage="blocked", blocked_from_stage="implementation", blocker_kind="numerical_control_failure",
            failure_scope="implementation", CPU_failure=ref(output), CPU_program=ref(program),
            component_diagnosis=ref(diagnosis),
            blocker="Original CPU full-token quadratic-input component failed its exact batch-permutation "
                "bitwise guard before full decoder tests. Exact repeat measured float32 1epsilon/float64 "
                "1epsilon differences; original unchanged guard still FAIL, no CPU/native PASS or admission.",
            retry_after=(datetime.now().astimezone() + timedelta(hours=4)).isoformat(),
            unblock_condition="New numerical evidence or a pre-registered bounded SAME-ID implementation "
                "revision that passes the original unchanged bitwise/scalar/full decoder/native controls. "
                "No timer-based replay, guard relaxation, rejection of the mechanism, or renamed replacement.",
            next_action="Preserve original failed CPU program/log, model/YAML bytes and exact component "
                "reproduction. Aware numerical implementation hold; do not replay or replace equality with "
                "tolerance. Only preregister one evidence-backed execution change under same Y97 before retest, "
                "with full affected controls unchanged. Earlier Y94/Y95/Y96 and original Y92/Y93 continue.")
        candidate["evidence"].extend([ref(output)] + refs)
        work["tick_attempt"] = dict(
            tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
            candidate_id=CID, kind="implementation_test", result="failed",
            command="python -B scripts/run_experiment_first_probe.py y97_cpu pipeline/checks/y97_cpu_" + token
                + "_program.py --token " + token + " --seconds 480",
            observation="Actual CPU original YAML/model/backbone/TRAIN initialization reached the component "
                "batch-permutation guard and failed exact equality. Independent unchanged-guard reproduction "
                "measured float32 1.1920928955078125e-7 and float64 2.220446049250313e-16. "
                "No full CPU suite/native/admission/efficacy PASS; numerical implementation hold retained.",
            evidence=[ref(output), ref(failed_log), ref(diagnosis)])
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(failure=ref(output), diagnosis=ref(diagnosis), stage=candidate["stage"],
            failure_scope=candidate["failure_scope"], guards_relaxed=False, native_executed=False, admitted=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    record(parser.parse_args().token)
