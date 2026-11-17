"""Preregister one same-ID execution revision without rewriting failed evidence."""
import argparse
import ast
import base64
import copy
from datetime import datetime, timedelta
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y97_language_quadratic_input import CID, PACKAGE as ORIGINAL_PACKAGE, files as original_files
from y97_contract import validate_cpu
import closure


PACKAGE = ORIGINAL_PACKAGE + "_rev1"
OLD = "        return projected + raw_tokens.square() @ self.weight"
NEW = """        tokens = raw_tokens.reshape(-1, raw_tokens.shape[-1])
        quadratic = torch.stack(
            [torch.mv(self.weight.t(), token.square()) for token in tokens.unbind(0)],
            dim=0,
        ).reshape_as(projected)
        return projected + quadratic"""


def files():
    return [path.parent.parent / PACKAGE / path.name for path in original_files()[:-1]] + [
        ROOT / "AdaRES/configs_song/model_paper5" / (PACKAGE + "_res.yaml")]


def candidate(work):
    return next(row for row in work["candidates"] if row["id"] == CID)


def register(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    row = candidate(work)
    assert row["stage"] == "blocked" and row["blocker_kind"] == "numerical_control_failure"
    assert row["failure_scope"] == "implementation" and not row.get("revision_history")
    assert not any(row.get(key) for key in ("CPU_result", "code_commit", "native_smoke_result", "admission_receipt"))
    for entry in row["evidence"]:
        closure.artifact(ROOT, entry)
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert view["experiment_first"]["enabled"]
    assert not any(path.exists() for path in files())
    previous = copy.deepcopy(row)
    path = ROOT / ("pipeline/checks/y97_revision1_registration_" + token + ".json")
    assert not path.exists()
    registered = datetime.now().astimezone().isoformat()
    revision = dict(
        revision=1, failure_scope="implementation",
        reason="The original collapsed-row GEMM gives one-epsilon differences after batch permutation. "
            "Working diagnosis: output-row placement changes the floating-point reduction kernel; "
            "this is not evidence against the quadratic-language hypothesis.",
        change="Evaluate the SAME raw-token square projection using one fixed-shape torch.mv per token, "
            "then stack and add the original projected tensor. Keep every parameter, initialization, "
            "full token/channel, active norm, consumer and protocol unchanged. Use a separate rev1 "
            "package under the SAME AdaRES-Y97 ID so original model/YAML paths remain immutable.",
        prediction="The original exact component batch-permutation guard should pass while scalar, "
            "sign, token-locality, gradient, RNG, seven-stage W2, bypass and two-update guards remain "
            "unchanged. The full observer STILL demands exact equality with its original batched-matmul "
            "expression; a mismatch there is a failed revision, not permission to replace the guard.",
        preregistered_at=registered,
        failure_evidence=[row["CPU_failure"], row["component_diagnosis"]],
        result="pending", evidence=[], result_evidence=[])
    tick.save(path, dict(
        complete=True, token=token, id=CID, registered_at=registered,
        original_candidate=previous, revision=revision,
        original_CPU_program=row["CPU_program"],
        original_native_body=row["prepared_native_body"],
        original_files=[ref(path) for path in original_files()],
        original_component_controls=ref("scripts/probe_y97_cpu_controls.py"),
        original_observer_core=ref("scripts/y97_observer_core.py"),
        replacement_old=OLD, replacement_new=NEW,
        all_original_guards_unchanged=True, guard_relaxation_authorized=False,
        validation_used=False, original_training_untouched=True,
        native_GPU_authorized_by_this_registration=False, helper=ref(__file__)))
    revision["evidence"] = [ref(path)]
    row["revision_history"] = [revision]
    row["revision1_registration"] = ref(path)
    row["stage"] = "implementation"
    row["next_action"] = "Prepare the preregistered rev1 package and run the unchanged full CPU program once. "
    row["next_action"] += "No original evidence, guard, hypothesis, resource pause or ongoing run may change."
    row["evidence"].extend([ref(path), ref(__file__)])
    closure.validate_revisions(ROOT, row, previous)
    tick.save(ROOT / "pipeline/research_work.json", work)
    print(json.dumps(dict(registration=ref(path), revision=1, tested=False)))


def prepare(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    row = candidate(work)
    assert row["stage"] == "implementation"
    revision = row["revision_history"][-1]
    assert revision["revision"] == 1 and revision["result"] == "pending"
    registration = tick.load(closure.artifact(ROOT, row["revision1_registration"]))
    assert registration["token"] == token and registration["replacement_new"] == NEW
    assert registration["replacement_old"] == OLD and registration["all_original_guards_unchanged"]
    for entry in registration["original_files"]:
        closure.artifact(ROOT, entry)
    for key in ("original_component_controls", "original_observer_core", "original_native_body"):
        closure.artifact(ROOT, registration[key])
    original_program = closure.artifact(ROOT, registration["original_CPU_program"])
    original_text = original_program.read_text(encoding="utf-8")
    payload = ast.literal_eval(ast.parse(original_text).body[0].value)
    source = base64.b64decode(payload["variant"]["python_files"]["dec.py"]["base64"]).decode("utf-8")
    assert source.count(OLD) == 1
    source = source.replace(OLD, NEW)
    ast.parse(source, feature_version=(3, 8))
    paths = files()
    assert not paths[0].parent.exists() and not any(path.exists() for path in paths)
    paths[0].parent.mkdir()
    for old_path, new_path in zip(original_files(), paths):
        new_path.write_bytes(source.encode("utf-8") if old_path.name == "dec.py" else old_path.read_bytes())
    assert list((ROOT / "AdaRES/configs_song").rglob(paths[-1].name)) == [paths[-1]]

    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)["sha256"])

    payload["variant"].update(package=PACKAGE,
        config=paths[-1].relative_to(ROOT / "AdaRES").as_posix(),
        python_files={path.name: encoded(path) for path in paths[:-1]}, config_data=encoded(paths[-1]))
    program = ROOT / ("pipeline/checks/y97_rev1_cpu_" + token + "_program.py")
    output = ROOT / ("pipeline/checks/y97_revision1_preparation_" + token + ".json")
    assert not program.exists() and not output.exists()
    _, separator, body = original_text.partition("\n")
    assert separator
    text = "PAYLOAD=" + repr(payload) + "\n" + body
    assert text.partition("\n")[2] == original_text.partition("\n")[2]
    ast.parse(text, feature_version=(3, 8))
    program.write_text(text, encoding="utf-8", newline="\n")
    tick.save(output, dict(
        complete=True, token=token, id=CID, revision=1, registration=row["revision1_registration"],
        original_CPU_program=registration["original_CPU_program"], revised_CPU_program=ref(program),
        CPU_program_after_payload_byte_identical=True, original_files_unchanged=registration["original_files"],
        new_files=[ref(path) for path in paths], only_decoder_change=dict(old=OLD, new=NEW),
        no_test_executed_yet=True, helper=ref(__file__)))
    row.update(implementation_package=PACKAGE, config=payload["variant"]["config"],
        revision1_CPU_program=ref(program), revision1_preparation=ref(output))
    row["revision_history"][-1]["evidence"].extend([ref(output), ref(program)] + [ref(path) for path in paths])
    row["evidence"].extend([ref(output), ref(program)] + [ref(path) for path in paths])
    closure.validate_revisions(ROOT, row)
    tick.save(ROOT / "pipeline/research_work.json", work)
    print(json.dumps(dict(preparation=ref(output), program=ref(program), original_guards_unchanged=True)))


def record(token):
    work = tick.load(ROOT / "pipeline/research_work.json")
    row = candidate(work)
    revision = row["revision_history"][-1]
    assert row["stage"] == "implementation" and revision["result"] == "pending"
    registration = tick.load(closure.artifact(ROOT, row["revision1_registration"]))
    for entry in registration["original_files"]:
        closure.artifact(ROOT, entry)
    program = closure.artifact(ROOT, row["revision1_CPU_program"])
    log = ROOT / ("pipeline/analyses/y97_rev1_cpu_" + token + ".log")
    result_path = log.with_suffix(".json")
    assert log.exists()
    output = ROOT / ("pipeline/checks/y97_revision1_result_" + token + ".json")
    assert not output.exists()
    passed = result_path.exists()
    if passed:
        validate_cpu(tick.load(result_path))
        observation = "The unchanged full CPU controls passed for the tokenwise execution revision. "
        observation += "Actual sparse GPU task, resources, native receipt and trained efficacy remain untested."
        actual = [ref(result_path), ref(log)]
    else:
        text = log.read_text(encoding="utf-8")
        assert "Traceback (most recent call last)" in text and "AssertionError" in text
        observation = "The preregistered tokenwise execution revision failed the unchanged full CPU program. "
        observation += "Preserve the exact assertion and full raw log; no CPU/native PASS or mechanism rejection."
        actual = [ref(log)]
    tick.save(output, dict(
        complete=True, token=token, id=CID, revision=1,
        original_failure_preserved=True, original_files_unchanged=registration["original_files"],
        registration=row["revision1_registration"], revised_program=ref(program),
        result="passed" if passed else "failed", observation=observation,
        actual_result=result_path.relative_to(ROOT).as_posix() if passed else None,
        original_CPU_guards_unchanged=True, original_native_guards_unchanged=True,
        error_tail=None if passed else "\n".join(text.splitlines()[-20:]),
        CPU_only=True, native_GPU_executed=False, admission=False, efficacy_measured=False,
        recorded_at=datetime.now().astimezone().isoformat(), evidence=actual + [ref(__file__)]))
    actual.append(ref(output))
    revision["result"] = "passed" if passed else "failed"
    revision["evidence"].extend(actual)
    revision["result_evidence"] = actual
    row["evidence"].extend(actual)
    row["revision1_result"] = ref(output)
    if passed:
        row["revision1_CPU_result"] = ref(result_path)
        row["next_action"] = "Retain only the tested rev1 files and bind new CPU-verified native plans. "
        row["next_action"] += "Earlier Y94/Y95/Y96 and both running owners remain protected; CPU is not native PASS."
    else:
        row.update(stage="blocked", blocked_from_stage="implementation",
            blocker_kind="numerical_control_failure", failure_scope="implementation",
            blocker="Same-ID revision1 failed unchanged full CPU numerical controls. Original failure and "
                "revision failure remain immutable. No guard relaxation, native PASS or hypothesis rejection.",
            retry_after=(datetime.now().astimezone() + timedelta(hours=4)).isoformat(),
            unblock_condition="New evidence supporting a bounded same-ID preregistered implementation revision "
                "with every original control preserved; not a timer, tolerance change or renamed route.",
            next_action="Preserve original and revision1 files/programs/logs. Do not replay either or relax "
                "the original guards. Earlier Y94/Y95/Y96 and original Y92/Y93 continue independently.")
    closure.validate_revisions(ROOT, row)
    command = "python -B scripts/run_experiment_first_probe.py y97_rev1_cpu "
    command += row["revision1_CPU_program"]["path"] + " --token " + token + " --seconds 480"
    work["tick_attempt"] = dict(
        tick_started_at=tick.load(ROOT / "pipeline/headless/.tick_model_view.json")["generated_at"],
        candidate_id=CID, kind="implementation_test", result="passed" if passed else "failed",
        command=command, observation=observation, evidence=actual)
    tick.save(ROOT / "pipeline/research_work.json", work)
    print(json.dumps(dict(result=ref(output), revision_result=revision["result"],
                          native_executed=False, admitted=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("register", "prepare", "record"))
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    lease(args.token, 700 if args.operation == "prepare" else 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        {"register": register, "prepare": prepare, "record": record}[args.operation](args.token)
    finally:
        writer.close()
