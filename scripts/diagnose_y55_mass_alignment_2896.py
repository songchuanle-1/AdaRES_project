"""Freeze, run and record one independent CPU bookkeeping diagnosis."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "289611be465740cc983c5246b0eb27f6"
ID = "AdaRES-Y55"
SKILL = Path("C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
sys.path.insert(0, str(SKILL))
import foreground_tick as tick
import pipeline_monitor as monitor

PLAN = "pipeline/analyses/y55_mass_alignment_prereg_" + TOKEN + ".json"
PROGRAM = "pipeline/analyses/y55_mass_alignment_program_" + TOKEN + ".py"
RESULT = "pipeline/analyses/y55_mass_alignment_result_" + TOKEN + ".json"
TRANSCRIPT = "pipeline/analyses/y55_mass_alignment_execution_" + TOKEN + ".json"
BODY = "scripts/probe_y55_mass_alignment_2896.py"


def ref(relative):
    path = (ROOT / relative).resolve()
    assert ROOT in path.parents and path.is_file()
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def owned(minimum_seconds=120):
    owner = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
    assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > minimum_seconds
    for name, key in ((".tick_context.json", "context_sha256"), (".tick_context_state.next.json", "pending_sha256")):
        assert ref("pipeline/headless/" + name)["sha256"] == owner[key]
    return owner


def prepare():
    owned(300)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        assert not any(candidate["id"] == ID for candidate in work["candidates"])
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        assert view["design_bootstrap"]["required"] and view["research"]["due_count"] == 0
        assert not view["research"]["reserve_only"]
        assert view["research"]["reserve_count"] == view["research"]["reserve_target"] == 2
        sources = []
        for name in ("ipdn.py", "dec.py"):
            local = "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/" + name
            sources.append(dict(**ref(local), remote_path=local[len("AdaRES/"):],
                lf_sha256=hashlib.sha256((ROOT / local).read_bytes().replace(b"\r\n", b"\n")).hexdigest()))
        local_config = "AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
        sources.append(dict(**ref(local_config), remote_path=local_config[len("AdaRES/"):],
            lf_sha256=hashlib.sha256((ROOT / local_config).read_bytes().replace(b"\r\n", b"\n")).hexdigest()))
        payload = dict(id=ID, token=TOKEN, sources=sources,
            remote_root="/bak_home/songcl/Project/AdaRES_project/AdaRES",
            capture_path="/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json",
            fixtures=[
                dict(name="one_scene_three_expressions", masses=[1, 4, 2, 7], offsets=[0, 4], scenes_len=[3]),
                dict(name="two_scenes_unequal_expression_counts", masses=[1, 5, 2, 4, 9, 3, 8], offsets=[0, 3, 7], scenes_len=[1, 3]),
                dict(name="three_scenes_nonuniform_support", masses=[2, 7, 1, 6, 3, 9, 4, 8, 5], offsets=[0, 2, 5, 9], scenes_len=[3, 1, 2]),
            ])
        source = "PAYLOAD = " + repr(payload) + "\n" + (ROOT / BODY).read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 8))
        assert not (ROOT / PLAN).exists() and not (ROOT / PROGRAM).exists()
        hypothesis = ("Sentence expansion may misalign W2's point-count mass with the scene/superpoint "
            "represented by each expanded feature row, corrupting metric-aligned query competition. "
            "The suspected defect is cross-module ordering, not changing the point-mass formula or score target.")
        control = ("Execute unchanged IPDN.point_masses and expand_and_fps bodies, DEC.get_batches and "
            "the exact native indicator on CPU. Independently count original point memberships with Counter "
            "and index the reference by expanded original-superpoint identity. Use three frozen unequal "
            "replication fixtures and the existing eight TRAIN scenes/four pairs/32 expressions, same seed1999. "
            "Identical-repeat/independent-reference controls must agree exactly; deliberately shifted valid "
            "masses are a sensitivity control. Substitute only FPS with deterministic CPU indices, explicitly "
            "not a full native/FPS test; FPS indices never change feature expansion order.")
        plan = dict(complete=True, id=ID, token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(),
            hypothesis_id="sentence_expansion_pointmass_alignment", hypothesis=hypothesis, matched_control=control,
            source_bindings=sources, body=ref(BODY), helper=ref("scripts/diagnose_y55_mass_alignment_2896.py"),
            first_test="Run the fixed CPU bookkeeping/packing and native indicator reference/shift controls; no new model or pending reserve.",
            falsification="Exact point-mass alignment across frozen actual inputs and adversarial unequal-copy fixtures, with active shift controls and matching source-order identity, rejects THIS expansion-mapping defect. Any mismatch stays diagnosis; do not relabel an unrun test.",
            boundaries="Diagnosis only, no model/config edits, Git commit, admission, weights, CUDA work or extra pending model reserve. No efficacy or novelty claim. This is distinct from Y52's score target and Y54's fine spatial support.",
            reserve_state="Two existing blocked prototypes retained; no additional prototype preparation is authorized or claimed.",
            program=None, result="pending")
        (ROOT / PROGRAM).write_text(source, encoding="utf-8", newline="\n")
        plan["program"] = ref(PROGRAM)
        tick.save(ROOT / PLAN, plan)
        work["candidates"].append(dict(id=ID, hypothesis_id=plan["hypothesis_id"], stage="diagnosis",
            hypothesis=hypothesis, matched_control=control, next_action=plan["first_test"],
            evidence=[ref(PLAN), ref(PROGRAM), ref(BODY), ref("scripts/diagnose_y55_mass_alignment_2896.py")],
            failure_scope="inconclusive", created_at=plan["preregistered_at"],
            gate_summary=dict(diagnosis="pending_contract_test", controls="pending", native_backward="not_run",
                resources="CPU_diagnostic_only", independence="not_implemented", constraints="no_admission")))
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(id=ID, stage="diagnosis", preregistration=ref(PLAN), new_model_files=0)))
    finally:
        lock.close()


def run():
    owned(240)
    plan = tick.load(ROOT / PLAN)
    assert ref(PROGRAM) == plan["program"] and ref(BODY) == plan["body"]
    assert not (ROOT / RESULT).exists() and not (ROOT / TRANSCRIPT).exists()
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "songcl@192.168.1.163",
        "env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/bak_home/songcl/Project/AdaRES_project/AdaRES "
        "/home/songcl/.conda/envs/adares_project/bin/python -B -c "
        "\"import sys; exec(compile(sys.stdin.read(), '<y55-frozen-cpu-probe>', 'exec'))\"",
    ]
    start = datetime.now().astimezone().isoformat()
    executed = subprocess.run(command, input=(ROOT / PROGRAM).read_text(encoding="utf-8"),
        text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=210)
    owned(20)
    transcript = dict(id=ID, token=TOKEN, started_at=start, observed_at=datetime.now().astimezone().isoformat(),
        returncode=executed.returncode, stdout=executed.stdout, stderr=executed.stderr, program=ref(PROGRAM))
    tick.save(ROOT / TRANSCRIPT, transcript)
    if executed.returncode:
        raise RuntimeError("Y55 CPU probe failed; preserved " + TRANSCRIPT)
    matches = [line[len("Y55_RESULT="):] for line in executed.stdout.splitlines() if line.startswith("Y55_RESULT=")]
    assert len(matches) == 1
    result = json.loads(matches[0])
    assert result["complete"] and result["token"] == TOKEN and result["id"] == ID
    result["execution"] = ref(TRANSCRIPT)
    result["observed_at"] = transcript["observed_at"]
    tick.save(ROOT / RESULT, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


def record():
    owned(90)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        result = tick.load(ROOT / RESULT)
        assert result["complete"] and result["mass_mapping_mismatches"] == 0
        assert result["indicator_reference_max_abs"] == result["indicator_repeat_max_abs"] == 0
        assert result["all_positive_controls_active"] and result["actual_TRAIN_expressions"] == 32
        assert result["cpu_only"] and not result["cuda_initialized"]
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(item for item in work["candidates"] if item["id"] == ID)
        assert candidate["stage"] == "diagnosis"
        for row in candidate["evidence"]:
            assert ref(row["path"]) == row
        candidate["evidence"].extend([ref(RESULT), ref(TRANSCRIPT)])
        candidate.update(stage="rejected", failure_scope="hypothesis",
            rejection_reason="The actual source's scene-major whole-scene repeats agree exactly; three unequal-copy fixtures and four frozen TRAIN pairs (32 expressions) have zero misassigned mass, exact totals/packing, zero native-indicator reference/repeat delta, and active deliberately shifted-mass controls. This specific ordering-defect premise is contradicted, not all metric-aligned evidence.",
            next_action="Preserve source-bound CPU falsification. Do not build a fix or requeue this nonexistent mapping defect; do not infer task accuracy, full native gates or novelty.",
            gate_summary=dict(diagnosis="hypothesis_refuted_in_fixed_contract_scope", controls="executed_reference_repeat_and_shift",
                native_backward="not_run_not_needed_for_refutation", resources="CPU_only_no_GPU_or_training",
                independence="no_model_variant_created", constraints="diagnostic_only"))
        blocked = [item for item in work["candidates"] if item.get("stage") == "blocked"]
        assert {item["id"] for item in blocked} == {"AdaRES-Y52", "AdaRES-Y54"}
        work["design_cycle"] = dict(stop_reason="external_resource_blocked", attempted_ids=[ID],
            evidence=[ref(PLAN), ref(RESULT)],
            resume_condition="Existing Y52/Y54 scoped tested-byte commit authorization or matching commit; revalidate current goal/source/gates/commit before gated dispatch and live resource checks. Keep existing four-hour authorization retry and wake paths; do not expand Y50 permission.",
            clarification="This tick tested and rejected an independent bookkeeping defect, without creating another model/prototype. The bounded reserve remains two unchanged blocked candidates, no due work. No known executable unfinished diagnosis/gate remains; missing explicit human commit permission, not GPU availability or invented route exhaustion, prevents their admission.")
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(id=ID, stage="rejected", failure_scope="hypothesis", reserve_count=2,
            reserve_candidates=["AdaRES-Y52", "AdaRES-Y54"], admissions=0, launches=0), indent=2))
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "record"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "record": record}[args.phase]()


if __name__ == "__main__":
    main()
