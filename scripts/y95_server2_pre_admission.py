"""Check pending Y95 server2 admission without inventing native or terminal evidence."""
import argparse
import ast
import copy
from datetime import datetime
import json

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y95_query_set_mean import files
from run_y95_bound_native_y94 import validate_binding, validate_terminal, runtime
from y95_contract import validate_result
from check_ef_execution import validate_frozen
from advance_y82 import unreserved
from y94_deployment import Y93, Y94, git
import closure


CID = "AdaRES-Y95"
COMMIT = "968112d44c506dcdd8dcf7709af997fb731e73b0"
NAME, GPU, POWER = "server-2", 0, 280.0
GRANT = "pipeline/authorizations/server2_gpu1_pause_20261104.json"
GRANT_SHA = "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45"
PAUSE, PAUSE_SHA = GRANT, GRANT_SHA
GOAL = "4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a"
PARENT_FREEZE_SHA = "8563a67a55b91f7c72b2899d03917b28d322e2f5ac14130c016511c72b0c7ef8"


def paths():
    return [path.relative_to(ROOT).as_posix() for path in files()]


def validate_delta(rows):
    assert len(rows) == 7 and all(len(row) == 2 and row[0] == "A" for row in rows)
    assert {row[1] for row in rows} == set(paths())


def pending(work, head):
    assert head == Y94, "New production HEAD requires a new tested execution binding"
    candidate = next(row for row in work["candidates"] if row["id"] == CID)
    assert candidate["code_commit"] == COMMIT
    binding = tick.load(closure.artifact(ROOT, candidate["additive_version_binding"]))
    validate_binding(binding, candidate, head)
    for dependency in binding["evidence"]:
        closure.artifact(ROOT, dependency)
    assert git("rev-parse", COMMIT + "^").decode().strip() == Y94
    delta = [row.split("\t") for row in git(
        "diff", "--no-renames", "--name-status", Y94, COMMIT).decode().splitlines()]
    validate_delta(delta)
    for path in paths():
        assert (ROOT / path).read_bytes() == git("show", COMMIT + ":" + path)
    plan = tick.load(closure.artifact(ROOT, candidate["native_smoke_plan_server2_gpu0"]))
    assert plan["server"] == NAME and plan["gpu"] == GPU and plan["runtime_power_limit_watts"] == POWER
    assert plan["execution_helper"] == ref("scripts/run_y95_bound_native_y94.py")
    assert plan["additive_parent"] == Y94 and plan["native_executed"] is False and plan["admitted"] is False
    program = closure.artifact(ROOT, plan["program"])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding="utf-8")).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    servers = monitor.load_servers(str(ROOT))
    closure.validate_adares_server(ROOT, servers[NAME])
    assert runtime(NAME, candidate, frozen, servers[NAME]) == POWER
    assert frozen["resource_authorization"] == ref(GRANT) and ref(GRANT)["sha256"] == GRANT_SHA
    assert ref(PAUSE)["sha256"] == PAUSE_SHA and ref("pipeline/goal.json")["sha256"] == GOAL
    assert servers["server-1"]["allowed_gpus"] == servers["server-2"]["allowed_gpus"] == [0]
    assert servers["server-6"]["allowed_gpus"] == []
    return candidate, plan, program, frozen


def require_native(candidate, frozen):
    assert candidate.get("native_smoke_result"), "Actual Y95 native evidence is mandatory"
    assert candidate["stage"] == "validation"
    assert (candidate["required_server"], candidate["required_gpu"]) == (NAME, GPU)
    assert candidate["native_smoke_plan"] == candidate["native_smoke_plan_server2_gpu0"]
    result = tick.load(closure.artifact(ROOT, candidate["native_smoke_result"]))
    validate_result(result, candidate, frozen)
    return result


def require_completed_owner():
    done = monitor.load_experiment(str(ROOT), "AdaRES-Y93")
    assert done is not None and done["status"] == "done_complete", "Y93 is not canonically complete"
    assert done["server"] == NAME and str(done["gpu"]) == "0" and done["code_commit"] == Y93
    validate_terminal(tick.load(closure.artifact(ROOT, done["terminal_evidence"])), NAME)
    return done


def make_freeze(parent, parentref):
    """Build a memory-only draft bound to the complete original protocol."""
    assert parentref["sha256"] == PARENT_FREEZE_SHA
    assert parent == tick.load(closure.artifact(ROOT, parentref))
    assert parent["complete"] is True and parent["id"] == "AdaRES-Y93" and parent["code_commit"] == Y93
    assert parent["goal"] == ref("pipeline/goal.json") and parent["goal"]["sha256"] == GOAL
    assert parent["evaluation"]["methods"][0] == "AdaRES-Y93"
    assert parent["resource_authorization"] == ref(GRANT) and ref(GRANT)["sha256"] == GRANT_SHA
    for key, value in dict(server=NAME, gpu=GPU, power_limit_watts=POWER).items():
        assert parent["resource_condition"][key] == value
    freeze = copy.deepcopy(parent)
    freeze.update(id=CID, frozen_at=datetime.now().astimezone().isoformat(),
        code_commit=COMMIT, parent_protocol=parentref,
        scientific_efficacy="Unknown until original full frozen evaluation.")
    freeze["evaluation"]["methods"][0] = CID
    restored = copy.deepcopy(freeze["evaluation"])
    restored["methods"][0] = parent["id"]
    assert restored == parent["evaluation"] and freeze["training"] == parent["training"]
    assert freeze["resource_condition"] == parent["resource_condition"]
    return freeze


def verify(token):
    lease(token, 90)
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate, _, _, frozen = pending(work, git("rev-parse", "HEAD").decode().strip())
    result = require_native(candidate, frozen)
    done = require_completed_owner()
    unreserved(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), NAME, GPU)
    print(json.dumps(dict(id=CID, actual_native_result=candidate["native_smoke_result"],
        losses=result["losses"], exact_original_terminal=done["terminal_evidence"],
        admitted=False, source_selected=False, launched=False,
        next_action="Bind the still-running Y94 terminal contract to actual Y95 native-tested "
            "additions with NEW tested helpers before selecting source. Then complete exact source "
            "selection/full original freeze/receipt/admission/canonical dispatch and increasing native "
            "counters in the same authorized tick when feasible. This verifier does not execute or "
            "certify those remaining operations or authorize the separate server1 alternative.")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    verify(parser.parse_args().token)
