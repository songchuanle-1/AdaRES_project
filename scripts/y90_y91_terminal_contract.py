"""Keep exact original Y90 identity while accepting only tested additive Y91."""
import ast
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT, tick
from y90_terminal_contract import SPEC, CHECKPOINT_DIRECTORY, CONFIG, IDENTITY, validate_server, validate_delta
from y91_deployment import Y90, Y91, adapt, contract, git, paths
from y91_contract import validate_result
from check_ef_execution import validate_frozen
import closure


def validate_marker(marker, cid):
    assert cid == "AdaRES-Y90" and marker in (Y90, Y91)


def build_contract(task, server):
    assert task["id"] == SPEC["id"] == "AdaRES-Y90"
    assert task["status"] == "running" and task["launch_verified"] is True
    assert str(task["gpu"]) == "0"
    assert all(task[key] == SPEC[key] for key in
               ("server", "pid", "code_commit", "config", "log"))
    validate_server(server)
    location = tick.load(closure.artifact(ROOT, SPEC["location_evidence"]))
    assert location["complete"] is True
    assert location["checkpoint_loaded"] is False and location["remote_mutations"] is False
    assert location["log"] == SPEC["log"] and location["checkpoint_directory"] == CHECKPOINT_DIRECTORY
    target = PurePosixPath(server["work_dir"]) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / "last.pth"
    assert PurePosixPath(location["declared_last_path"]) == target
    assert git("rev-parse", "HEAD").decode().strip() in (Y90, Y91)
    assert git("rev-parse", Y91 + "^").decode().strip() == Y90
    work = tick.load(ROOT / "pipeline/research_work.json")
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y91")
    assert candidate["code_commit"] == Y91 and candidate.get("native_smoke_result")
    plan = tick.load(closure.artifact(ROOT, candidate["native_smoke_plan"]))
    program = closure.artifact(ROOT, plan["program"])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding="utf-8")).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    validate_result(tick.load(closure.artifact(ROOT, candidate["native_smoke_result"])), candidate, frozen)
    deployments = contract(Y90, Y91, "AdaRES-Y90")
    assert set(deployments) == {Y90, Y91}
    for deployment in deployments.values():
        deployment["configs"] = {path: adapt(value, server) for path, value in deployment["configs"].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
