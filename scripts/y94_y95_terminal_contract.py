"""Bind only actual-native-tested Y95 additions to the original Y94 lifecycle."""
import copy
from pathlib import PurePosixPath

from prepare_y60_stage_heads import ROOT, ref, tick
from y94_terminal_contract import (
    SPEC, CID, CHECKPOINT_DIRECTORY, CONFIG, IDENTITY, validate_delta, paths, validate_server,
)
from y95_deployment import Y94, Y95, adapt, contract, git
from advance_y95 import prepared
from y95_contract import validate_result
import closure


def validate_marker(marker, candidate):
    assert candidate == CID and marker in (Y94, Y95)


def native_evidence(work):
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y95")
    assert candidate.get("native_smoke_result"), "Actual additive Y95 native evidence required"
    candidate, _, _, frozen = prepared(work)
    entry = copy.deepcopy(candidate["native_smoke_result"])
    validate_result(tick.load(closure.artifact(ROOT, entry)), candidate, frozen)
    return entry


def build_contract(task, server):
    assert task["id"] == CID and task["status"] == "running"
    assert task["launch_verified"] is True and str(task["gpu"]) == "0"
    assert all(task[key] == SPEC[key] for key in
               ("server", "pid", "code_commit", "config", "log"))
    validate_server(server)
    assert ref("pipeline/goal.json")["sha256"] == (
        "4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a")
    assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == (
        "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45")
    location = tick.load(closure.artifact(ROOT, SPEC["location_evidence"]))
    assert location["complete"] is True and location["checkpoint_loaded"] is False
    assert location["remote_mutations"] is False and location["log"] == SPEC["log"]
    assert location["checkpoint_directory"] == CHECKPOINT_DIRECTORY
    expected = PurePosixPath(server["work_dir"]) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / "last.pth"
    assert PurePosixPath(location["declared_last_path"]) == expected
    assert git("rev-parse", "HEAD").decode().strip() in (Y94, Y95)
    assert git("rev-parse", Y95 + "^").decode().strip() == Y94
    native_evidence(tick.load(ROOT / "pipeline/research_work.json"))
    deployments = contract(Y94, Y95, CID)
    assert list(deployments) == [Y94, Y95]
    for deployment in deployments.values():
        deployment["configs"] = {path: adapt(value, server)
                                 for path, value in deployment["configs"].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
