"""Preserve the original Y95 contract across actual-native Y96 additions."""
import copy
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT, ref, tick
import y95_terminal_contract as original
from y95_terminal_contract import SPEC, CID, CHECKPOINT_DIRECTORY, CONFIG, IDENTITY
from y96_deployment import Y95, Y96, contract, git, paths, validate_delta as validate_new_delta
from advance_y96 import prepared
from y96_contract import validate_result
import closure


def validate_delta(rows, cid):
    if cid == "AdaRES-Y95":
        return original.validate_delta(rows, cid)
    return validate_new_delta(rows, cid)


def validate_marker(marker, candidate):
    assert candidate == CID and marker in (Y95, Y96)


def native_evidence(work):
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y96")
    assert candidate.get("native_smoke_result"), "Actual additive Y96 native evidence required"
    candidate, _, _, frozen = prepared(work)
    entry = copy.deepcopy(candidate["native_smoke_result"])
    validate_result(tick.load(closure.artifact(ROOT, entry)), candidate, frozen)
    return entry


def build_contract(task, server):
    head = git("rev-parse", "HEAD").decode().strip()
    assert head in (Y95, Y96)
    assert git("rev-parse", Y96 + "^").decode().strip() == Y95
    assert task["id"] == CID and task["status"] == "running" and task["launch_verified"] is True
    assert str(task["gpu"]) == "0"
    assert all(task[key] == SPEC[key] for key in ("server", "pid", "code_commit", "config", "log"))
    closure.validate_adares_server(ROOT, server)
    assert server["allowed_gpus"] == [0] and server["max_tasks_per_gpu"] == 1
    for key, value in dict(host="192.168.1.163", expected_hostname="dl01-MS-7C60",
            python="/home/songcl/.conda/envs/adares_project/bin/python",
            remote_code_root="/bak_home/songcl/Project/AdaRES_project/AdaRES",
            work_dir="/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints").items():
        assert server[key] == value
    assert not server.get("runtime_path_mappings")
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
    native_evidence(tick.load(ROOT / "pipeline/research_work.json"))
    deployments = contract(Y95, Y96, CID)
    assert list(deployments) == [Y95, Y96]
    return dict(allowed_markers=list(deployments), deployments=deployments)
