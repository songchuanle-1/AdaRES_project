"""Preserve original Y91 identity with an explicit real-native Y92 additive edge."""
import copy
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT, ref, tick
from y91_terminal_contract import SPEC, CID, CHECKPOINT_DIRECTORY, CONFIG, IDENTITY
from y91_terminal_contract import validate_delta
from y92_deployment import Y91, Y92, contract, git, paths, adapt
from advance_y92 import prepared
from y92_contract import validate_result
import closure


def validate_marker(marker, candidate):
    assert candidate == CID and marker in (Y91, Y92)


def native_evidence(work):
    candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y92")
    assert candidate.get("native_smoke_result"), "Actual additive Y92 native evidence required"
    candidate, _, _, frozen = prepared(work)
    entry = copy.deepcopy(candidate["native_smoke_result"])
    validate_result(tick.load(closure.artifact(ROOT, entry)), candidate, frozen)
    return entry


def build_contract(task, server):
    assert task["id"] == CID and task["status"] == "running" and task["launch_verified"] is True
    assert str(task["gpu"]) == "0"
    assert all(task[key] == SPEC[key] for key in ("server", "pid", "code_commit", "config", "log"))
    closure.validate_adares_server(ROOT, server)
    assert server["allowed_gpus"] == [0] and server["max_tasks_per_gpu"] == 1
    for key, value in dict(
        host="192.168.1.163", expected_hostname="dl01-MS-7C60",
        python="/home/songcl/.conda/envs/adares_project/bin/python",
        site_packages="/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages",
        pointnet_egg="/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg",
        remote_code_root="/bak_home/songcl/Project/AdaRES_project/AdaRES",
        work_dir="/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints",
        data_root="/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/",
    ).items():
        assert server[key] == value
    assert not server.get("runtime_path_mappings")
    assert ref("pipeline/goal.json")["sha256"] == (
        "4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a")
    assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"] == (
        "3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45")
    location = tick.load(closure.artifact(ROOT, SPEC["location_evidence"]))
    assert location["checkpoint_loaded"] is False and location["remote_mutations"] is False
    assert location["log"] == SPEC["log"] and location["checkpoint_directory"] == CHECKPOINT_DIRECTORY
    target = PurePosixPath(server["work_dir"]) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / "last.pth"
    assert PurePosixPath(location["declared_last_path"]) == target
    assert git("rev-parse", "HEAD").decode().strip() in (Y91, Y92)
    assert git("rev-parse", Y92 + "^").decode().strip() == Y91
    native_evidence(tick.load(ROOT / "pipeline/research_work.json"))
    deployments = contract(Y91, Y92, CID)
    assert list(deployments) == [Y91, Y92]
    for deployment in deployments.values():
        deployment["configs"] = {path: adapt(value, server)
                                 for path, value in deployment["configs"].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
