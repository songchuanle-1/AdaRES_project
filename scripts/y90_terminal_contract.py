"""Exact original Y90 run; future source advances require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT
from y84_terminal_contract import IDENTITY
from y90_deployment import Y90, adapt, contract, git, paths, validate_delta
import closure

SPEC = {'id': 'AdaRES-Y90', 'server': 'server-1', 'pid': 2442962, 'code_commit': '2d2cccb4f6f6ef9daf960b0ac255a3a3ee5e9f71', 'package': 'adares_y90_query_peer_affinity', 'config': 'configs_song/model_paper5/adares_y90_query_peer_affinity_res.yaml', 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y90_query_peer_affinity_res_20261107_071419.log', 'admission_sha256': 'f790bc29637612751ae4e483a2ad21efa6c468d273ddadf4523ad182bb585f56', 'freeze_sha256': 'ad60ab5a94db03b12405a043be738aae978786d560d3b707357150994ec95395', 'checkpoint_directory': '20261107_071428', 'location_evidence': {'path': 'pipeline/checks/y90_terminal_original_location_c9cb85a8c1c94672bfcf593f05905cbb.json', 'sha256': 'b61e808472b435075592edca90f5aea2fb589c99739be172a11f2c52b568e855'}}
MAPPINGS = [{'source': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data', 'target': '/data1/songcl/Project/AdaRES_project/data'}, {'source': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'target': '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints'}]

CID = SPEC["id"]
CHECKPOINT_DIRECTORY = SPEC["checkpoint_directory"]
CONFIG = SPEC["config"]


def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y90


def validate_server(server):
    closure.validate_adares_server(ROOT, server)
    assert server["allowed_gpus"] == [0] and server["max_tasks_per_gpu"] == 1
    for key, value in dict(
        host="192.168.1.175", expected_hostname="dl02-MS-7B73",
        python="/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python",
        runtime_library_path="/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64",
        remote_code_root="/data1/songcl/Project/AdaRES_project/AdaRES",
        work_dir="/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints",
    ).items():
        assert server[key] == value
    assert server["runtime_path_mappings"] == MAPPINGS


def build_contract(task, server):
    assert task["id"] == CID and task["status"] == "running"
    assert task["launch_verified"] is True and str(task["gpu"]) == "0"
    assert all(task[key] == SPEC[key] for key in
               ("server", "pid", "code_commit", "config", "log"))
    validate_server(server)
    location = json.loads(closure.artifact(ROOT, SPEC["location_evidence"]).read_text(encoding="utf-8"))
    assert location["complete"] is True
    assert location["checkpoint_loaded"] is False and location["remote_mutations"] is False
    assert location["log"] == SPEC["log"] and location["checkpoint_directory"] == CHECKPOINT_DIRECTORY
    expected = PurePosixPath(server["work_dir"]) / PurePosixPath(CONFIG).stem / CHECKPOINT_DIRECTORY / "last.pth"
    assert PurePosixPath(location["declared_last_path"]) == expected
    assert git("rev-parse", "HEAD").decode().strip() == Y90
    deployments = contract(Y90, Y90, CID)
    assert set(deployments) == {Y90}
    for deployment in deployments.values():
        deployment["configs"] = {path: adapt(value, server) for path, value in deployment["configs"].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
