"""Exact original Y94 run; future source advances require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT
from y84_terminal_contract import IDENTITY
from y94_deployment import Y94, adapt, contract, git, paths, validate_delta
import closure

SPEC = {'id': 'AdaRES-Y94', 'server': 'server-1', 'pid': 218513, 'code_commit': '3f1c351f8a04e12745526e9abb3dbac46af806f1', 'config': 'configs_song/model_paper5/adares_y94_query_geometry_bias_res.yaml', 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y94_query_geometry_bias_res_20261109_111829.log', 'package': 'adares_y94_query_geometry_bias', 'admission_sha256': 'b9deb51efa3c274283f8a23ae678d42a4f0c11a29e758d19ae696e80df13cb84', 'freeze_sha256': '7e40371552d170a1d764f9dab160a8dcb12b0b6fe512e5c978313cb7b642241b', 'checkpoint_directory': '20261109_111838', 'location_evidence': {'path': 'pipeline/checks/y94_terminal_original_location_9510373f9f21442591b649640928a5e5.json', 'sha256': '17a4919015244b7b335841ccd85044e21e3c81eb47e3eb3e0f087b1705dfa1a9'}}
MAPPINGS = [{'source': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data', 'target': '/data1/songcl/Project/AdaRES_project/data'}, {'source': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'target': '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints'}]

CID = SPEC["id"]
CHECKPOINT_DIRECTORY = SPEC["checkpoint_directory"]
CONFIG = SPEC["config"]


def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y94


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
    assert git("rev-parse", "HEAD").decode().strip() == Y94
    deployments = contract(Y94, Y94, CID)
    assert set(deployments) == {Y94}
    for deployment in deployments.values():
        deployment["configs"] = {path: adapt(value, server) for path, value in deployment["configs"].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
