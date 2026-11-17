"""Exact original Y92 run; future source advances require a new tested binding."""
import json
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT
from y84_terminal_contract import IDENTITY
from y92_deployment import Y92, adapt, contract, git, paths, validate_delta
import closure

SPEC = {'id': 'AdaRES-Y92', 'server': 'server-1', 'pid': 1305468, 'code_commit': '27d6f67e7121a5168bbfeca9db745377e0bd49e5', 'config': 'configs_song/model_paper5/adares_y92_mask_feature_quadratic_res.yaml', 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y92_mask_feature_quadratic_res_20261108_090155.log', 'package': 'adares_y92_mask_feature_quadratic', 'admission_sha256': '56c0a076cc22dd82200f934781dd50b7a5972ab2adb3c383acb42cb6cb0822dc', 'freeze_sha256': 'b866bf8a784b9b44ff63cb1dd7de74fd6288b0b84a1f2280a45c9eaa85a76f95', 'checkpoint_directory': '20261108_090202', 'location_evidence': {'path': 'pipeline/checks/y92_terminal_original_location_9cbbb7d004b84132a4129c4f75781a62.json', 'sha256': '1b8348412351e420320c09bc16fd71917551b721c88f05144334b5ef7d26b553'}}
MAPPINGS = [{'source': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data', 'target': '/data1/songcl/Project/AdaRES_project/data'}, {'source': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'target': '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints'}]

CID = SPEC["id"]
CHECKPOINT_DIRECTORY = SPEC["checkpoint_directory"]
CONFIG = SPEC["config"]


def validate_marker(marker, candidate):
    assert candidate == CID and marker == Y92


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
    assert git("rev-parse", "HEAD").decode().strip() == Y92
    deployments = contract(Y92, Y92, CID)
    assert set(deployments) == {Y92}
    for deployment in deployments.values():
        deployment["configs"] = {path: adapt(value, server) for path, value in deployment["configs"].items()}
    return dict(allowed_markers=list(deployments), deployments=deployments)
