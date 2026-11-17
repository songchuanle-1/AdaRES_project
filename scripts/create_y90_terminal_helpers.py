"""Specialize existing server-1 lifecycle checks for the exact live Y90 run."""
import argparse
import ast
from datetime import datetime
from pathlib import Path, PurePosixPath

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finalize_y61_fixed30 import DISCOVER
from finalize_y88_fixed30 import remote
from y84_terminal_contract import IDENTITY
from y90_deployment import Y90, adapt, contract, git, paths
import closure
import yaml


CID = "AdaRES-Y90"
SCRIPTS = ROOT / "scripts"
SPEC = dict(
    id=CID,
    server="server-1",
    pid=2442962,
    code_commit=Y90,
    package="adares_y90_query_peer_affinity",
    config="configs_song/model_paper5/adares_y90_query_peer_affinity_res.yaml",
    log="/data1/songcl/Project/AdaRES_project/AdaRES/logs/"
        "train_adares_y90_query_peer_affinity_res_20261107_071419.log",
    admission_sha256="f790bc29637612751ae4e483a2ad21efa6c468d273ddadf4523ad182bb585f56",
    freeze_sha256="ad60ab5a94db03b12405a043be738aae978786d560d3b707357150994ec95395",
)
MAPPINGS = [
    dict(source="/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data",
         target="/data1/songcl/Project/AdaRES_project/data"),
    dict(source="/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints",
         target="/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints"),
]
LIVE_IDENTITY = """
process=pathlib.Path('/proc')/str(P['pid'])
assert process.is_dir() and (process/'cwd').resolve()==pathlib.Path(P['root'])
args=(process/'cmdline').read_bytes().split(b'\\0')
assert b'tools/train.py' in args and P['config'].encode() in args
"""
CONTRACT = '''
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
'''


class Specialize(ast.NodeTransformer):
    def __init__(self, directory):
        self.replacements = [
            ("adares_y88_mask_query_quadratic", SPEC["package"]),
            ("20261106_053553", directory),
            ("cb4a5813755d38b3d1593bb803726aabd7d0a79b", Y90),
            ("39f9a71debe48314d479b42b604565cf557ccbe40e5582e949a66959587b3572",
             SPEC["admission_sha256"]),
            ("93bf51b65a580cbdd5a7ade69fae370d8d0e6e6db5ab665645892c775d84aa81",
             SPEC["freeze_sha256"]),
            ("scripts/current_terminal_contract.py", "scripts/create_y90_terminal_helpers.py"),
            ("Y88", "Y90"),
            ("y88", "y90"),
        ]

    def text(self, value):
        for old, new in self.replacements:
            value = value.replace(old, new)
        return value

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.text(node.value)
        elif type(node.value) is int and node.value == 3527278:
            node.value = SPEC["pid"]
        return node

    def visit_ImportFrom(self, node):
        node.module = self.text(node.module)
        return self.generic_visit(node)

    def visit_alias(self, node):
        node.name = self.text(node.name)
        return node

    def visit_Name(self, node):
        node.id = self.text(node.id)
        return node

    def visit_keyword(self, node):
        if node.arg:
            node.arg = self.text(node.arg)
        return self.generic_visit(node)


def create(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y90
        before = {path: ref(path) for path in (
            "pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
        index = git("ls-files", "--stage", "-z")
        work = tick.load(ROOT / "pipeline/research_work.json")
        original_candidates = {row["id"]: row for row in work["candidates"] if row["id"] != CID}
        candidate = next(row for row in work["candidates"] if row["id"] == CID)
        assert candidate["stage"] == "training" and not candidate.get("terminal_handoff_plan")
        assert not candidate.get("terminal_evidence")
        queue = monitor.load_queue(str(ROOT))
        task = next(row for row in queue["gpu_queues"]["server-1"] if row["id"] == CID)
        assert task["status"] == "running" and task["launch_verified"] is True
        assert str(task["gpu"]) == "0"
        assert all(task[key] == SPEC[key] for key in
                   ("id", "server", "pid", "code_commit", "config", "log"))
        assert monitor.load_running(str(ROOT))["server-1"]["0"] == CID
        server = monitor.load_servers(str(ROOT))["server-1"]
        closure.validate_adares_server(ROOT, server)
        assert server["runtime_path_mappings"] == MAPPINGS
        assert server["python"] == "/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python"
        assert server["runtime_library_path"] == "/data1/songcl/Project/AdaRES_project/runtime/cuda113lib64"
        closure.admission(ROOT, task)
        assert ref("pipeline/admissions/" + CID + ".json")["sha256"] == SPEC["admission_sha256"]
        assert candidate["post_training_freeze"]["sha256"] == SPEC["freeze_sha256"]
        closure.artifact(ROOT, candidate["post_training_freeze"])
        names = ("finalize_y90_fixed30.py", "test_y90_terminal_handoff.py",
                 "prepare_y90_terminal_handoff.py", "y90_terminal_contract.py")
        outputs = {name: SCRIPTS / name for name in names}
        assert not any(path.exists() for path in outputs.values())
        originals = [ref("scripts/" + name.replace("y90", "y88")) for name in names]
        originals.extend(ref("scripts/" + name) for name in (
            "finalize_y89_fixed30.py", "prepare_y89_terminal_handoff.py",
            "test_y89_terminal_handoff.py", "y89_terminal_contract.py"))
        config = yaml.safe_load((ROOT / "AdaRES" / SPEC["config"]).read_bytes())
        deployments = contract(Y90, Y90, CID)
        payload = dict(
            id=CID, pid=SPEC["pid"], root=server["remote_code_root"], work=server["work_dir"],
            config=SPEC["config"], log=SPEC["log"], basename=Path(SPEC["config"]).stem,
            code_commit=Y90, expected_config=adapt(config, server),
            source_files={path[len("AdaRES/"):]: ref(path)["sha256"] for path in paths(CID)[:-1]},
            deployments=deployments, allowed_markers=list(deployments))
        body = DISCOVER.replace("dl01-MS-7C60", "dl02-MS-7B73").replace(
            "print(json.dumps(dict(complete=True",
            IDENTITY + LIVE_IDENTITY + "print(json.dumps(dict(complete=True")
        program = "P=" + repr(payload) + "\n" + body
        ast.parse(program, feature_version=(3, 8))
        base = ROOT / ("pipeline/checks/y90_terminal_original_location_" + token)
        source, location_path = base.with_suffix(".py"), base.with_suffix(".json")
        assert not source.exists() and not location_path.exists()
        source.write_text(program, encoding="utf-8", newline="\n")
        location = remote(server, program, location_path)
        assert location["complete"] is True
        assert location["checkpoint_loaded"] is False and location["remote_mutations"] is False
        assert location["log"] == SPEC["log"]
        directory = location["checkpoint_directory"]
        expected = PurePosixPath(server["work_dir"]) / payload["basename"] / directory / "last.pth"
        assert PurePosixPath(location["declared_last_path"]) == expected
        tick.save(location_path, location)
        spec = dict(SPEC, checkpoint_directory=directory, location_evidence=ref(location_path))
        generated = {
            "y90_terminal_contract.py":
                '"""Exact original Y90 run; future source advances require a new tested binding."""\n'
                "import json\nfrom pathlib import PurePosixPath\n"
                "from prepare_y60_stage_heads import ROOT\n"
                "from y84_terminal_contract import IDENTITY\n"
                "from y90_deployment import Y90, adapt, contract, git, paths, validate_delta\n"
                "import closure\n\nSPEC = " + repr(spec) + "\nMAPPINGS = " + repr(MAPPINGS) + "\n" + CONTRACT
        }
        for name in names[:-1]:
            template = (SCRIPTS / name.replace("y90", "y88")).read_text(encoding="utf-8")
            tree = Specialize(directory).visit(ast.parse(template))
            ast.fix_missing_locations(tree)
            generated[name] = ast.unparse(tree) + "\n"
        for name, text in generated.items():
            ast.parse(text, feature_version=(3, 8))
            assert "adares_y88_mask_query_quadratic" not in text
            assert "3527278" not in text and "20261106_053553" not in text
            assert not outputs[name].exists()
        for name, text in generated.items():
            outputs[name].write_text(text, encoding="utf-8", newline="\n")
        assert [ref(ROOT / item["path"]) for item in originals] == originals
        assert all(ref(path) == value for path, value in before.items())
        assert git("ls-files", "--stage", "-z") == index
        assert {row["id"]: row for row in work["candidates"] if row["id"] != CID} == original_candidates
        output = ROOT / ("pipeline/checks/y90_terminal_helpers_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(
            complete=True, token=token, id=CID, spec=spec,
            prepared_at=datetime.now().astimezone().isoformat(), originals=originals,
            evidence=[ref(source), ref(location_path), ref(location_path.with_suffix(".log")),
                      ref(__file__)] + [ref(path) for path in outputs.values()],
            source_specialization_only=True, original_helpers_unchanged=True,
            checkpoint_loaded=False, archived=False, queue_changed=False, remote_mutations=False,
            main_index_unchanged=True, other_candidates_unchanged=True))
        candidate["evidence"].append(ref(output))
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(__import__("json").dumps(dict(
            helpers=ref(output), original_checkpoint_directory=directory,
            checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
