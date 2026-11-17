"""Prepare exact original Y96 lifecycle tools without loading live checkpoints."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finalize_y61_fixed30 import DISCOVER
from y84_terminal_contract import IDENTITY
from create_y89_terminal_helpers import LIVE_IDENTITY
from y96_deployment import Y96, contract, git, paths, adapt
from advance_y96 import prepared
from y96_contract import validate_result
import closure
import yaml

CID = "AdaRES-Y96"
RECEIPT = "pipeline/admissions/AdaRES-Y96_provenance_4198029edcde47b7a8686b8e8dcafad4.json"
ADMISSION = "5dcd15c8da9efca80c530a5c61767fc68f9976a2e0c70aedcef3f5807ea42b1c"
FREEZE = "50213e2f1fda98e5e9477eaa5c030952d1d913d3d1aeff86d1fc6aa984dfc154"


class Specialize(ast.NodeTransformer):
    def __init__(self, directory, log):
        replacements = {
            "pipeline/admissions/AdaRES-Y94.json": RECEIPT,
            "b9deb51efa3c274283f8a23ae678d42a4f0c11a29e758d19ae696e80df13cb84": ADMISSION,
            "7e40371552d170a1d764f9dab160a8dcb12b0b6fe512e5c978313cb7b642241b": FREEZE,
            "3f1c351f8a04e12745526e9abb3dbac46af806f1": Y96,
            "/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y94_query_geometry_bias_res_20261109_111829.log": log,
            "adares_y94_query_geometry_bias": "adares_y96_seed_query_skip",
            "20261109_111838": directory,
            "AdaRES-Y92": "AdaRES-Y94",
            "AdaRES-Y94": CID,
            "Y94": "Y96",
            "y94": "y96",
        }
        self.replacements = replacements
        self.pattern = re.compile("|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True)))

    def text(self, value):
        return self.pattern.sub(lambda match: self.replacements[match.group()], value)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.text(node.value)
        elif type(node.value) is int and node.value == 218513:
            node.value = 3893460
        return node

    def visit_Name(self, node):
        node.id = self.text(node.id)
        return node

    def visit_ImportFrom(self, node):
        node.module = self.text(node.module)
        for alias in node.names:
            alias.name = self.text(alias.name)
        return node

    def visit_alias(self, node):
        node.name = self.text(node.name)
        return node

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "SPEC" for target in node.targets):
            return node
        return self.generic_visit(node)


def create(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y96
        before = {path: ref(path) for path in ("pipeline/queue.json", "pipeline/running.json",
            "pipeline/goal.json", "SERVERS.yaml", RECEIPT)}
        index = git("ls-files", "--stage", "-z")
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate, _, _, frozen = prepared(work)
        assert candidate["stage"] == "training" and not candidate.get("terminal_handoff_plan")
        validate_result(tick.load(closure.artifact(ROOT, candidate["native_smoke_result"])), candidate, frozen)
        assert candidate["admission_receipt"] == RECEIPT and ref(RECEIPT)["sha256"] == ADMISSION
        assert candidate["post_training_freeze"]["sha256"] == FREEZE
        queue = monitor.load_queue(str(ROOT))
        task = next(row for row in queue["gpu_queues"]["server-1"] if row["id"] == CID)
        assert task["pid"] == 3893460 and task["code_commit"] == Y96
        assert task["status"] == "running" and task["launch_verified"] is True and str(task["gpu"]) == "0"
        assert monitor.load_running(str(ROOT))["server-1"]["0"] == CID
        server = monitor.load_servers(str(ROOT))["server-1"]
        closure.validate_adares_server(ROOT, server)
        closure.admission(ROOT, task)
        outputs = {name: ROOT / "scripts" / name for name in (
            "finalize_y96_fixed30.py", "test_y96_terminal_handoff.py",
            "prepare_y96_terminal_handoff.py", "y96_terminal_contract.py")}
        assert not any(path.exists() for path in outputs.values())
        deployments = contract(Y96, Y96, CID)
        config = yaml.safe_load((ROOT / "AdaRES" / task["config"]).read_bytes())
        payload = dict(id=CID, pid=task["pid"], root=server["remote_code_root"], work=server["work_dir"],
            config=task["config"], log=task["log"], basename=Path(task["config"]).stem,
            code_commit=Y96, expected_config=adapt(config, server),
            source_files={path[len("AdaRES/"):]: ref(path)["sha256"] for path in paths(CID)[:-1]},
            deployments=deployments, allowed_markers=list(deployments))
        body = DISCOVER.replace("dl01-MS-7C60", "dl02-MS-7B73").replace(
            "print(json.dumps(dict(complete=True", IDENTITY + LIVE_IDENTITY + "print(json.dumps(dict(complete=True")
        program = "P=" + repr(payload) + "\n" + body
        ast.parse(program, feature_version=(3, 8))
        base = ROOT / ("pipeline/checks/y96_terminal_original_location_" + token)
        location_path, source, log = base.with_suffix(".json"), base.with_suffix(".py"), base.with_suffix(".log")
        assert not any(path.exists() for path in (location_path, source, log))
        source.write_text(program, encoding="utf-8", newline="\n")
        response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            server["user"] + "@" + server["host"], "PYTHONDONTWRITEBYTECODE=1 " + server["python"] + " -B -"],
            input=program.encode(), capture_output=True, timeout=45)
        log.write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0, "Preserve failed read-only discovery; never restart"
        location = json.loads(response.stdout)
        assert location["complete"] is True and location["checkpoint_loaded"] is False
        assert location["remote_mutations"] is False and location["log"] == task["log"]
        tick.save(location_path, location)
        import y94_terminal_contract as original
        spec = dict(id=CID, server="server-1", pid=3893460, code_commit=Y96,
            config=task["config"], log=task["log"], package="adares_y96_seed_query_skip",
            admission_sha256=ADMISSION, freeze_sha256=FREEZE,
            checkpoint_directory=location["checkpoint_directory"], location_evidence=ref(location_path))
        old_names = ("finalize_y94_fixed30.py", "test_y94_terminal_handoff.py",
                     "prepare_y94_terminal_handoff.py", "y94_terminal_contract.py")
        originals = [ref("scripts/" + name) for name in old_names]
        transform = Specialize(location["checkpoint_directory"], task["log"])
        for old_name, target in zip(old_names, outputs.values()):
            tree = transform.visit(ast.parse((ROOT / "scripts" / old_name).read_text(encoding="utf-8")))
            if old_name == "y94_terminal_contract.py":
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(
                            isinstance(value, ast.Name) and value.id == "SPEC" for value in node.targets):
                        node.value = ast.parse(repr(spec), mode="eval").body
            text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
            ast.parse(text, feature_version=(3, 8))
            target.write_text(text, encoding="utf-8", newline="\n")
        assert [ref(entry["path"]) for entry in originals] == originals
        assert all(ref(path) == entry for path, entry in before.items())
        assert git("ls-files", "--stage", "-z") == index
        output = ROOT / ("pipeline/checks/y96_terminal_helpers_" + token + ".json")
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, id=CID, spec=spec,
            prepared_at=datetime.now().astimezone().isoformat(), originals=originals,
            source_specialization_only=True, checkpoint_loaded=False, archived=False,
            queue_changed=False, remote_mutations=False,
            evidence=[ref(__file__), ref(source), ref(log), ref(location_path)]
                + [ref(path) for path in outputs.values()]))
        candidate["evidence"].append(ref(output))
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(helpers=ref(output), original_checkpoint_directory=location["checkpoint_directory"],
                              checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    create(parser.parse_args().token)
