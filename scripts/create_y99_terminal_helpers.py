"""Bind future Y99 terminal checks to its actual original server1 directory."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finalize_y61_fixed30 import DISCOVER
from y84_terminal_contract import IDENTITY
from create_y89_terminal_helpers import LIVE_IDENTITY
from y99_deployment_v2 import Y99, contract, git, paths, adapt
from advance_y99_v2 import prepared, validate_result
import closure

CID = "AdaRES-Y99"
RECEIPT = "pipeline/admissions/AdaRES-Y99.json"
ADMISSION = "4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede"
FREEZE = "fd07c0536891d905a7cd89e2b4211b32ed31a532b7b70ac9d71a8d7a66e97dbc"


class Specialize(ast.NodeTransformer):
    def __init__(self, directory, log):
        changes = {
            "pipeline/admissions/AdaRES-Y96_provenance_4198029edcde47b7a8686b8e8dcafad4.json": RECEIPT,
            "5dcd15c8da9efca80c530a5c61767fc68f9976a2e0c70aedcef3f5807ea42b1c": ADMISSION,
            "50213e2f1fda98e5e9477eaa5c030952d1d913d3d1aeff86d1fc6aa984dfc154": FREEZE,
            "acb8c1769b7f3d66a92d287da8859a06a138b8b0": Y99,
            "/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y96_seed_query_skip_res_20261110_165652.log": log,
            "adares_y96_seed_query_skip": "adares_y99_spatial_mask_skip",
            "20261110_165702": directory,
            "AdaRES-Y94": "AdaRES-Y96",
            "only_exact_Y94_archived": "only_exact_Y99_archived",
            "y96_deployment": "y99_deployment_v2",
            "advance_y96": "advance_y99_v2",
            "Y96": "Y99", "y96": "y99",
        }
        self.changes = changes
        self.pattern = re.compile("|".join(re.escape(k) for k in sorted(changes, key=len, reverse=True)))

    def text(self, value):
        return self.pattern.sub(lambda m: self.changes[m.group()], value)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.text(node.value)
        elif type(node.value) is int and node.value == 3893460:
            node.value = 2980836
        return node

    def visit_Name(self, node):
        node.id = self.text(node.id)
        return node

    def visit_Attribute(self, node):
        node.attr = self.text(node.attr)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        node.module = self.text(node.module)
        return self.generic_visit(node)

    def visit_alias(self, node):
        node.name = self.text(node.name)
        if node.asname:
            node.asname = self.text(node.asname)
        return node

    def visit_Assign(self, node):
        if any(isinstance(n, ast.Name) and n.id == "SPEC" for n in node.targets):
            return node
        return self.generic_visit(node)


def create(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse", "HEAD").decode().strip() == Y99
        before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml", RECEIPT)}
        index = git("ls-files", "--stage", "-z")
        work = tick.load(ROOT / "pipeline/research_work.json")
        c, _, _, frozen = prepared(work)
        assert c["stage"] == "training" and not c.get("terminal_handoff_plan")
        validate_result(tick.load(closure.artifact(ROOT, c["native_smoke_result"])), c, frozen)
        assert c["admission_receipt"] == RECEIPT and ref(RECEIPT)["sha256"] == ADMISSION
        assert c["post_training_freeze"]["sha256"] == FREEZE
        queue = monitor.load_queue(str(ROOT))
        task = next(r for r in queue["gpu_queues"]["server-1"] if r["id"] == CID)
        assert (task["pid"], task["code_commit"], task["status"], task["launch_verified"], str(task["gpu"])) == (2980836, Y99, "running", True, "0")
        assert monitor.load_running(str(ROOT))["server-1"]["0"] == CID
        server = monitor.load_servers(str(ROOT))["server-1"]
        closure.validate_adares_server(ROOT, server)
        closure.admission(ROOT, task)
        old_names = ("finalize_y96_fixed30.py", "test_y96_terminal_handoff.py", "prepare_y96_terminal_handoff.py", "y96_terminal_contract.py")
        targets = [ROOT / "scripts" / name.replace("y96", "y99") for name in old_names]
        assert not any(p.exists() for p in targets)
        deployments = contract(Y99, Y99, CID)
        payload = dict(id=CID, pid=task["pid"], root=server["remote_code_root"], work=server["work_dir"],
            config=task["config"], log=task["log"], basename=Path(task["config"]).stem,
            code_commit=Y99, expected_config=adapt(yaml.safe_load((ROOT / "AdaRES" / task["config"]).read_bytes()), server),
            source_files={p[len("AdaRES/"):]: ref(p)["sha256"] for p in paths(CID)[:-1]},
            deployments=deployments, allowed_markers=list(deployments))
        body = DISCOVER.replace("dl01-MS-7C60", "dl02-MS-7B73").replace("print(json.dumps(dict(complete=True", IDENTITY + LIVE_IDENTITY + "print(json.dumps(dict(complete=True")
        program = "P=" + repr(payload) + "\n" + body
        ast.parse(program, feature_version=(3, 8))
        base = ROOT / ("pipeline/checks/y99_terminal_original_location_" + token)
        location_path, source, log = (base.with_suffix(s) for s in (".json", ".py", ".log"))
        assert not any(p.exists() for p in (location_path, source, log))
        source.write_text(program, encoding="utf-8", newline="\n")
        response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", server["user"] + "@" + server["host"],
            "PYTHONDONTWRITEBYTECODE=1 " + server["python"] + " -B -"], input=program.encode(), capture_output=True, timeout=45)
        log.write_bytes(response.stdout + response.stderr)
        assert response.returncode == 0, "Preserve failed read-only discovery; never restart"
        location = json.loads(response.stdout)
        assert location["complete"] and not location["checkpoint_loaded"] and not location["remote_mutations"]
        assert location["log"] == task["log"]
        tick.save(location_path, location)
        spec = dict(id=CID, server="server-1", pid=2980836, code_commit=Y99,
            config=task["config"], log=task["log"], package="adares_y99_spatial_mask_skip",
            admission_sha256=ADMISSION, freeze_sha256=FREEZE,
            checkpoint_directory=location["checkpoint_directory"], location_evidence=ref(location_path))
        originals = [ref("scripts/" + name) for name in old_names]
        transform = Specialize(location["checkpoint_directory"], task["log"])
        for name, target in zip(old_names, targets):
            tree = transform.visit(ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8")))
            if name == "y96_terminal_contract.py":
                for n in tree.body:
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPEC" for t in n.targets):
                        n.value = ast.parse(repr(spec), mode="eval").body
            text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
            ast.parse(text, feature_version=(3, 8))
            target.write_text(text, encoding="utf-8", newline="\n")
        assert [ref(p["path"]) for p in originals] == originals
        assert all(ref(p) == r for p, r in before.items()) and git("ls-files", "--stage", "-z") == index
        out = ROOT / ("pipeline/checks/y99_terminal_helpers_" + token + ".json")
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, id=CID, spec=spec, prepared_at=datetime.now().astimezone().isoformat(),
            originals=originals, source_specialization_only=True, checkpoint_loaded=False, archived=False,
            queue_changed=False, remote_mutations=False, evidence=[ref(__file__), ref(source), ref(log), ref(location_path)] + [ref(p) for p in targets]))
        c["evidence"].append(ref(out))
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(helpers=ref(out), original_checkpoint_directory=location["checkpoint_directory"], checkpoint_loaded=False, archived=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True)
    create(p.parse_args().token)
