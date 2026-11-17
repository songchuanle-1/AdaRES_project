"""Prepare read-only terminal/evaluation bindings without modifying frozen plans."""

import argparse
import ast
from datetime import datetime
import hashlib
import json

import authorize_prepared_candidates as scoped

ROOT = scoped.ROOT
TOKEN = "6debf88fed634a13b772f27bdc244bbd"
COMMIT = "8943c8c460501b200a8dea958964579ffea65f76"
FREEZES = {
    "AdaRES-Y52": "pipeline/analyses/AdaRES-Y52_post_training_freeze_v2_e4e47d5857d44da683b1d9fed03bfe66.json",
    "AdaRES-Y54": "pipeline/analyses/y54_post_training_freeze_fde01739a9864bae9bb342a577235e66.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("terminal", "evaluations"))
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    scoped.lease(args.token)
    lock = scoped.monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = scoped.tick.load(ROOT / "pipeline/research_work.json")
        candidates = {c["id"]: c for c in work["candidates"]}
        for cid, path in FREEZES.items():
            assert scoped.ref(path) == candidates[cid]["post_training_freeze"]
            freeze = scoped.tick.load(ROOT / path)
            assert freeze["protocol"]["primary"] == "FIRST full evaluation mIoU"
            assert freeze["evaluator"]["sha256"] == scoped.ref(
                "AdaRES/tools/evaluate_adares_frozen_full.py")["sha256"]
        if args.operation == "terminal":
            source = (ROOT / "pipeline/analyses/y57_native_task_ef2c7bd97d37400ea1b1a32da2909760_program.py").read_text(encoding="utf-8")
            assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "FROZEN"
                                      for t in n.targets))
            frozen = ast.literal_eval(assignment.value)
            assert frozen["commit"] == COMMIT
            sources = {r["path"]: r for r in frozen["sources"]}
            methods = {}
            for cid in FREEZES:
                experiment = scoped.tick.load(ROOT / ("pipeline/experiments/" + cid + ".json"))
                assert experiment["status"] == "done_complete" and experiment["epochs_final"] == "30/30"
                assert experiment["code_commit"] == COMMIT
                directory = experiment["resumed_work_dir"]
                methods[cid] = dict(
                    config=experiment["config"], checkpoint=directory + "/best.pth",
                    last=directory + "/last.pth",
                    config_sha256=hashlib.sha256((ROOT / "AdaRES" / experiment["config"]).read_bytes()).hexdigest())
            freeze = scoped.tick.load(ROOT / FREEZES["AdaRES-Y54"])
            for identity, value in freeze["comparators"].items():
                methods[identity] = dict(
                    config=value["config"], checkpoint=value["path"],
                    checkpoint_sha256=value["sha256"],
                    config_sha256=hashlib.sha256((ROOT / "AdaRES" / value["config"]).read_bytes()).hexdigest())
            packages = {
                "AdaRES-Y52": "adares_y52_point_target",
                "AdaRES-Y54": "adares_y54_fine_support",
                "Y36": "adares_y36_local_adjacency",
                "W2": "adares_w2_pointmass_competition",
                "B1": "adares_b1_standard",
            }
            for identity, package in packages.items():
                for path in sorted((ROOT / "AdaRES/ipdn/model_paper5" / package).glob("*.py")):
                    relative = path.relative_to(ROOT / "AdaRES").as_posix()
                    sources[relative] = dict(
                        path=relative, lf_sha256=hashlib.sha256(path.read_bytes().replace(
                            b"\r\n", b"\n")).hexdigest())
            historical = scoped.tick.load(ROOT / freeze["comparators"]["Y36"]["source_manifest"]["path"])
            for row in historical["files"]:
                relative = row["path"].removeprefix("AdaRES/")
                if relative.endswith(".py"):
                    assert sources[relative]["lf_sha256"] == row["sha256"], relative
            bindings = dict(commit=COMMIT, methods=methods, sources=list(sources.values()))
            body = "scripts/probe_terminal_bindings_6deb.py"
            program = "BINDINGS=" + repr(bindings) + "\n" + (ROOT / body).read_text(encoding="utf-8")
            path = "pipeline/analyses/terminal_bindings_" + TOKEN + "_program.py"
            assert not (ROOT / path).exists()
            ast.parse(program, feature_version=(3, 8))
            (ROOT / path).write_text(program, encoding="utf-8", newline="\n")
            plan = dict(
                complete=True, token=TOKEN, prepared_at=datetime.now().astimezone().isoformat(),
                freezes={cid: scoped.ref(value) for cid, value in FREEZES.items()},
                program=scoped.ref(path), body=scoped.ref(body),
                operation="Read-only best/last full-SHA metadata/source/no-process inspection, "
                          "then unchanged frozen evaluation. No training or new selection.")
            scoped.tick.save(ROOT / ("pipeline/analyses/terminal_bindings_" + TOKEN + "_prereg.json"), plan)
        else:
            terminal_path = "pipeline/analyses/terminal_bindings_v3_" + TOKEN + ".json"
            terminal = scoped.tick.load(ROOT / terminal_path)
            assert terminal["complete"] and terminal["code_commit"] == COMMIT
            source = (ROOT / "scripts/probe_y48_frozen_full_12be.py").read_text(encoding="utf-8")
            adapter = ast.literal_eval(next(n.value for n in ast.parse(source).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "adapter"
                                                    for t in n.targets)))
            programs = []
            for identity in ("AdaRES-Y52", "AdaRES-Y54", "W2", "B1", "Y36"):
                payload = dict(identity=identity, terminal_sha256=scoped.ref(terminal_path)["sha256"],
                               evaluator_sha256=scoped.ref("AdaRES/tools/evaluate_adares_frozen_full.py")["sha256"],
                               adapter=adapter)
                body = "scripts/probe_terminal_frozen_eval_6deb.py"
                program = "EVAL=" + repr(payload) + "\n" + (ROOT / body).read_text(encoding="utf-8")
                path = "pipeline/analyses/frozen_" + identity.lower().replace("-", "_") + "_" + TOKEN + "_program.py"
                assert not (ROOT / path).exists()
                ast.parse(program, feature_version=(3, 8))
                (ROOT / path).write_text(program, encoding="utf-8", newline="\n")
                programs.append(scoped.ref(path))
            plan_path = "pipeline/analyses/frozen_terminal_eval_" + TOKEN + "_bindings.json"
            plan = dict(
                complete=True, token=TOKEN, prepared_at=datetime.now().astimezone().isoformat(),
                freezes={cid: scoped.ref(value) for cid, value in FREEZES.items()},
                terminal=scoped.ref(terminal_path), programs=programs,
                body=scoped.ref(body), adapter_sha256=hashlib.sha256(adapter.encode()).hexdigest(),
                protocol="All9508expressions/141scenes twice per method; FIRST primary, "
                         "scene-bootstrap10000/1999 and TRAIN cuts1487/3325/harm rules unchanged.",
                comparative_result="pending")
            scoped.tick.save(ROOT / plan_path, plan)
            for cid in FREEZES:
                candidate = candidates[cid]
                assert terminal["methods"][cid]["fixed30_finished"]
                candidate.update(
                    stage="validation", scientific_status="fixed30_finished_frozen_evaluation_pending",
                    next_action="Execute same frozen full two-pass evaluation for Y52/Y54/W2/B1/Y36; "
                                "FIRST primary, then paired scene-clustered/subgroup/Pareto review. "
                                "No resume, new checkpoint selection, threshold or epoch tuning.")
                candidate["evidence"].extend([
                    scoped.ref(terminal_path), scoped.ref(plan_path), *programs])
        scoped.lease(args.token)
        scoped.tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps({k: v for k, v in plan.items() if k != "programs"}, indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
