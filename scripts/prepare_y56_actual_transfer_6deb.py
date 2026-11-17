"""Freeze one native case follow-up after, and explicitly conditional on, discovery."""

import argparse
import ast
from datetime import datetime

import authorize_prepared_candidates as scoped

ROOT = scoped.ROOT
TOKEN = "6debf88fed634a13b772f27bdc244bbd"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    scoped.lease(args.token)
    lock = scoped.monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        old = "pipeline/analyses/y56_native_batch2_222ed2a6e80c43f6b1de2d050be97fa0_program.py"
        source = (ROOT / old).read_text(encoding="utf-8")
        tree = ast.parse(source)
        boundary = next(n.lineno for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "records"
                                for t in n.targets))
        body = "scripts/probe_y56_actual_transfer_6deb.py"
        program = ("".join(source.splitlines(keepends=True)[:boundary - 1]) + "\n"
                   + (ROOT / body).read_text(encoding="utf-8"))
        ast.parse(program, feature_version=(3, 8))
        path = "pipeline/analyses/y56_actual_transfer_" + TOKEN + "_program.py"
        assert not (ROOT / path).exists()
        (ROOT / path).write_text(program, encoding="utf-8", newline="\n")
        plan_path = path.replace("_program.py", "_prereg.json")
        plan = dict(
            complete=True, id="AdaRES-Y56", token=TOKEN,
            preregistered_at=datetime.now().astimezone().isoformat(),
            discovery=scoped.ref("pipeline/analyses/y56_broader_unique_" + TOKEN + ".json"),
            setup=scoped.ref(old), body=scoped.ref(body), program=scoped.ref(path),
            sample="Conditional follow-up of discovery pair8, all8expressions/all7layers. "
                   "Fixed focal0. No outcome/large-delta query selection.",
            intervention="Only for a unique focal0 and actual duplicate nonfocal group, "
                         "replace largest-index removable copy with the first different "
                         "nonfocal prototype. Q128/unique joint set/focal/masses/GT unchanged.",
            controls="Exact native trained head repeat and joint query permutation; inverse "
                     "joint-multiplicity reference must retain focal output within dtype bound. "
                     "No-copy and focal-duplicate cases reported not-applicable, never forced.",
            prediction="Native focal confidence may respond while inverse-multiplicity "
                       "reference is invariant. Preserve zero or reversed results.",
            limits="Not a causal prediction-harm or efficacy claim; post-discovery case "
                   "sampling cannot estimate prevalence. No formal correction or admission.")
        scoped.tick.save(ROOT / plan_path, plan)
        work = scoped.tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y56")
        candidate["evidence"].extend([
            plan["discovery"], scoped.ref(path), scoped.ref(plan_path)])
        candidate["next_action"] = "Run frozen native actual-copy fixed-focal transfer; "
        candidate["next_action"] += "do not substitute synthetic or hardmask duplicates for real joint copies."
        scoped.lease(args.token)
        scoped.tick.save(ROOT / "pipeline/research_work.json", work)
        print("Froze actual-copy matched transfer, explicitly conditional on discovery.")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
