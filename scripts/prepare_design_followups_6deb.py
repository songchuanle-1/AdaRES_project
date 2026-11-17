"""Freeze the next two same-ID diagnostics under the current owned lease."""

import argparse
import ast
from datetime import datetime

import authorize_prepared_candidates as scoped

ROOT = scoped.ROOT
TOKEN = "6debf88fed634a13b772f27bdc244bbd"
SPECS = [
    ("AdaRES-Y57", "y57_full_input_signal",
     "pipeline/analyses/y57_native_task_ef2c7bd97d37400ea1b1a32da2909760_program.py",
     "scripts/probe_y57_full_input_signal_6deb.py"),
    ("AdaRES-Y56", "y56_broader_incidence",
     "pipeline/analyses/y56_native_batch2_222ed2a6e80c43f6b1de2d050be97fa0_program.py",
     "scripts/probe_y56_broader_incidence_6deb.py"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    scoped.lease(args.token)
    lock = scoped.monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = scoped.tick.load(ROOT / "pipeline/research_work.json")
        for cid, name, original, body in SPECS:
            candidate = next(c for c in work["candidates"] if c["id"] == cid)
            assert candidate["stage"] == "diagnosis"
            source = (ROOT / original).read_text(encoding="utf-8")
            tree = ast.parse(source)
            boundary = next(n.lineno for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "records"
                                    for t in n.targets))
            setup = "".join(source.splitlines(keepends=True)[:boundary - 1])
            program = ("import os\nos.environ['OPENBLAS_NUM_THREADS']='4'\n"
                       "os.environ['OMP_NUM_THREADS']='4'\n" + setup + "\n"
                       + (ROOT / body).read_text(encoding="utf-8"))
            ast.parse(program, feature_version=(3, 8))
            compile(program, name, "exec")
            relative = "pipeline/analyses/" + name + "_" + TOKEN
            assert not (ROOT / (relative + "_program.py")).exists()
            (ROOT / (relative + "_program.py")).write_text(
                program, encoding="utf-8", newline="\n")
            plan = dict(
                complete=True, id=cid, token=TOKEN,
                preregistered_at=datetime.now().astimezone().isoformat(),
                source_setup=scoped.ref(original), body=scoped.ref(body),
                program=scoped.ref(relative + "_program.py"),
                same_stable_id=True, result="pending",
                no_validation_tuning=True, no_production_model_or_training=True)
            if cid == "AdaRES-Y57":
                plan.update(
                    sample="Same8TRAIN scenes/32expressions/Q128, final layer6 only; no layer selection.",
                    readouts="Reconstruct all769native normalized head inputs. LOSO quality "
                             "ridge: all inputs+confidence versus plus dispersion; same-capacity "
                             "within-expression cyclic-shift1 dispersion negative control.",
                    fitting="Fixed sum-SSE+1*L2, train-fold-only standardization, unpenalized "
                            "intercept, no clipping/alpha/feature/seed search. All8folds retained.",
                    uncertainty="Paired whole-scene descriptive bootstrap10000/1999; eight "
                                "scene clusters, not4096independent queries.",
                    prediction="Aligned dispersion may improve held-out MSE over full-input "
                               "and shifted controls; null/reverse retained, not an admission threshold.",
                    limits="Only a fixed function class, not conditional independence or causality.")
                for path in (
                    "pipeline/analyses/y57_native_signal_" + TOKEN + ".json",
                    "pipeline/analyses/y57_native_signal_" + TOKEN + ".log"):
                    row = scoped.ref(path)
                    if row not in candidate["evidence"]:
                        candidate["evidence"].append(row)
                candidate["scientific_status"] = (
                    "Confidence-conditional LOSO observational gain .00680041238; "
                    "one negative fold retained. Full-input specificity and causal task deficit unknown.")
            else:
                plan.update(
                    sample="First32TRAIN dataset indices excluding the old8indices, four "
                           "expressions per scene/all7layers/batch2/Q128; freeze before predictions.",
                    readouts="Exact joint query+softmask duplicate excess; query/softmask/hardmask "
                             "uniqueness reported separately. No approximate threshold invented.",
                    prediction="May be zero or positive; no zero population claim, no forced copies.",
                    disposition="If zero in both fixed cohorts, conclude this planned exact-copy "
                                "diagnosis has no observed task incidence, complete diagnostic "
                                "without implementing a correction or rejecting counterfactual algebra.")
            scoped.tick.save(ROOT / (relative + "_prereg.json"), plan)
            candidate["evidence"].extend([
                scoped.ref(body), scoped.ref(relative + "_program.py"),
                scoped.ref(relative + "_prereg.json")])
            candidate["next_action"] = "Execute frozen " + relative + "_program.py using " \
                "run_adares_authorized_probe.py under live authorized idle GPU lease; preserve all outcomes."
        scoped.lease(args.token)
        scoped.tick.save(ROOT / "pipeline/research_work.json", work)
        print("Froze Y57 full-input controls and Y56 disjoint32scene incidence; no admission.")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
