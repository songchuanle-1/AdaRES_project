"""Lease-bound structured evidence persistence for the continuing foreground tick."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import foreground_tick as tick
import pipeline_monitor as monitor

TOKEN = "1cb15474bfd346dcbc34837f3e9b79d3"
parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=("register", "freeze", "bind"))
parser.add_argument("--id")
parser.add_argument("--stage")
parser.add_argument("--next-action")
parser.add_argument("--paths", nargs="*", default=[])
args = parser.parse_args()
owner = tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])


def ref(relative):
    path = (ROOT / relative).resolve()
    assert ROOT in path.parents and path.is_file()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    path = ROOT / "pipeline/research_work.json"
    work = tick.load(path)
    if args.operation == "register":
        assert not any(c["id"] == "AdaRES-Y50" for c in work["candidates"])
        capture = "pipeline/analyses/y49_train24_capture_v2_1cb1_" + TOKEN + ".json"
        diagnosis = "pipeline/analyses/y49_train24_diagnosis_1cb1_" + TOKEN + ".json"
        y49 = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y49")
        y49["evidence"].extend(ref(p) for p in (capture, diagnosis, "scripts/diagnose_y49_train24_1cb1.py", "pipeline/analyses/y49_capture_compile_failure_1cb1.md"))
        y49["next_action"] = "Final substantive review of current realization: TRAIN24/96 has0 witnesses, all6314subgraphs forest, original runtime spikes retained. Do not preregister an unsupported cyclic-fallback revision."
        relative = "pipeline/analyses/y50_diagnosis_prereg_1cb1_" + TOKEN + ".json"
        record = dict(complete=True, id="AdaRES-Y50", preregistered_at=datetime.now().astimezone().isoformat(),
                      hypothesis="The W2 indicator explicitly retains weighted pairwise mask overlap, but need not retain higher-order joint spatial incidence of three or more candidate masks. A per-query conditional second moment of centered competitor coverage may expose that information without changing masks, selector or target.",
                      matched_control="Full128 parity fixtures: first3 rows use even {000,011,101,110} versus odd {111,100,010,001}, probabilities .25/.75, remaining125rows .5, same point masses, SAME query features/head/seed. Areas and full weighted pairwise Gram/IoU exactly match. Compare actual W2 indicator, not an invented parent.",
                      descriptor="For each q, centered competitor coverage h_q(x)=sum_{r!=q}(p_r(x)-weighted_mean(p_r))/(Q-1). Return E_{mass*p_q}[h_q^2] and E_{mass*(1-p_q)}[h_q^2]. Soft sigmoid probabilities, no threshold/temperature/weight sweep.",
                      controls=["32fixed seeds1999..2030, actual W2 indicator float64 on full128fixtures; original8eps*nodes*Q envelope, no fitted tolerance.",
                                "Joint point permutation invariant; query permutation equivariant; padded points ignored; empty denominator clamp matches parent.",
                                "Frozen native TRAIN8/32 plus newTRAIN24/96 inputs, all expressions, no result-based subset selection. Diagnose ranges and point-mass IoU associations descriptively, not efficacy or parent blindness.",
                                "If physical implementation follows: independent copied native W2 parent, released init, same loss/config/seed/fixed30/128queries/single selector; exact disabled-parent and RNG controls; native loss gradients and resource tests before admission."],
                      failure_rules=["If actual W2 distinguishes area/Gram-matched fixtures above original numerical envelope, reject the missing-indicator-information premise for this realization.",
                                     "If proposed statistic does not distinguish matched fixtures above same envelope, reject proposed statistic, not all higher-order mechanisms.",
                                     "Native constancy or unexplained harmful/reversed task-relevant evidence is not PASS; investigate before admission."],
                      independence="Not Y47 pairwise missing-overlap; not Y29 ranking loss, Y40 word/support alignment or Y49 single-mask topology. All historical closures retained.",
                      reserve_budget_used=1, no_validation_selection=True, no_admission=True)
        assert not (ROOT / relative).exists()
        tick.save(ROOT / relative, record)
        work["candidates"].append(dict(id="AdaRES-Y50", stage="diagnosis", hypothesis=record["hypothesis"],
                                       matched_control=record["matched_control"],
                                       next_action="Execute full128 equal-area/Gram parity test against actual native W2 indicator, then frozen native TRAIN descriptor controls; no training before all gates.",
                                       evidence=[ref(relative)], failure_scope="inconclusive"))
        work.pop("design_cycle", None)
        tick.save(path, work)
        print(relative)
    elif args.operation == "freeze":
        extension = ref("scripts/y50_joint_incidence_1cb1.py")
        program = ref("scripts/probe_y50_diagnosis_1cb1.py")
        for evidence in (extension, program):
            compile((ROOT / evidence["path"]).read_text(), evidence["path"], "exec")
        relative = "pipeline/analyses/y50_source_freeze_1cb1.json"
        assert not (ROOT / relative).exists()
        tick.save(ROOT / relative, dict(complete=True, id="AdaRES-Y50", preregistered_at=datetime.now().astimezone().isoformat(),
                                       source_sha256=extension["sha256"], source=extension, program=program,
                                       protocol="SAME registered parity128/32seeds and frozen native128 expressions; no result-based changes."))
        candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y50")
        candidate["evidence"].extend([extension, program, ref(relative)])
        tick.save(path, work)
        print(relative)
    else:
        candidate = next(c for c in work["candidates"] if c["id"] == args.id)
        if args.stage:
            assert candidate["stage"] not in ("rejected", "closed", "complete")
            candidate["stage"] = args.stage
        if args.next_action:
            candidate["next_action"] = args.next_action
        for evidence in map(ref, args.paths):
            if evidence not in candidate["evidence"]:
                candidate["evidence"].append(evidence)
        tick.save(path, work)
        print(candidate["id"], candidate["stage"], len(candidate["evidence"]))
finally:
    lock.close()
