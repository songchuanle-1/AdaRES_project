"""Register a falsifiable global-topology diagnosis before reviewing its route."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    assert not any(candidate["id"] == "AdaRES-Y49" for candidate in work["candidates"])
    output = ROOT / ("pipeline/analyses/y49_diagnosis_prereg_c3e5_" + args.token + ".json")
    prereg = dict(
        complete=True, id="AdaRES-Y49",
        hypothesis_id="global-mask-resolvent-coherence",
        preregistered_at=datetime.now().astimezone().isoformat(),
        hypothesis="A mask's connected-component mass/mean summaries and one-hop confidence summaries need not encode global within-mask graph connectivity. Test whether a fixed graph resolvent exposes an explicit, distinguishable confidence information path before considering a model.",
        matched_control="Actual W2/Y36/Y46 confidence readouts on controlled constant-feature masks; hold point mass, component partition/mass and all inherited local summaries fixed where the sources permit. Contrast exact global topology, not a different numerical weight.",
        first_test="Audit the actual W2/Y36/Y46 and relevant closed Y41/Y8 implementations. Construct a pair of connected regular graphs with equal node/component masses and identical constant tokens; compare actual reachable parent summary functions and the fixed unit resolvent trace tr((I+L)^-1)/n. Include joint permutations and repeatability; native geometry incidence is a separate required test.",
        graph_rule="Only inherited exact-nearest-tie graph for native tensors. Combinatorial witnesses must be labelled constructed, not claimed to be realizable native geometry. No new radius/k, graph threshold, diffusion-time sweep, filtering, validation selection or trained benefit claim.",
        interpretation="If actual sources already compute this quantity, or matched controls cannot separate a genuine information path, retain that disconfirmation on this same ID. Information distinction alone is not task diagnosis, utility, novelty, native reachability or admission.",
        possible_followup="Only after source/control support: test prediction-blind frozen TRAIN8 captures and a bounded explicit global-topology readout, then full native controls/backward/resources/independence. Do not clone Y46's component-head optimization or reopen a rejected route.",
        source_evidence=[
            dict(path="pipeline/analyses/Y46_topology_prereg_d179.md",
                 sha256=hashlib.sha256((ROOT / "pipeline/analyses/Y46_topology_prereg_d179.md").read_bytes()).hexdigest()),
            dict(path="pipeline/goal.json",
                 sha256=hashlib.sha256((ROOT / "pipeline/goal.json").read_bytes()).hexdigest())])
    with output.open("x", encoding="utf-8") as stream:
        json.dump(prereg, stream, indent=2)
    reference = dict(path=output.relative_to(ROOT).as_posix(),
                     sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    work["candidates"].append(dict(
        id="AdaRES-Y49", stage="diagnosis",
        hypothesis_id=prereg["hypothesis_id"], hypothesis=prereg["hypothesis"],
        matched_control=prereg["matched_control"],
        next_action=prereg["first_test"], evidence=[reference]))
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(complete=True, id="AdaRES-Y49", stage="diagnosis",
                         preregistered_at=prereg["preregistered_at"], evidence=reference)))
finally:
    writer.close()
