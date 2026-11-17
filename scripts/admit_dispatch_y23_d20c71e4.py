"""Hash reviewed evidence and invoke only the canonical gated dispatcher."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
TOKEN="d20c71e418354a6b818d9f37ab72fc7b"
owner=json.loads((ROOT/"pipeline/headless/.foreground.lockdir/owner.json").read_text(encoding="utf-8"))
assert owner["token"]==TOKEN and owner["thread_id"]==os.environ["CODEX_THREAD_ID"]
sys.path.insert(0,"C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import closure
import pipeline_monitor as monitor

def evidence(path):
    return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())

commit=closure.git(ROOT,"rev-parse","HEAD")
assert commit.startswith("4a24410")
bundle=json.loads((ROOT/"pipeline/analyses/y23_source_bundle_d20c71e4.json").read_text())
for item in bundle["files"]:
    blob=subprocess.check_output(["git","-C",str(ROOT),"show",commit+":"+item["path"]])
    assert hashlib.sha256(blob).hexdigest()==item["sha256"]
    assert evidence(item["path"])==item
receipt_path=ROOT/"pipeline/admissions/AdaRES-Y23.json"
assert not receipt_path.exists()
common="pipeline/analyses/Y23_admission_review_d20c71e4.md"
names={
    "diagnosis":["pipeline/analyses/y23_population_d20c71e4.json"],
    "controls":["pipeline/analyses/y23_contracts_d20c71e4.json",
        "pipeline/analyses/y23_population_d20c71e4.json",
        "pipeline/analyses/y23_inference_d20c71e4.json",
        "pipeline/analyses/y23_repeat_audit_d20c71e4.json",
        "pipeline/analyses/y23_scene_repeat_d20c71e4.json"],
    "native_backward":["pipeline/analyses/y23_backward_d20c71e4.json",
        "pipeline/analyses/y23_integration_d20c71e4.json"],
    "resources":["pipeline/analyses/y23_stress_d20c71e4.json",
        "pipeline/analyses/y23_parent_stress_d20c71e4.json",
        "pipeline/analyses/y23_inference_profile_d20c71e4.json"],
    "independence":["pipeline/analyses/y23_source_bundle_d20c71e4.json"],
    "constraints":[common,"pipeline/designs/exp_design_Y23_d20c71e4.md",
        "pipeline/reviews/motivation_review_d20c71e4.md"],
}
receipt=dict(id="AdaRES-Y23",config="configs_song/model_paper5/adares_y23_valid_token_support_res.yaml",
    code_commit=commit,goal_sha256=evidence("pipeline/goal.json")["sha256"],
    decision=evidence("pipeline/decisions/20261002-112412_next_batch.md"),
    gates={name:dict(passed=True,evidence=[evidence(p) for p in paths]) for name,paths in names.items()},
    scientific_status="admitted_for_falsifiable_training_not_efficacy",
    frozen_protocol=dict(epochs=30,seed=1999,gpu="0",server="server-2"),
    limitations="Inherited native FPS/sample repeat instability and resume numerical noise retained; no efficacy, universal determinism, p95 latency, trained Pareto or novelty claim.")
receipt_path.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
task=dict(id="AdaRES-Y23",label="Y23 valid-token semantic support",
    config=receipt["config"],status="waiting",depends_on=[],code_commit=commit,
    admission_receipt="pipeline/admissions/AdaRES-Y23.json",requires_exclusive_ownership=True,
    min_epochs=30,patience=30,total_epochs=30)
closure.admission(ROOT,task)
lock=monitor.acquire_lock(str(ROOT))
if lock is None:
    raise RuntimeError("Canonical monitor lock unavailable; receipt retained, no queue mutation")
try:
    queue=monitor.load_queue(str(ROOT))
    assert not any(t["id"]=="AdaRES-Y23" for g in queue["gpu_queues"].values() for t in g)
    active=[t for t in queue["gpu_queues"]["server-2"] if t["status"] in ("running","waiting","claiming")]
    assert len(active)==1 and active[0]["id"]=="AdaRES-Y18" and active[0]["status"]=="running"
    server=monitor.load_servers(str(ROOT))["server-2"]
    assert server["host"]=="192.168.1.163" and server["allowed_gpus"]==[0,1]
    processes=monitor.find_running_processes(server["host"],server["user"],server["remote_code_root"])
    used={str(p[2]) for p in processes.values()}
    assert "0" not in used and "1" in used
    queue["gpu_queues"]["server-2"].append(task)
    monitor.save_queue(str(ROOT),queue)
    events=[]
    monitor._process_queue(queue,"server-2",server["host"],server["user"],str(ROOT),
        used,{"0","1"},events,max_tasks_per_gpu=1,remote_code_root=server["remote_code_root"],
        current_tasks=len(processes),max_tasks_total=2)
    current=monitor.load_queue(str(ROOT))
    selected=next(t for t in current["gpu_queues"]["server-2"] if t["id"]=="AdaRES-Y23")
    result=dict(observed_at_local=datetime.now().astimezone().isoformat(),events=events,
        task=selected,receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        scope="Canonical gated dispatch only; no monitor/goal preflight repeated")
    out=ROOT/"pipeline/analyses/y23_dispatch_d20c71e4.json"
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
finally:
    lock.close()
