"""Preregister and execute one bounded independent CPU reserve diagnosis."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import authorize_prepared_candidates as scoped
import closure
import foreground_tick as tick
import pipeline_monitor as monitor

CHINA = timezone(timedelta(hours=8))
BODY = "scripts/probe_y56_competitor_density_cpu.py"
ID = "AdaRES-Y56"


def paths(token):
    return {key: f"pipeline/analyses/y56_density_{key}_{token}.{suffix}"
            for key, suffix in (("prereg", "json"), ("program", "py"),
                                ("execution", "json"), ("result", "json"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("operation", choices=("prepare", "run"))
    args = parser.parse_args()
    scoped.lease(args.token)
    lock = monitor.acquire_lock(str(ROOT))
    if lock is None:
        raise ValueError("Project writer is busy")
    try:
        scoped.lease(args.token)
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        p = paths(args.token)
        work = tick.load(ROOT / "pipeline/research_work.json")
        queue = monitor.load_queue(str(ROOT))
        resource = f"pipeline/analyses/storage_regression_{args.token}.json"
        resource_ref = scoped.ref(resource)
        if args.operation == "prepare":
            if (not view["research"]["reserve_only"] or view["research"]["reserve_budget"] < 1
                    or view["queue"]["candidate_budget"] != 0
                    or any(c["id"] == ID for c in work["candidates"])):
                raise ValueError("A fresh bounded reserve ID is not authorized")
            source_path = "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/dec.py"
            source = (ROOT / source_path).read_text(encoding="utf-8")
            committed = scoped.git("show", closure.git(ROOT, "rev-parse", "HEAD")+":"+source_path)
            if (ROOT/source_path).read_bytes().replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
                raise ValueError("Parent source is not the committed source")
            node = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.ClassDef) and n.name == "QueryCompetitionIndicator")
            class_source = ast.get_source_segment(source, node)
            now = datetime.now(CHINA)
            hypothesis = ("For a fixed focal query and a fixed unique set of mask/query competitors, "
                "changing only how many exact copies represent each non-focal competitor may "
                "change native confidence through overlap-row-normalized context. Diagnose "
                "counterfactual multiplicity nuisance before proposing a model correction.")
            control = ("Replay the exact committed native256-d indicator class on CPU with128 "
                "queries, fixed seed1999 and fixed masks/point masses. Compare counts[1,63,64] "
                "versus[1,64,63], keeping the focal query and unique competitor set identical. "
                "Use repeat/row-permutation/identical-competitor-feature negative controls; "
                "an inverse-multiplicity fixture reference must reproduce the singleton focal "
                "output of the same native head on three unique prototypes. Use float32 and "
                "float64 and dtype-derived arithmetic bounds, never a task-gain threshold.")
            review = f"pipeline/reviews/motivation_review_Y56_{args.token}.md"
            design = f"pipeline/designs/exp_design_Y56_{args.token}.md"
            goal = tick.load(ROOT/"pipeline/goal.json")
            (ROOT/review).write_text(
                "# Y56 Motivation Review\n\n## 1. Goal\n```json\n"+
                json.dumps(goal,ensure_ascii=False,indent=2)+"\n```\nChampion CSV Y36 .5110, baseline B1 .5020, target .5150; no new metric.\n"
                "## 2. Story Status\n| Hypothesis | Status | Evidence | Gap |\n|---|---|---|---|\n"
                "| Y52/Y54 mechanisms | admitted, not proven effective | existing admission receipts | trained comparison; storage restored |\n"
                "| Y56 competitor multiplicity | untested | native overlap-row normalization | frozen CPU controls and actual TRAIN incidence |\n"
                "Current storage regression stops deployment/recovery.\n"
                "## 3. Experiment Assessment\n| Experiment | mIoU | Motivation | Contribution |\n|---|---|---|---|\n"
                "| Y48 | not remeasured | ownership targets for mask-equivalent queries | rejected; do not reopen |\n"
                "| Y52/Y54 | unknown | score targets / fine spatial support | admitted, efficacy pending |\n"
                "This test holds supervision/selector unchanged and isolates the multiplicity of other competitors in native context aggregation. Y52 target, Y54 spatial-mask and Y55 expansion mechanisms are untouched.\n"
                "## 4. Gap Analysis\nPairwise overlap context normalizes over rows, so representation multiplicity is a distinct possible nuisance. Actual TRAIN incidence and trained-head significance are unknown.\n"
                "## 5. Design Rationale\n"+hypothesis+"\n"+control+
                "\nCPU component diagnosis only; no novelty or trained-benefit claim, no literature search represented as completed.\n"
                "## 6. Next Batch\n"+design+"; prepare only this stable ID under reserve_budget1, no queue entry or model implementation.\n",
                encoding="utf-8")
            (ROOT/design).write_text(
                "# Y56 Competitor Multiplicity Diagnosis\n\n## 1. Motivation\n"+goal["description"]+
                "\nReserve-only preparation, admission budget0.\n## 2. Current Best Results\n"
                "| ID | Config | State | mIoU | Acc50 | spmIoU | Epochs |\n|---|---|---|---|---|---|---|\n"
                "| Y36 | not loaded this tick | historical complete | .5110 | unknown | unknown | unknown |\n"
                "| B1 | not loaded this tick | historical baseline | .5020 | unknown | unknown | unknown |\n"
                "## 3. Analysis And Direction\nTest native context sensitivity, not Y48 target changes or scalar tuning. Task efficacy is not inferred from fixtures.\n"
                "## 4. Experiment Plan\n| # | ID | Config | Hypothesis | Readout | Priority | Dependency |\n|---|---|---|---|---|---|---|\n"
                "| 1 | Y56 | none; component diagnosis | competitor-row multiplicity nuisance | focal context/logits vs repeat/permutation/numerical controls | reserve | exact source, CPU Torch |\n"+
                control+"\n## 5. GPU Allocation Plan\n| Server | GPU | Assignment | Config | Justification |\n|---|---|---|---|---|\n"
                "| server-2 | none | CPU component diagnosis only | none | existing project Python; CUDA hidden, one Torch thread |\nNo GPU computation or training launch.\n"
                "## 6. Implementation Checklist\nPreregister source/program/fixtures, execute unchanged native class, retain actual result, then require actual TRAIN duplicate incidence and trained-parent native evidence before any formal variant or gate. No commit, queue, model or checkpoint write.\n"
                "## 7. Risk Assessment\nExact synthetic duplicates may not occur in actual decoder output. Random-head fixture effects are not trained benefit. Source is local committed replay, not verification of an unavailable mounted remote source. Preserve negatives and leave full native/resource/independence gates pending.\n"
                "## 8. Smell Check\nNo weight/threshold/temperature/feature/epoch scan. Fixed representation intervention, not a new name for Y48 ownership supervision. A future model would require real task incidence and independent implementation; none is admitted here.\nStatus: DRAFT, diagnosis only.\n",
                encoding="utf-8")
            payload = dict(class_source=class_source, id=ID, token=args.token)
            program = "PAYLOAD = "+repr(payload)+"\n"+(ROOT/BODY).read_text(encoding="utf-8")
            ast.parse(program, feature_version=(3,8))
            (ROOT/p["program"]).write_text(program,encoding="utf-8",newline="\n")
            plan = dict(complete=True,id=ID,token=args.token,preregistered_at=now.isoformat(),
                hypothesis_id="competitor_representation_multiplicity_context_nuisance",
                hypothesis=hypothesis,matched_control=control,source=scoped.ref(source_path),
                class_sha256=hashlib.sha256(class_source.encode()).hexdigest(),
                code_commit=closure.git(ROOT,"rev-parse","HEAD"),goal=scoped.ref("pipeline/goal.json"),
                program=scoped.ref(p["program"]),body=scoped.ref(BODY),
                expected=("Focal context/logit differences above fixed arithmetic/repeat controls "
                    "under multiplicity-only change, absent with identical competitor features; "
                    "inverse-count focal reference agrees with unique prototypes. No efficacy claim."),
                native_full_gates="not executed",actual_train_incidence="unknown",status="pending")
            tick.save(ROOT/p["prereg"],plan)
            work["candidates"].append(dict(id=ID,stage="diagnosis",created_at=now.isoformat(),
                hypothesis_id=plan["hypothesis_id"],hypothesis=hypothesis,matched_control=control,
                next_action="Run the frozen CPU native-indicator multiplicity controls now; no GPU or model mutation.",
                failure_scope="inconclusive",evidence=[scoped.ref(p["prereg"]),scoped.ref(p["program"]),
                    scoped.ref(review),scoped.ref(design),resource_ref],
                gate_summary=dict(diagnosis="pending CPU component controls",native_backward="not run full model",
                    resources="CPU only; no full resource gate",independence="no formal model implemented",constraints="reserve only, no admission")))
            tick.save(ROOT/"pipeline/research_work.json",work)
            print(json.dumps(dict(id=ID,stage="diagnosis",preregistration=scoped.ref(p["prereg"]),
                                  program=scoped.ref(p["program"]),new_candidates=1),indent=2))
            return
        plan = tick.load(ROOT/p["prereg"])
        if scoped.ref(p["program"]) != plan["program"] or scoped.ref(BODY) != plan["body"]:
            raise ValueError("Frozen program changed")
        if (ROOT/p["execution"]).exists() or (ROOT/p["result"]).exists():
            raise ValueError("Do not replay this frozen test")
        candidate = next(c for c in work["candidates"] if c["id"]==ID)
        command=["ssh","-T","-o","BatchMode=yes","-o","PasswordAuthentication=no",
            "-o","KbdInteractiveAuthentication=no","-o","StrictHostKeyChecking=yes",
            "-o","ClearAllForwardings=yes","-o","ConnectTimeout=8","songcl@192.168.1.163",
            "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages /home/songcl/.conda/envs/adares_project/bin/python -B -S -"]
        started=datetime.now(CHINA).isoformat()
        try:
            result=subprocess.run(command,input=(ROOT/p["program"]).read_text(encoding="utf-8"),
                capture_output=True,encoding="utf-8",errors="replace",timeout=120)
        except (subprocess.TimeoutExpired, OSError) as error:
            def output_text(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
            tick.save(ROOT/p["execution"],dict(complete=False,observed_from=started,
                observed_until=datetime.now(CHINA).isoformat(),command=command,
                returncode=None,error_type=type(error).__name__,error=str(error),
                stdout=output_text(getattr(error,"stdout",None)),
                stderr=output_text(getattr(error,"stderr",None)),program=plan["program"]))
            candidate["evidence"].append(scoped.ref(p["execution"]))
            tick.save(ROOT/"pipeline/research_work.json",work)
            raise ValueError("CPU diagnosis did not finish; execution evidence preserved") from error
        ended=datetime.now(CHINA)
        tick.save(ROOT/p["execution"],dict(complete=True,observed_from=started,
            observed_until=ended.isoformat(),command=command,returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr,program=plan["program"]))
        if result.returncode:
            candidate["evidence"].append(scoped.ref(p["execution"]))
            tick.save(ROOT/"pipeline/research_work.json",work)
            raise ValueError("CPU diagnosis failed; execution evidence preserved, do not relabel PASS")
        measured=json.loads(result.stdout)
        tick.save(ROOT/p["result"],measured)
        candidate["evidence"].extend([scoped.ref(p["execution"]),scoped.ref(p["result"])])
        candidate.update(stage="blocked",blocker_kind="external_resource",required_server="server-2",
            blocker="CPU component controls executed, but actual TRAIN duplicate incidence, trained-parent diagnostic weights and full native/data/resource gates require the unavailable original /bak_home mount. No formal model has been implemented or admitted.",
            retry_after=(ended+timedelta(hours=1)).isoformat(),
            unblock_condition="Original server-2 source/data/checkpoints restored and validated; measure actual fixed TRAIN query multiplicity and trained-parent focal effects before a formal implementation or admission.",
            next_action="Continue SAME Y56: first capture actual TRAIN duplicate-query incidence and source-bound trained-parent controls when original data volume is available. Do not claim fixture signal is efficacy or reopen Y48 targets.",
            gate_summary=dict(diagnosis="CPU component only; actual TRAIN incidence unknown",
                controls="fixed native128-query multiplicity/repeat/permutation/negative/unique-reference controls executed",
                native_backward="CPU indicator gradients only, not full native TRAIN",
                resources="no GPU; full latency/memory unmeasured",independence="not implemented",
                constraints="reserve only; no queue/admission/model changes"))
        for current in work["candidates"]:
            if current["id"] not in ("AdaRES-Y52","AdaRES-Y54"):
                continue
            current["evidence"].append(resource_ref)
            if current.get("blocker"):
                previous={k:current[k] for k in ("blocker_kind","blocker","retry_after","unblock_condition") if k in current}
                previous.update(superseded_at=ended.isoformat(),superseded_by=resource_ref)
                current.setdefault("blocker_history",[]).append(previous)
            current.update(stage="blocked",blocker_kind="external_resource",required_server="server-2",
                blocker="SSH recovered but /bak_home is again unmounted; configured source/data/checkpoints and source marker unavailable. Original Y52 PID15934 is confirmed absent. Do not infer a metric result or blindly restart.",
                retry_after=(ended+timedelta(hours=1)).isoformat(),
                unblock_condition=tick.load(ROOT/resource)["required_action"],
                next_action=tick.load(ROOT/resource)["required_action"])
        work["design_cycle"]=dict(stop_reason="external_resource_blocked",attempted_ids=["AdaRES-Y54",ID],
            evidence=[resource_ref,scoped.ref(p["result"])],
            resume_condition="Restore original /bak_home, reconcile Y52 log/checkpoints and canonical terminal state, then continue Y54 and the SAME Y56 native data diagnosis without blind restarts.",
            clarification="One independent reserve CPU test executed; no route-exhaustion or trained-benefit claim.")
        closure.validate_work(ROOT,tick.load(ROOT/"pipeline/headless/.tick_context.json"),work["candidates"],queue)
        scoped.lease(args.token)
        tick.save(ROOT/"pipeline/research_work.json",work)
        decision=monitor.log_decision(str(ROOT),"other","Storage regression and available CPU reserve budget1.",
            "Preserve Y52/Y54 queue and evidence; record missing original PID/mount; execute independent Y56 native-indicator CPU controls only.",
            "No new training/admission or model update; fixture signal needs actual TRAIN incidence and original assets.",p["result"]+"; "+resource)
        print(json.dumps(dict(id=ID,stage=candidate["stage"],result=scoped.ref(p["result"]),
            execution=scoped.ref(p["execution"]),signal=measured["structural_fixture_signal"],
            decision=decision,new_candidates=1,component_cpu_tests=1,training_launches=0),indent=2))
    finally:
        lock.close()


if __name__=="__main__":
    main()
