"""Close verified Y95 archive and Y98 launch with fresh original-run evidence."""
import argparse
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finish_y92_binding_tick import command, SKILL
from finish_y95_preparation_tick import fresh, GOAL, PAUSE
from y98_deployment import Y96, Y98, git
import closure
import gpu_occupation

RUNS=(("server-1","AdaRES-Y96",Y96),("server-2","AdaRES-Y98",Y98))

def evidence(entry):
    return entry["path"]+" SHA"+entry["sha256"]

def record(token):
    lease(token,180)
    writer=monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git("rev-parse","HEAD").decode().strip()==Y98
        before={path:ref(path) for path in ("pipeline/queue.json","pipeline/running.json","pipeline/goal.json","SERVERS.yaml")}
        index=git("ls-files","--stage","-z")
        assert before["pipeline/goal.json"]["sha256"]==GOAL
        assert ref("pipeline/authorizations/server2_gpu1_pause_20261104.json")["sha256"]==PAUSE
        queue=monitor.load_queue(str(ROOT))
        assert monitor.load_running(str(ROOT))=={"server-1":{"0":"AdaRES-Y96"},"server-2":{"0":"AdaRES-Y98"}}
        work=tick.load(ROOT/"pipeline/research_work.json")
        candidates={row["id"]:row for row in work["candidates"]}
        view=tick.load(ROOT/"pipeline/headless/.tick_model_view.json")
        context=tick.load(ROOT/"pipeline/headless/.tick_context.json")
        dispatch=ref("pipeline/checks/adares-y98_dispatch_"+token+".json")
        launch=tick.load(closure.artifact(ROOT,dispatch))
        assert launch["task"]["launch_verified"] is True and launch["task"]["status"]=="running"
        observations,occupations={},{}
        for name,cid,commit in RUNS:
            task=next(row for row in queue["gpu_queues"][name] if row["id"]==cid)
            assert (task["status"],task["launch_verified"],task["code_commit"],str(task["gpu"]))==("running",True,commit,"0")
            assert task["pid"]==(3893460 if cid=="AdaRES-Y96" else launch["task"]["pid"])
            closure.admission(ROOT,task)
            entry=candidates[cid]["latest_training_observation"]
            value=tick.load(closure.artifact(ROOT,entry))
            assert value["token"]==token and value["native_progress_confirmed"] is True
            assert value["pid"]==task["pid"] and value["source_marker"] in (commit,Y98)
            fresh(value,"observed_local")
            observations[cid]=dict(evidence=entry,phase=value["after"]["latest"],pid=task["pid"])
            occupation=ref("pipeline/checks/gpu_occupation_"+name+"_release_"+token+".json")
            fresh(tick.load(closure.artifact(ROOT,occupation)),"observed_at")
            held=next(row for row in work["candidates"] if row.get("stage")=="blocked" and row.get("blocker_kind")=="exclusive_gpu_slot" and row.get("required_server")==name)
            server=next(row for row in context["queue"]["servers"] if row["name"]==name)
            state=gpu_occupation.validated_state(ROOT,context,held,server,queue)
            assert state["busy_gpu_ids"]==state["reserved_gpu_ids"]==[0] and not state["idle_gpu_ids"]
            occupations[name]=occupation
        y95,y96,y98=(candidates["AdaRES-"+name] for name in ("Y95","Y96","Y98"))
        assert y95["scientific_status"]=="fixed30_complete_frozen_review_pending" and y95["stage"]=="blocked"
        terminal=tick.load(closure.artifact(ROOT,y95["terminal_evidence"]))
        assert terminal["token"]==token and terminal["completed_epochs"]=={p:list(range(1,31)) for p in ("Train","Val")}
        assert [row["epoch"] for row in terminal["checkpoints"]]==[30,20]
        assert all(row["model_finite"] and row["optimizer_finite"] and row["optimizer_state_count"]==544 for row in terminal["checkpoints"])
        native=tick.load(closure.artifact(ROOT,y98["native_smoke_result"]))
        assert y98["native_smoke_result"]["path"].endswith("y98_server2_smoke_"+token+".json")
        assert native["losses"]==[4.615511417388916,6.450445175170898]
        receipt=tick.load(closure.artifact(ROOT,ref(y98["admission_receipt"])))
        smoke=tick.load(closure.artifact(ROOT,receipt["native_smoke"]))
        assert smoke["command"]=="python -B scripts/run_y98_bound_native_y96.py --server server-2 --token "+token
        handoff=tick.load(closure.artifact(ROOT,y96["terminal_handoff_plan"]))
        assert handoff["tested_additive_native_evidence"]==y98["native_smoke_result"]
        assert handoff["payload"]["allowed_markers"]==[Y96,Y98]
        assert handoff["CPU_guard_tests_passed"]==30 and handoff["additive_guard_tests_passed"]==8
        failed=handoff["preserved_failed_test"]
        assert "FAILED (failures=2)" in closure.artifact(ROOT,failed).read_text(encoding="utf-8")
        guardlog=ref("pipeline/checks/adares-y98_admission_guards_"+token+".log")
        assert "Ran 11 tests" in closure.artifact(ROOT,guardlog).read_text(encoding="utf-8")
        paused=next(row for row in queue["gpu_queues"]["server-2"] if row["id"]=="AdaRES-Y57")
        assert paused["status"]=="paused" and paused["dispatch_hold"] and paused["resume_requires_user_authorization"]
        selection=ref("pipeline/checks/adares-y98_source_selection_"+token+".json")
        output=ROOT/("pipeline/checks/y98_launch_closeout_"+token+".json")
        assert not output.exists()
        tick.save(output,dict(complete=True,token=token,observed_at=datetime.now().astimezone().isoformat(),
            observations=observations,occupations=occupations,new_scientific_candidates=0,
            scientific_CPU_replays=0,native_GPU_tasks=1,admissions=1,launches=1,archives=1,
            original_Y95_terminal=y95["terminal_evidence"],actual_Y98_native=y98["native_smoke_result"],
            source_selection=selection,dispatch=dispatch,preserved_operational_failure=failed,
            full_original_protocol_preserved=True,formal_FIRST_or_cost_executed=False,
            evidence=[ref(__file__),selection,dispatch,ref(y98["admission_receipt"]),y98["post_training_freeze"],
                y96["terminal_handoff_plan"],guardlog]+[row["evidence"] for row in observations.values()]+list(occupations.values())))
        memory=ROOT/"pipeline/codex_memory.md"
        backup=ROOT/("pipeline/snapshots/codex_memory_before_"+token+".md")
        assert not backup.exists()
        original=memory.read_bytes()
        backup.write_bytes(original)
        old=original.decode("utf-8").splitlines()
        keep=[]
        section=""
        for line in old:
            if line.startswith("## "):
                section=line
            if section in ("## Goal And Phase","## Champion And Baseline","## Closed Or Failed Directions"):
                if line.startswith("- Current token "):
                    line="- Current token "+token+"; snapshot "+view["generated_at"]+"; deadline "+view["deadline_at"]+"; never extend."
                if line.startswith("- Exact previous seven-section memory:"):
                    line="- Exact previous seven-section memory: "+evidence(ref(backup))+"; all historical routes, protocols, receipts, negative controls, bans and dedicated recovery requirements remain byte-preserved; compression never authorizes reopening."
                keep.append(line)
        split=keep.index("## Closed Or Failed Directions")
        lines=["# AdaRES_project Research Memory"]+keep[:split]+[
            "## Active Hypotheses",
            "- Y98 independent W2 full-existing1024 appearance-to-static-mask bypass mask_static=original_x_mask(x)+raw_existing_2D@zeroW1024x256 once; +262144/total164891781/no RNG. Complete original mask MLP/appearance projection/fusion/all channels/all points/seven shared static consumers/queries/SCG/hard gates/loss/selector/backbone/data/seed1999/fixed30 unchanged; not a feature subset/scan/combination or reopening.",
            "- Y98 original CPU evidence/commit03799b0b2cacca31987a604c539a4457498175a1 preserved. NEW same-byte HEADY96/current-original-owner binding "+evidence(y98["additive_version_binding"])+";10 operational guards passed, original native program bodies unchanged, no CPU scientific replay.",
            "- FIRST actual Y98 native server2GPU0/280W PASS "+evidence(y98["native_smoke_result"])+"; losses[4.615511417388916,6.450445175170898], peak_reserved6763315200; two finite TRAIN backward/optimizer steps, four distinct scenes, full1024/base/projection/seven static consumers/shared heads and original gates verified. No efficacy/full-data memory proof.",
            "- Production HEADY98="+Y98+"; exact seven already tested additions selected "+evidence(selection)+"; unrelated index/working files and original Y96 preserved; only server2 safely synchronized.",
            "- Y98 11 actual-native admission guards passed, full COMPLETE ORIGINAL Y95 server2GPU0/280W training/FIRST/cost/subgroup protocol freeze "+evidence(y98["post_training_freeze"])+"; receipt "+evidence(ref(y98["admission_receipt"]))+"; canonical launch "+evidence(dispatch)+" and increasing native counters verified.",
            "- Preserve initial untested draft imports, v2 fixes and failed two-of-eight test alias log "+evidence(failed)+". Corrected alias v3 passes8; original30 lifecycle PASS reused unchanged. Operational test-import specialization only, not native failure, scientific revision, relaxed guard or repeated native.",
            "- Y96 original sampled-seed five-layer query-generator bypass remains independent and unchanged. Original native PASS in411304, losses[4.615510940551758,6.309134006500244], peak_reserved6759120896, no replay. Complete original server1GPU0/350W freeze "+evidence(y96["post_training_freeze"])+"; corrected provenance receipt "+evidence(ref(y96["admission_receipt"]))+"; every original receipt/smoke retained.",
            "- Y99 full32 sparse-input mask-MLP bypass remains CPU-prepared e5151a722778af3eecb28d0d4f6ae75e18a60e0c; no GPU/admission. Needs NEW tested identical-byte HEADY98/current-owner binding before native. No filler candidate, Y97 replay or guard relaxation.",
        ]+keep[split:]+["## Running And Queue",
            "- Current observer=y98_deployment.py observe --id AdaRES-Y96|AdaRES-Y98 --token CURRENT --label NEW. Older observers reject HEADY98. Never live-observe/rearchive Y95/Y94 or earlier archived runs."]
        for name,cid,commit in RUNS:
            row=observations[cid]
            phase=row["phase"]
            lines.append("- "+cid+" original PID"+str(row["pid"])+"/"+name+"GPU0, "+phase["phase"]+str(phase["epoch"])+"/30 step"+str(phase["step"])+"/"+str(phase["count"])+"; original source/YAML/private runtime/power/native counter increase "+evidence(row["evidence"])+"; commit"+commit+".")
        lines += [
            "- Y95 exact original20261109_081912 naturally completed; full Train/Val1..30 stable finite last30/best20/model/optimizer544/scheduler30-20 verified ONCE/canonical archive "+evidence(y95["terminal_evidence"])+"; trainingbest.508540153503418 is NOT formal FIRST. Full original server2/280W review waits onY98; no restart/migration/substitution.",
            "- Y96 exact original directory20261110_165702 (not launch165652) now has30-tested baseline terminal plan and NEW actual-nativeY98 compatibility "+evidence(y96["terminal_handoff_plan"])+". Future finalize_y96_fixed30_y98.py ONLY at HEADY96/Y98 after natural completion, exact stable finite last/best/model/optimizer/scheduler + full Train/Val required. No Y96 checkpoint loaded/archived this tick.",
            "- Y98 future exact terminal plan still needs preparing at HEADY98; do not derive checkpoint directory from launch filename or load live checkpoints. Receipt/native/YAML/source/runtime/280W binding must remain exact.",
            "- Y94 original20261109_111838 verified/archived once, last30/best21/optimizer544/scheduler30-21, trainingbest.5075652599334717 not FIRST. Y93/Y92/Y91 and Y81/Y82/Y83 products and all earlier negative evidence/protocols remain in exact memory backup chain; no terminal replay.",
            "- Fresh canonical occupation "+"; ".join(evidence(value) for value in occupations.values())+"; both authorizedGPU0 busy+reserved. All original GPU1 frozen FIRST/cost remain user-held, never moved/substituted.",
            "- Server1=songcl@192.168.1.175/dl02-MS-7B73, /data1/songcl/Project/AdaRES_project/AdaRES, private runtime/adares_project/bin/python +cuda113lib64, exact ../data/, only two SERVERS mappings. Server2=songcl@192.168.1.163/dl01-MS-7C60, /bak_home/songcl/Project/AdaRES_project/AdaRES, /home/songcl/.conda/envs/adares_project/bin/python, exact /bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/.",
            "## Open Questions And Risks",
            "- No formal FIRST/cost/promotion. Training best, native feasibility and increasing counters do not prove efficacy. Both GPU1 all-use and Y57 separate pause remain; no new user action needed.",
            "- Next tick first observe current Y96/Y98 and dispatch qualified waiting work; do not replay Y95 terminal or Y98 native. Prepare Y98 exact terminal tools and Y99 NEW HEAD/current-owner binding only within snapshot bounds; actual native after matching natural30/full exact checkpoint/canonical archive + fresh exclusive authorized idle/unreserved GPU0. Full original selected-device protocol mandatory.",
            "## Last Tick",
            "- "+token+": expired interrupted2a707 tick verified no helper writer, reported/aborted INCOMPLETE and released only old lease before fresh begin. Generated Y96 tools reused,30 lifecycle guards passed/plan registered without checkpoint loading or scientific replay.",
            "- Y95 exact natural fixed30 proof/canonical archive; Y98 new same-byte binding10 guards, FIRST actual native PASS, actual-evidence Y96 additive binding30+8, exact source selection,11 admission guards/full original server2 freeze/receipt/canonical dispatch and increasing counters verified. All failed import-test evidence retained.",
            "- New scientificIDs0/scientificCPUreplays0/nativeGPUtasks1/admissions1/launches1/archives1/FIRST0/cost0. Goal/old receipts/data/runtime/mappings/seed1999fixed30/350W280W/foreign jobs/bans/GPU1+Y57 pauses preserved. Closeout "+evidence(ref(output))+"; official finish/receipt/release are authoritative.",
        ]
        assert len(lines)<120 and sum(line.startswith("## ") for line in lines)==7
        memory.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
        phases=[]
        for cid,row in observations.items():
            p=row["phase"]
            phases.append(cid+" "+p["phase"]+str(p["epoch"])+"/30 step"+str(p["step"])+"/"+str(p["count"]))
        notes=ROOT/("pipeline/reports/notes_"+token+".json")
        assert not notes.exists()
        tick.save(notes,dict(actions=[
            "仅AdaRES_project，核验owner/Disabled后台/最新暂停后，确认旧tick无写入helper，按原token报告并abort释放超时租约，再fresh begin。先查进展和已有合格任务；Y96原运行递增确认。",
            "完成：Y95原目录20261109_081912的完整Train/Val1..30、稳定有限last30/best20/model/optimizer544/scheduler30-20核验并canonical归档。原产物保留，训练最佳.508540153503418不是正式FIRST。",
            "新准备：复用Y96已生成工具，30项保护测试通过且登记精确目录165702终态计划；未加载在跑checkpoint。Y98同字节新版本/当前owner绑定10项通过，CPU与原生程序体保持。",
            "Y98首次真实server2GPU0/280W原生两步loss4.615511417388916、6.450445175170898，新增完整1024外观/静态mask七次消费者及有限backward/optimizer通过。11项准入检查后完整继承原Y95设备协议、入队并当轮canonical派发。",
            "Y96兼容工具中的继承导入/测试alias错误已保留原稿和失败日志；只修正操作工具，30项原生命周期及8项新兼容检查通过，不放松科学门槛或重跑native。",
            "运行："+"；".join(phases)+"。原PID/source/YAML/收据/私有runtime/350W280W及训练步数递增已核验，物理两张GPU0均busy+reserved。",
            "本轮新科学ID/科学CPU重放/FIRST/成本均0，真实native1/准入1/启动1/归档1。备份七节记忆并用官方report/finish收尾，以实际结果与释放收据为准。"
        ],decisions=[
            "不重放已归档Y95/Y94或已通过Y98 native；Y99和未来工具须新的HEAD/owner绑定，不凑额度新增候选，不重跑Y97或改数值控制。",
            "全部GPU1及原GPU1冻结后评成本继续挂起，Y57独立暂停，禁止server6、跨项目、迁移替代、调度/功耗变更或干预外部任务。原goal、旧签名收据、精确原数据/私有runtime/部署映射和完整FIRST/有害子组/原设备成本协议保持。"
        ],evidence=[ref(output),ref(backup),failed]))
        assert all(ref(path)==entry for path,entry in before.items())
        assert git("ls-files","--stage","-z")==index
        return output,notes,len(lines)
    finally:
        writer.close()

def finish(token):
    lease(token,300)
    for _,cid,_ in RUNS:
        command(["scripts/y98_deployment.py","observe","--id",cid,"--token",token,"--label","release"])
    for name,_,_ in RUNS:
        command([SKILL+"/gpu_occupation.py","AdaRES_project","--token",token,"--server",name,"--label","release"])
    output,notes,lines=record(token)
    command([SKILL+"/foreground_tick.py","AdaRES_project","report","--token",token,"--notes",str(notes)])
    command([SKILL+"/foreground_tick.py","AdaRES_project","finish","--token",token])
    receipt=tick.load(ROOT/("pipeline/tick_receipts/"+token+".json"))
    assert receipt["status"]=="FINALIZED" and receipt["postflight"]=="PASS" and receipt["context_committed"] is True
    assert receipt["lease_release"]=="released" and not (ROOT/"pipeline/headless/.foreground.lockdir").exists()
    assert not (ROOT/"pipeline/headless/.tick_context_state.next.json").exists()
    pending=ROOT/"pipeline/archive_pending"
    assert not pending.exists() or not any(path.is_file() for path in pending.rglob("*"))
    print(json.dumps(dict(closeout=ref(output),status=receipt["status"],postflight=receipt["postflight"],
        context_committed=True,lease_release="released",memory_lines=lines,new_scientific_candidates=0,
        native_GPU_tasks=1,admissions=1,launches=1,archives=1)))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--token",required=True)
    finish(parser.parse_args().token)
