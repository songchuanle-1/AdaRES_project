"""Persist real Y49 CPU progress and a live, externally occupied GPU stop."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
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


def ref(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def result(stem):
    path = "pipeline/analyses/" + stem + "_" + args.token + ".json"
    value = json.loads((ROOT / path).read_text())
    assert value["complete"]
    return ref(path), value


source_ref, source = result("y49_source_controls_c3e5")
capture_ref, capture = result("y49_native_capture_cpu_c3e5")
hook_ref, hook = result("y49_hook_cpu_c3e5")
resource_ref, resource = result("y48_resources_c3e5")
assert source["CPU_cases"] == hook["CPU_cases"] == capture["native_captured_expressions"] == 32
assert hook["new_tensors"] == 4 and hook["prototype_parameters"] == 66561
assert not hook["native_backward_executed"]
assert capture["wrong_correspondence_separated_expressions"] == 32
assert capture["expressions_with_distinct_global_values_at_equal_component_masses"] == 0
assert not resource["free_authorized_gpus"]
assert all(row["pids"] for row in resource["second"]["GPUs"])
old = "601bcda714184133920945f8f067b382"
old_bundle = json.loads((ROOT / (
    "pipeline/analyses/y48_local_bundle_601b_" + old + ".json")).read_text())
assert resource["source_hashes"] == {
    row["path"].removeprefix("AdaRES/"): row["sha256"] for row in old_bundle["files"]}
program = "pipeline/analyses/y49_instrumentation_program_c3e5_" + args.token + ".py"
compile((ROOT / program).read_text(encoding="utf-8"), program, "exec")
writer = monitor.acquire_lock(str(ROOT))
assert writer is not None
try:
    now = datetime.now().astimezone()
    freeze_path = "pipeline/analyses/y49_next_native_freeze_c3e5_" + args.token + ".json"
    with (ROOT / freeze_path).open("x", encoding="utf-8") as stream:
        json.dump(dict(
            complete=True, id="AdaRES-Y49", frozen_at=now.isoformat(),
            source_controls=source_ref, native_input_CPU=capture_ref, CPU_hook=hook_ref,
            program=ref(program), program_native_branch_compiled=True,
            native_branch_executed=False,
            next_test="Under an actually idle/exclusively leased authorized server2 GPU, execute the SAME program native branch on fixed TRAIN8: actual W2 model/native full loss/backward, four finite/nonzero new gradients, exact cached-indicator repeat/wrong/query-only controls and paired time/memory.",
            runner="scripts/run_adares_authorized_probe.py y49_native_c3e5 <frozen-program-path> --token <fresh-owned-token> --seconds 900 --gpu --gpu-id <live-idle-authorized-card>",
            later_gates="Independent physical package/config, released-SPFormer initialization, actual tools.train updates and complete model/optimizer/scheduler restore, full resources and substantive six-gate review before any admission.",
            retained_limitations=[
                "Zero native equal-component-mass/different-global-value witness expressions.",
                "Physical constructed pair differs in degree; equal-degree witness is abstract.",
                "CPU surrogate gradients and descriptor runtime are not native utility or resources PASS.",
                "No production package, code commit, receipt or queue entry."]), stream, indent=2)
    freeze_ref = ref(freeze_path)
    work_path = ROOT / "pipeline/research_work.json"
    work = json.loads(work_path.read_text(encoding="utf-8"))
    y49 = next(candidate for candidate in work["candidates"] if candidate["id"] == "AdaRES-Y49")
    y48 = next(candidate for candidate in work["candidates"] if candidate["id"] == "AdaRES-Y48")
    assert y49["stage"] == "implementation" and y48["stage"] == "training"
    y49.update(
        stage="blocked", scientific_status="CPU_information_and_hook_supported_native_GPU_pending",
        blocker_kind="live_external_gpu_occupation", required_server="server-2",
        blocker="The next full native loss/backward/resources test needs an idle exclusive GPU. At local13:00:31-36 both authorized cards had live compute PIDs: GPU0 foreign19723 and GPU1 original Y48 8877. Source/native-input CPU/hook preparation for this next test has been completed, without a native or resource PASS.",
        retry_after=(now + timedelta(hours=1)).isoformat(),
        unblock_condition="A server2 GPU0/1 is freshly confirmed live-idle with no compute PID and an exclusive diagnostic lease. Changed resources can wake earlier; the timer alone is not evidence of a free GPU.",
        next_action="Run SAME hash-frozen actual W2 TRAIN8 native full loss/backward/cached controls/paired costs on a genuinely idle authorized GPU. Preserve zero native component-collision witnesses and all other limits; do not admit or produce variants from CPU gains.",
        attempted_alternatives=[
            "32 actual-source W2/Y31/Y36/Y41/Y46 controls on frozen integer-coordinate physical geometries.",
            "Separate equal-degree connected abstract graph control, joint permutations and repeated fixed unit resolvent.",
            "All frozen native TRAIN8/32 expressions/128 queries: CPU descriptor, cyclic wrong mask, query-only, four proxy gradients, fixed first4 permutation/block-solve checks.",
            "32 CPU final-indicator hook cases: final-foreground only, auxiliary6/disabled/repeat exact and wrong correspondence.",
            "Exact helper AST native program compiled; live authorized occupation checked without allocating a GPU."],
        diagnostic_gates={
            "diagnosis": "constructed explicit-information distinction supported; native task deficiency and utility remain unproved",
            "controls": "32 source,32 captured-input and32 surrogate-hook cases completed, not native loss evidence",
            "native_backward": "unrun",
            "resources": "66561 prototype parameters; native time/memory/Pareto pending",
            "independence": "no independent production package/config yet",
            "constraints": "no validation tuning, query-score diffusion, closed-route reopening or unauthorized GPU use"},
        pending_gates=["native_task_diagnosis", "native_backward", "resources",
                       "independent_production", "released_init_tools_train_restore", "six_gate_review"],
        failure_scope="inconclusive",
        limitations=[
            "Native same-component-mass/different-global-value witness count0.",
            "Physical witness degree3vs4; abstract degree-matched graph is not native geometry.",
            "No novelty, universal blindness, native usefulness or efficacy proof."])
    evidence = [
        source_ref, capture_ref, hook_ref, resource_ref, freeze_ref,
        ref("pipeline/reviews/motivation_review_Y49_c3e5.md"),
        ref("pipeline/designs/exp_design_Y49_c3e5.md"), ref(program),
        ref("scripts/probe_y49_source_controls_c3e5.py"),
        ref("scripts/probe_y49_native_capture_cpu_c3e5.py"),
        ref("scripts/y49_instrumentation_harness_c3e5.py"),
        ref("scripts/build_y49_instrumentation_c3e5.py")]
    y49["evidence"].extend(item for item in evidence if item not in y49["evidence"])
    if resource_ref not in y48["evidence"]:
        y48["evidence"].append(resource_ref)
    # Y46 remains on its original bounded retry; unchanged work is not reopened.
    work["design_cycle"] = dict(
        stop_reason="external_resource_blocked", attempted_ids=["AdaRES-Y49"],
        evidence=[resource_ref, source_ref, capture_ref, hook_ref, freeze_ref],
        resume_condition=y49["unblock_condition"] + " Continue original Y46/Y49 stable IDs before expanding reserve.",
        retry_after=y49["retry_after"],
        rationale="Reserve preparation is now2/2 but the physical GPU gap is external occupation. No new admission, fake exhaustion, deadline padding or CPU-to-native PASS.")
    work_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    logged = subprocess.run(
        ["C:/Program Files (x86)/Git/bin/bash.exe",
         "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/log_decision.sh",
         "AdaRES_project", "other",
         "Y49全局图读数的构造源控制与CPU原型是否足以直接补训练空位？",
         "不准入。仅保留同一Y49为有界储备；完成实际源32例、原生捕获32表达式与钩子32例，等待独占空卡补原生及资源门槛；保留Y46和原Y48。",
         "两授权卡实际占用。Y49原生同组件分区新增全局值见证为0，任务缺口/效用尚未证明；66561参数与代理梯度不能代替原生可达性、成本或独立生产。",
         "; ".join(item["path"] for item in (source_ref, capture_ref, hook_ref, resource_ref, freeze_ref))],
        capture_output=True, encoding="utf-8", timeout=30)
    assert logged.returncode == 0, logged.stderr
    actions = [
        "设计快照2026-10-15 12:48:41+08，真实截止13:48:41+08；本轮一次begin/preflight，0-3段复用脚本事实表。报告生成时postflight尚未执行，最终本轮状态以finish/收据及租约释放核验为准。",
        "紧凑快照due=0、Y46阻塞未到期、储备1/2、准备预算1、准入预算1，design_bootstrap要求先注册候选。12:51:37先注册稳定Y49(global-mask-resolvent-coherence)，再做定向源审查；不是复制Y46修订或重开Y8等旧路线。",
        "12:54:32-35实际W2/Y31/Y36/Y41/Y46源函数32例：整数12点物理几何、恒定特征及相同组件质量，父读数最大差2.22e-16、Y46组件行/权重/分组精确相同，全局读数0.3416667与0.2761905。物理图度数3与4有混杂；另8点同三正则抽象图读数0.3428571与0.3420290不同，但不声称原生几何。只支持构造输入的信息区别。",
        "12:56:20-44固定TRAIN8/all32表达式/all128查询CPU原型：点质量加权FG/BG全局图读数，四张量66561参数，32/32错配发生变化，代理CE最小梯度0.0101877；固定前四表达式联合反序与分块求解对照通过。总测试22.62秒、最慢表达式5.08秒，不是成对原生资源门槛。",
        "重要阴性/局限：原生捕获中，同组件质量分区但全局值不同的见证表达式=0；不删去、不解释成已证明真实任务缺失，也不外推拒绝整个机制。文献新颖性、必要性、效用、训练后收益均未知。",
        "12:59:15-55同一描述符/小头AST精确绑定的CPU钩子32例：仅最终前景改变，六辅助/禁用父路径/重复精确相同，错配最小差0.00689847、四张量代理CE最小梯度0.000360640。已冻结同源原生TRAIN8程序并编译；GPU分支、真实native loss/backward、优化器恢复、生产独立性仍未执行。",
        "13:00:31-36本地实测原Y48 PID8877/配置/cwd/argvGPU1/提交标记及七文件哈希吻合；第2/30轮4070→4079/4686。GPU0外部19723、94%、23586/24576MiB；GPU1 Y48、78%、22714/24576MiB；实际无空闲授权卡。未建立CUDA诊断上下文，未动原训练或外部任务。",
        "科学状态：Y48=training_not_efficacy，Y49=CPU支持/原生GPU验证blocked，Y46原阻塞和修订1保持不变。储备现2/2，本轮新稳定候选1、准入0、新训练0；没有把逻辑空位当物理空卡。",
        "用量收据尚未生成，调用/输入缓存/输出及收据实际产出计数暂未知；finish后只读核实。窗口只含begin至收据的本地可见记录，排除早期读取、最终回复和迟到记录；缓存属于输入、推理属于输出，不重复相加，不作为账单。",
    ]
    decisions = [
        "按真实双卡占用用external_resource_blocked收尾；Y49设置有界1小时复查和真实空卡/独占租约唤醒条件。Y46原有复查时间不被本轮重置。不耗尽截止、不宣称路线穷尽、不降低六项门槛。",
        "Y49仅为储备原型，禁止从CPU代理控制签发准入；后续先同源原生全损失/反传/成本，再独立生产、released-SPFormer实际tools.train/完整状态恢复、资源及逐项审查。任何局部修订仍须原ID预注册，不做验证选择。",
        "保持Y48原released-SPFormer/seed1999/固定30轮/单卡训练和冻结终态评价，W2科学比较对象不变；旧Y45等结论不重开、不重测或晋升。",
        "仅AdaRES_project/server2 GPU0、1，后台调度保持Disabled；没有其他项目、server6、调度更改、新任务或嵌套执行。下一保存自动化时刻未知。",
        "本轮只有finish明确PASS/COMPLETE并独立核实匹配租约消失才称COMPLETED；否则保留证据并按契约abort，不把科学blocked与执行失败混淆。"]
    notes = ROOT / ("pipeline/headless/.notes_" + args.token + ".json")
    with notes.open("x", encoding="utf-8") as stream:
        json.dump(dict(actions=actions, decisions=decisions), stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(complete=True, id="AdaRES-Y49", stage="blocked",
                         retry_after=y49["retry_after"], next_test=freeze_ref,
                         report_notes=str(notes), decision=logged.stdout.strip()),
                     ensure_ascii=False, indent=2))
finally:
    writer.close()
