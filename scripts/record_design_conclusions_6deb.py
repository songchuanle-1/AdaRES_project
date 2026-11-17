"""Persist actual diagnostics, frozen comparisons, next designs and tick notes."""

import argparse
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
import statistics
import unittest

import authorize_prepared_candidates as scoped
from analyze_y27_frozen_terminal import comparison

ROOT = scoped.ROOT
TOKEN = "6debf88fed634a13b772f27bdc244bbd"


def lease():
    owner = scoped.tick.load(ROOT / "pipeline/headless/.foreground.lockdir/owner.json")
    assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
    remaining = (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds()
    assert remaining > 45
    for name, key in ((".tick_context.json", "context_sha256"),
                      (".tick_context_state.next.json", "pending_sha256")):
        assert scoped.tick.digest(ROOT / "pipeline/headless" / name) == owner[key]
    return owner, remaining


def load(name):
    path = "pipeline/analyses/" + name + "_" + TOKEN + ".json"
    value = scoped.tick.load(ROOT / path)
    assert value["complete"]
    return value, scoped.ref(path)


def cycle(work, evidence):
    owner, remaining = lease()
    assert remaining <= 180, "Deadline is not near; continue executable work"
    work["design_cycle"] = dict(
        stop_reason="tick_deadline_near",
        attempted_ids=["AdaRES-Y56", "AdaRES-Y57", "AdaRES-Y52", "AdaRES-Y54"],
        evidence=evidence,
        resume_condition="Continue SAME Y56 native task-loss reachability and SAME Y57 "
                         "full-channel moment prototype controls/backward/resource gates; "
                         "retain scalar null, sparse copy incidence, adverse scene folds and "
                         "frozen terminal comparisons. No automatic commit/admission.",
        actual_deadline_at=owner["deadline_at"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--deadline-close", action="store_true")
    args = parser.parse_args()
    assert args.token == TOKEN
    owner, remaining = lease()
    lock = scoped.monitor.acquire_lock(str(ROOT))
    assert lock is not None, "An experiment probe is still writing"
    try:
        work = scoped.tick.load(ROOT / "pipeline/research_work.json")
        if args.deadline_close:
            review = scoped.ref("pipeline/analyses/design_and_terminal_review_" + TOKEN + ".json")
            cycle(work, [review])
            scoped.tick.save(ROOT / "pipeline/research_work.json", work)
            print("Recorded actual immutable deadline disposition; no gate changed.")
            return
        candidates = {c["id"]: c for c in work["candidates"]}
        names = ("y52", "y54", "w2", "b1")
        wrappers, eval_refs = {}, {}
        for name in names:
            wrappers[name], eval_refs[name] = load("frozen_" + name)
        captures = {name: value["rows"] for name, value in wrappers.items()}
        for capture in captures.values():
            assert capture["complete"] and capture["completed_passes"] == 2
            assert capture["active_queries"] == [128] and capture["selected_queries"] == 1
            assert all(p["summary"]["expressions"] == 9508 and p["summary"]["scenes"] == 141
                       for p in capture["passes"])
        comparisons, harms = {}, {}
        for name in ("y52", "y54"):
            comparisons[name] = {ref: comparison(captures[name], captures[ref])
                                 for ref in ("w2", "b1")}
            harms[name] = [
                dict(reference=ref, group=group, result=row)
                for ref, groups in comparisons[name].items() for group, row in groups.items()
                if group != "all" and row.get("scene_clustered_ci95", [0, 0])[1] < 0]
        native56, native56_ref = load("y56_native_batch2")
        broad56, broad56_ref = load("y56_broader_unique")
        transfer56, transfer56_ref = load("y56_actual_transfer")
        weak57, weak57_ref = load("y57_native_signal")
        scalar57, scalar57_ref = load("y57_full_input_signal")
        vector57, vector57_ref = load("y57_vector_signal")
        assert native56["exact_joint_duplicate_excess"] == 0
        assert broad56["fixed_scenes"] == 32 and broad56["expression_layer_records"] == 896
        assert broad56["exact_joint_duplicate_excess"] == 3
        assert transfer56["actual_duplicate_records"] == transfer56["transfer_cases"] == 3
        cases = [r for r in transfer56["cases"] if r["transfer"] == "executed"]
        assert all(r["inverse_multiplicity_focal_change"] <= r["arithmetic_bound"] for r in cases)
        assert all(r["native_repeat_max_abs"] == 0 for r in cases)
        review_path = "pipeline/analyses/design_and_terminal_review_" + TOKEN + ".json"
        assert not (ROOT / review_path).exists()
        test_path = ROOT / "scripts/test_design_followups_6deb.py"
        spec = importlib.util.spec_from_file_location("design_followup_tests", str(test_path))
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        log = io.StringIO()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(tests))
        assert result.wasSuccessful() and result.testsRun == 6
        tested = dict(tests_run=result.testsRun, passed=True, output=log.getvalue(),
                      source=scoped.ref(test_path.relative_to(ROOT).as_posix()))
        review = dict(
            complete=True, token=TOKEN, reviewed_at=datetime.now().astimezone().isoformat(),
            terminal_evaluations=eval_refs,
            pending_evaluation="Y36 SAME frozen two-pass program was NOT started: runner "
                               "remaining-time reserve check rejected before SSH/log/output. "
                               "No Y36 result is substituted from historical evaluation.",
            comparative_status="PARTIAL: four methods/eight full passes, Y36 comparison pending",
            analysis_implementation=scoped.ref("scripts/analyze_y27_frozen_terminal.py"),
            summary={name: capture["passes"][0]["summary"] for name, capture in captures.items()},
            comparisons=comparisons, frozen_subgroup_harm_review=harms,
            first_primary=True, scene_bootstrap=dict(draws=10000, seed=1999),
            train_size_cuts=[1487, 3325], target=0.515,
            any_new_candidate_reaches_target=any(
                captures[name]["passes"][0]["summary"]["miou"] >= 0.515 for name in ("y52", "y54")),
            costs={name: dict(
                parameters=capture["parameters"],
                warm_first_chunk_median_seconds=statistics.median(capture["warm_first_chunk_seconds"]),
                peak_allocated_gib=capture["peak_allocated_gib"],
                peak_reserved_gib=capture["peak_reserved_gib"],
                repeat_miou_difference=capture["repeat_miou_difference"],
                repeat_max_expression_difference=capture["repeat_max_iou_difference"])
                   for name, capture in captures.items()},
            Y56=dict(
                native=native56_ref, broader=broad56_ref, transfer=transfer56_ref,
                disjoint_scene_clusters=40, expressions=160, correlated_layer_records=1120,
                exact_copy_excess=3, positive_scene_clusters=1, positive_layer=0,
                final_layer_copies=0,
                confidence_change_range=[min(r["native_focal_confidence_change"] for r in cases),
                                         max(r["native_focal_confidence_change"] for r in cases)],
                diagnosis="Rare actual copy occurrence and fixed-focal nuisance response observed; "
                          "not task loss reachability, prediction harm, population prevalence or efficacy."),
            Y57=dict(
                confidence_only_screen=weak57_ref, full_input_scalar=scalar57_ref,
                full_input_vector=vector57_ref,
                scalar_gain=scalar57["loso_gain_over_full_inputs"],
                vector_gain=vector57["loso_gain_over_full_inputs"],
                vector_vs_shifted=vector57["loso_gain_over_same_capacity_shifted"],
                adverse_scene_folds=[r for r in vector57["folds"] if r["gain_over_full_inputs"] < 0],
                diagnosis="Full-vector matched-control signal warrants a same-ID prototype; "
                          "interval versus full native inputs crosses0. Eight TRAIN scenes, "
                          "adaptive scalar-to-vector follow-up, not independent confirmation "
                          "or conditional independence/causal/efficacy proof."),
            focused_regressions=tested,
            limitations=[
                "One training seed per method; scene bootstrap does not estimate training seed uncertainty.",
                "Repeats estimate observed evaluation variability only; not a universal noise bound.",
                "Latency is five warm first-chunk forwards; full-dataset latency distribution unknown.",
                "No new production model/config, optimizer steps, admission, training launch, Git commit or scheduler change."])
        scoped.tick.save(ROOT / review_path, review)
        reference = scoped.ref(review_path)
        for name, cid in (("y52", "AdaRES-Y52"), ("y54", "AdaRES-Y54")):
            candidate = candidates[cid]
            candidate["evidence"].extend([*eval_refs.values(), reference])
            candidate.update(
                stage="validation", scientific_status="fixed30_complete_Y36_frozen_comparison_pending",
                scientific_outcome=reference,
                failure_scope="implementation" if harms[name] else "inconclusive",
                next_action="Execute SAME frozen Y36 full two-pass evaluation under a fresh "
                            "owning token/idle lease, then compare both candidates with Y36 "
                            "using unchanged frozen rules. Preserve original fixed30/best/last and FIRST-primary full "
                            "comparison; do not promote, tune epochs/thresholds, automatically "
                            "rerun or infer whole-mechanism falsification. " +
                            ("Predeclared subgroup harm triggers a review." if harms[name] else
                             "No frozen subgroup harm trigger; inspect paired uncertainty and costs "
                             "before claiming improvement."))
        c56, c57 = candidates["AdaRES-Y56"], candidates["AdaRES-Y57"]
        c56["evidence"].extend([broad56_ref, transfer56_ref, reference,
            scoped.ref("pipeline/analyses/y56_broader_unique_" + TOKEN + ".log"),
            scoped.ref("pipeline/analyses/y56_actual_transfer_" + TOKEN + ".log")])
        c56.update(
            stage="design", scientific_status="rare_initial_layer_copy_nuisance_native_witness",
            next_action="Preregister and execute SAME Y56 actual native task-loss/gradient "
                        "reachability for exact joint-copy inverse-multiplicity context, "
                        "with unchanged-mask/unique-input controls and all seven loss layers. "
                        "Rare layer0 incidence is not final-selector harm or trained benefit.")
        c56["gate_summary"].update(
            diagnosis="Actual40disjointTRAIN scenes/160expressions/1120correlated layer records: "
                      "3exact-copy excess in one scene/layer0; final-layer incidence0.",
            controls="Actual3fixed-focal transfers, unique set/focal/GT unchanged; parent "
                     "confidence shifts .00171-.00444, inverse-multiplicity reference "
                     "<=1.20e-7; repeat0, permutation<=6.26e-7. Task loss not yet tested.")
        c57["evidence"].extend([vector57_ref, reference,
            scoped.ref("pipeline/analyses/y57_vector_signal_" + TOKEN + ".log")])
        c57.update(
            stage="design", scientific_status="all_channel_moment_observational_signal_inconclusive",
            next_action="Prepare SAME Y57 independent full256diagonal peer-variance input "
                        "prototype, retaining all existing769inputs/all7attachments/W2loss. "
                        "Before admission execute same-capacity shifted/permutation/equal-peer "
                        "controls, actual native backward for every new tensor, and paired "
                        "resources. Retain scalar null and two adverse scene folds; no validation search.")
        c57["gate_summary"].update(
            diagnosis="Confidence-only gain .0068004 disappears for full-input scalar "
                      "(-.00037594). Full256moment mean gain .0237282 but CI crosses0; "
                      "matched shifted gain .0243900. Not causal or efficacy proof.",
            controls="All769native normalized head inputs reconstructed; fixed unit ridge "
                     "LOSO with train-fold scaling/allchannels, no alpha/layer/seed scan. "
                     "Production intervention controls and backward remain pending.")
        for candidate in (c56, c57):
            for revision in candidate.get("diagnostic_revision_history", []):
                if revision["scope"] in ("prediction_blind_sampling_unit", "all_channel_second_moment_not_scalar"):
                    revision["result"] = "Executed; full results and limitations preserved."
                    revision["result_evidence"] = [reference]
        for candidate in candidates.values():
            if candidate["id"] in ("AdaRES-Y52", "AdaRES-Y54"):
                for row in candidate["evidence"]:
                    scoped.closure.artifact(ROOT, row)
        now = datetime.now().astimezone().isoformat()
        goal = scoped.tick.load(ROOT / "pipeline/goal.json")
        review_md = "pipeline/reviews/motivation_review_Y56_Y57_" + TOKEN + ".md"
        design_md = "pipeline/designs/exp_design_Y56_Y57_" + TOKEN + ".md"
        table = "\n".join(
            "| " + name.upper() + " | %.8f | %.6f | %.6f |" % (
                capture["passes"][0]["summary"]["miou"],
                capture["passes"][0]["summary"]["acc25"],
                capture["passes"][0]["summary"]["acc50"])
            for name, capture in captures.items())
        motivation = f"""# Motivation Review - AdaRES_project
采集/审查：{now}。上游证据：`{review_path}`。本轮不是新训练。
## 1. Goal
```json
{json.dumps(goal, ensure_ascii=False, indent=2)}
```
历史规范冠军Y36 .5110、目标.5150、差.0040；本轮冻结重评估另列，不混用监控分段best。
## 2. Story Status
| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| Y52点质量目标 | fixed30和自身冻结评估完成，比较未全闭环 | 全9508表达/141场景双FIRST；W2/B1已各双次 | 新Y36双次及配对比较待补，不自动晋升 |
| Y54细支持有效性 | 同上 | 同协议，当前W2/B1比较已做 | Y36待补，不把单次下降当机制反证 |
| Y56复制数量干扰上下文 | 局部原生见证 | 40独立场景中一个场景初始层3记录 | 原生任务损失可达性及最终伤害未知 |
| Y57均值丢失peer散布 | 完整向量有观察信号，仍不确定 | 原头全部输入条件下固定LOSO | 原生可达干预/后向/资源及训练收益未知 |
## 3. Experiment Assessment
以下四方法均本轮FIRST全量结果；Y36尚未运行，不替代历史结果，单位保持原mIoU/Acc：
| 实验 | mIoU | Acc25 | Acc50 |
|---|---|---|---|
{table}
预注册伤害触发数Y52={len(harms['y52'])}、Y54={len(harms['y54'])}，逐组保留；
是否超过基线或冠军须看配对场景区间，不能用rounded best代替。
## 4. Gap Analysis
Y56初8场景无复制；新增32个不同场景有3个精确复制额外项，均为scene0029_00/layer0，
不是1120独立样本，也不是hardmask相等。固定focal0/唯一集/GT的实际复制转移使置信度变化
.0017069-.0044413；逆复制参考误差<=1.20e-7，重复0，排列<=6.26e-7。并未证明任务损失变化。
Y57置信度条件正结果不能代替全输入条件证据：散布标量增益-.00037594、区间跨0；
完整256通道增益.02372823、区间[-.00259745,.04683551]，对同容量错位对照
.02439002/[.00053385,.04511317]。scene0004_00和scene0005_00下降保留。
该向量补测在标量结果之后预注册，属于自适应探索，不是独立确认或条件独立性证明。
## 5. Design Rationale
同一Y56先验证稀少复制是否能通过实际任务损失产生梯度；不从局部置信度扰动直接上训练。
同一Y57保留原头全部输入并增加逐通道二阶矩，避免标量抹去方向；
结构固定于既有d_model256，不扫描特征、层、正则、阈值或超参数。新颖性尚未检索确认。
## 6. Next Batch
见`{design_md}`。两项均未正式实现、提交、准入或训练；不存在缺新GPU授权的当前阻塞。
"""
        design = f"""# Experiment Design - AdaRES_project / Y56, Y57
创建：{now}。Status: DRAFT。上游：`{review_md}`。保持原稳定ID，不新增占位候选。
## 1. Motivation
项目动机原文：{goal['description']}
队列0/容量2，准入预算2；续做两个已有候选，不以填满GPU替代门槛。
## 2. Current Best Results
| 实验 | mIoU | Acc25 | Acc50 |
|---|---|---|---|
{table}
以上四方法9508表达/141场景各双次；FIRST为主。Y36新配对评估因真实剩余时间不足未启动，
完整五方法比较尚未闭环。重复、当前子组和成本见`{review_path}`。
## 3. Analysis & Direction
### 3.1 当前结果
Y52/Y54原fixed30结束；不因epoch30日志的分段best较低覆盖原best，也不新选epoch。
### 3.2 负证据
Y57标量在全部输入条件下无稳定增益，完整向量有两个下降场景，均保留。
Y56最终层未见复制，不能把初始层局部现象泛化。
### 3.3 尚缺证据
两项正式独立实现、控制分离、原生后向、新张量梯度、配对资源、训练收益均未完成。
### 3.4 决定
Y56优先做原生任务损失可达性，不直接实现生产去重；
Y57下一项是全通道二阶矩诊断原型及匹配对照，不从观察关联生成准入。
## 4. Experiment Plan
| ID | 配置 | 下一项 | 预期/失败判据 | 依赖 |
|---|---|---|---|---|
| Y56 | 尚未创建 | 当前真实复制case与无复制输入，冻结逆复制上下文干预并逐层捕获原生损失/梯度 | 可为无活动/反向/有活动，均保留；不是固定收益门槛 | 本轮case证据 |
| Y57 | 尚未创建 | 独立包拟名adares_y57_peer_second_moment，保留769维并拼接256维逐通道方差；全部7层共享原挂接 | 对齐/错位同容量、排列/同peer、原生loss与每个新张量有限非零梯度 | 本轮完整向量证据和全部负结果 |
固定W2监督/30轮/seed1999/Q128/单argmax/SPFormer初始化，不使用诊断训练W2权重正式初始化。
Y57仅改peer上下文信息流；不重开Y42超点内部特征分布或Y50联合mask覆盖。
设计预测参数增量66048，仅是形状计算，不是资源PASS；不裁通道或选层。
## 5. GPU Allocation Plan
仅server-2 GPU0/1，每卡单任务。下项每次重查空闲/进程并获得独占诊断租约；
本轮冻结评估在GPU1串行执行，不从旧空闲快照宣称当前卡空闲。不用server-6。
## 6. Implementation Checklist
- 已完成：source/checkpoint/GT绑定、原生batch2观察、同容量/错位和实际复制匹配诊断、6项分析回归。
- 待完成：正式物理独立6文件包+唯一YAML、代码级控制、原生全后向/新参数梯度、配对时延显存。
- 待门槛全部通过后，按实际已有授权核对精确测试文件的提交范围；当前无测试完生产包，不制造授权阻塞。
- 未来训练前继续冻结全9508/141双FIRST，与W2/B1/Y36各双次；scene-bootstrap10000/1999，
  TRAIN cuts1487/3325，任何有意义子组对任一参考CI上界<0触发伤害审查。
  报告mIoU/Acc25/Acc50/Q/时延/显存/参数；噪声来自独立重复，不造常数。
## 7. Risk Assessment
八场景与自适应补测不能证明总体收益；局部复制反事实不是预测伤害。
未建立新颖性、跨seed稳定性或全局Pareto。正式实现前所有受影响门槛仍须实测。
## 8. Smell Check
Y57增加均值聚合不可恢复的peer二阶矩信息通路，而非调整温度、loss系数或验证阈值；
全部256通道来自原d_model，未择优。Y56是复制计数的不变性诊断，并非改名重开历史拒绝方向。
"""
        for path, text in ((review_md, motivation), (design_md, design)):
            assert not (ROOT / path).exists()
            (ROOT / path).write_text(text, encoding="utf-8", newline="\n")
        for candidate in (c56, c57):
            candidate["evidence"].extend([scoped.ref(review_md), scoped.ref(design_md)])
        notes = dict(
            actions=[
                "本轮20:26规范快照design，Y52/Y54已done_complete且队列0；不重复preflight，不修改调度。",
                "Y56原8场景224记录无精确复制；修正语言chunk采样单位后新增32不同场景896记录，发现一个场景初始层3复制记录。原失败日志保留。",
                "Y56实际3案例固定focal0/唯一集/GT复制转移，原头置信度变化.00171-.00444，逆复制参考<=1.20e-7；未证明任务损失/最终分割伤害。",
                "Y57置信度条件信号在全输入标量对照消失；完整256向量有错位对照分离，但对全输入区间跨0、两个下降场景，仍不确定。",
                "Y52/Y54/W2/B1四方法各新执行9508表达/141场景完整双次FIRST冻结评估；同一原best断点，全SHA/完整配置值/运行源码绑定通过，原始字段和序列化检查失败均留档。Y36因runner剩余时间保护启动前拒绝，未执行且未创建日志，不虚报第五方法。",
                "本轮FIRST mIoU：" + "；".join(
                    name.upper() + " %.12f" % capture["passes"][0]["summary"]["miou"]
                    for name, capture in captures.items()) + "；不选择较好重复。",
                "预注册任何子组对任一参考CI上界<0触发数：Y52 " + str(len(harms["y52"])) +
                "，Y54 " + str(len(harms["y54"])) + "；逐组及完整overall区间均保留，不只看均值。",
                "完整配对scene-bootstrap10000/1999、预注册子组伤害和成本已写审查JSON；6项聚焦回归通过。"],
            decisions=[
                "Y52/Y54固定训练和自身全量评估完成，当前W2/B1比较已做，仍待同冻结Y36双次及配对审查；不由rounded监控或重启分段best替代规范评估，不自动晋升、重跑或调参。",
                "只续做原Y56/Y57稳定ID；下一项分别为实际任务损失可达性、完整逐通道二阶矩独立原型与控制/后向/资源。未新增模型、Git提交、准入或训练。",
                "完整六段报告0-3保留20:26 preflight事实，4-5记载本轮后续研究；finish前不预称校验成功。"])
        note_path = "pipeline/reports/notes_" + TOKEN + ".json"
        scoped.tick.save(ROOT / note_path, notes)
        summary = review["summary"]
        memory = f"""# AdaRES_project Research Memory
## Goal And Phase
- 本轮token {TOKEN}，规范快照2026-10-20 20:26:25.438415上海；真实deadline21:26:25.438415，不修改。manual“继续设计吧”，mode design/active0/capacity2/budget2/reserve_budget0；仅本项目。
- owner01a07e32-78bf-7670-925b-370911ea48db由真实CODEX_THREAD_ID核实；只server-2 GPU0/1各单实验，不改调度，不跨项目。
- 原goal目标.5150；历史规范CSV冠军Y36 .5110/B1 .5020/差.0040。本轮全量比较单列，不混用rounded monitor指标。
## Champion And Baseline
- 本轮四方法各新9508表达/141scene双次FIRST：Y52 {summary['y52']['miou']:.12f}；Y54 {summary['y54']['miou']:.12f}；W2 {summary['w2']['miou']:.12f}；B1 {summary['b1']['miou']:.12f}。Y36未执行，不替换历史值。
- 逐对scene-bootstrap10000/1999、TRAIN cuts1487/3325、任何子组CI上界<0伤害触发、重复噪声和成本见`{review_path}`，SHA{reference['sha256']}。
- Y52/Y54固定30轮正常结束，best分别epoch24/.5097084641456604与epoch27/.5063441395759583；last均epoch30、optimizer543状态、scheduler30；best/last全SHA核对，原checkpoint不改。
- 不把monitor最新resume分段best .5090/.5015当保留best或新评估；训练seed不确定性仍未知，不由scene区间代替。
## Active Hypotheses
- Y56/Y57已从resolved GPU阻塞推进design，非缺新授权；未创建正式候选包/config/准入，不把未执行门槛当反证。
- Y56原8TRAIN场景32表达224层记录精确复制0；新的32个不同场景128表达896记录中复制额外项3，均scene0029_00/layer0/ann0,1,3。合计40独立场景，不是1120独立样本；最终层0，hardmask相等单列不混淆。
- Y56首次扩大采样误将chunk索引当独立场景，末尾断言失败并留日志；同ID冻结metadata首32不同场景且排除旧8后重跑，不选预测或删负证据。
- 实际三个固定focal0复制转移保持唯一query+softmask集/focal/GT；原头置信度变化.0017068982-.0044413209，逆复制focal logit变化<=1.1920929e-7，重复0/排列<=6.25849e-7。原生任务loss/梯度和最终伤害未知。
- Y56下一项：冻结同3实际case和无复制对照，检验逆复制上下文干预经各原生损失层的可达性/梯度；不直接修正或训练。
- Y57新鲜同8TRAIN场景/32表达/all7层source/checkpoint/point-GT绑定完成，最终层6为主。模型参数不改、BN buffers还原，无optimizer/权重保存。
- 置信度条件LOSO OLS均值gain .00680041238，但全769head inputs+confidence条件下标量ridge gain -.000375942875/CI[-.00312000814,.00160019069]，正解释被削弱，完整保留。
- 同ID诊断测量revision1：全部256逐通道variance代替L2标量，固定sumSSE+1L2、LOSO/train-fold标准化/不clip；均值gain .02372822827/CI[-.00259744816,.04683550922]。
- 对同容量query轴shift1向量对照gain .02439002189/CI[.00053384573,.04511316674]；scene0004_00和scene0005_00下降。对照证据不是条件独立性/因果任务缺陷/收益或新颖性证明；自适应补测不是独立确认。
- Y57下一项：同ID物理独立全256二阶矩原型，保留原769输入/全部7层/W2loss，拟参数增量66048只是形状计算；先匹配控制、每新张量原生梯度及配对资源，禁止验证集特征/层/alpha选择。
## Closed Or Failed Directions
- Y52/Y54保持validation：固定训练及自身双次评估已结束，当前W2/B1子组伤害触发数Y52 {len(harms['y52'])}/Y54 {len(harms['y54'])}；待Y36完整新双次及配对比较后才能闭环。不自动晋升、重跑或调epoch/阈值，不外推全机制证伪。
- Y55/Y53/Y50/Y51/Y49rev1/Y46/Y48等历史拒绝/关闭不变，不重开或改名；Y56采样失败、terminal reader scheduler字段/配置序列化失败均保留，非科学反证。
## Running And Queue
- 本轮无新训练启动/入队/认领/准入/Git提交或推送。Y52/Y54规范done_complete/30轮，队列0。
- 当前HEAD仍8943c8c460501b200a8dea958964579ffea65f76；保留所有无关dirty文件。旧Y52/Y54精确提交授权不外推到未测试Y56/Y57生产包。
- 本轮诊断与冻结评估均经当前租约、每次GPU占用/独占检查串行执行；不从20:27空闲缓存推定结束时GPU空闲。新启动必须重查。
## Open Questions And Risks
- Y56稀少初始层现象如何影响实际loss仍未知；Y57完整向量信号对全输入区间跨0，场景/seed泛化、真实干预/后向/成本及文献新颖性未建立。
- 完整设计：`{design_md}`；上游六段review：`{review_md}`；不把设计和形状预算写成实测PASS。
- 仅单训练seed；两次评估不是通用噪声上界；时延只在相同首chunk的5次warm forward测量，official unique/multiple未知。
- Y36程序pipeline/analyses/frozen_y36_{TOKEN}_program.py启动前因runner时间余量校验拒绝，未SSH/未建日志或结果。下一轮新token重查原source/checkpoint/空闲，执行同科学程序后补两项对Y36比较，不扩大deadline或复用旧token。
## Last Tick
- 已执行：两项冻结原生观察、置信度/全输入标量/全向量对照、32新场景复制诊断与实际case转移、四方法8次全量评估、当前W2/B1配对伤害/成本审查、6项分析回归；五方法比较未全闭环。
- 完整证据/收据保留当前token。finish前未预称PASS；以canonical finish退出/FINALIZED/context提交及锁/pending消失确认，释放后不补写。
"""
        assert len(memory.splitlines()) < 120
        (ROOT / "pipeline/codex_memory.md").write_text(memory, encoding="utf-8", newline="\n")
        if remaining <= 180:
            cycle(work, [reference, scoped.ref(design_md)])
        lease()
        scoped.tick.save(ROOT / "pipeline/research_work.json", work)
        scoped.monitor.log_decision(
            str(ROOT), "next_batch", "继续设计，固定训练已结束，需区分局部机制证据与效益。",
            "保留Y56实际复制证据和Y57全部负结果；仅续做同ID原生可达性/全通道二阶矩原型。"
            "Y52/Y54完整冻结比较留档，不自动晋升、重跑或准入。",
            "四方法全量双FIRST与当前W2/B1预注册scene/子组审查，Y36尚未启动；实际复制匹配对照及全输入二阶矩对照。",
            review_path + "; " + design_md)
        print(json.dumps(dict(
            review=reference, notes=note_path,
            summary=summary, harm_trigger_counts={k: len(v) for k, v in harms.items()},
            tests=6, candidates="Y52/Y54 validation pendingY36; Y56/Y57 design; no new training",
            remaining_seconds=lease()[1]), ensure_ascii=False, indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
