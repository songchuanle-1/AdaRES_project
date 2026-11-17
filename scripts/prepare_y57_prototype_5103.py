"""Realize the already frozen full-channel Y57 design, without committing it."""

import argparse
import ast
import base64
from datetime import datetime
import hashlib
import json

import yaml
from research_5103 import ROOT, TOKEN, lease, ref, save, bind, tick, monitor

PACKAGE = "adares_y57_peer_second_moment"
PARENT = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
DEST = ROOT / "AdaRES/ipdn/model_paper5" / PACKAGE
CONFIG = ROOT / "AdaRES/configs_song/model_paper5" / (PACKAGE + "_res.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    lease()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(c for c in work["candidates"] if c["id"] == "AdaRES-Y57")
        assert candidate["stage"] == "design" and not DEST.exists() and not CONFIG.exists()
        prior_design = "pipeline/designs/exp_design_Y56_Y57_6debf88fed634a13b772f27bdc244bbd.md"
        assert ref(prior_design) in candidate["evidence"]
        source = (PARENT / "dec.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        head = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QueryCompetitionIndicator")
        old_head = ast.unparse(head)
        assert old_head.count("3 * d_model + 1") == 2
        new_head = old_head.replace("3 * d_model + 1", "4 * d_model + 1")
        old = "max_overlap = overlap.max(-1).values.unsqueeze(-1)"
        assert new_head.count(old) == 1
        new_head = new_head.replace(old,
            "second_moment = torch.einsum('bij,bjd->bid', competition, normalized_query.square())\n"
            "        peer_variance = (second_moment - context.square()).clamp_min(0.0)\n"
            "        " + old)
        old = "normalized_query - context, max_overlap]"
        assert new_head.count(old) == 1
        new_head = new_head.replace(old, "normalized_query - context, max_overlap, peer_variance]")
        lines = source.splitlines(keepends=True)
        changed = "".join(lines[:head.lineno - 1]) + new_head + "\n" + "".join(lines[head.end_lineno:])
        parsed = ast.parse(changed, feature_version=(3, 8))
        old_nodes = [n for n in tree.body if n is not head]
        new_nodes = [n for n in parsed.body if not (isinstance(n, ast.ClassDef) and n.name == head.name)]
        assert [ast.dump(n, include_attributes=False) for n in old_nodes] == [
            ast.dump(n, include_attributes=False) for n in new_nodes]
        goal = tick.load(ROOT / "pipeline/goal.json")
        review_path = "pipeline/reviews/motivation_review_Y57_" + TOKEN + ".md"
        review = f"""# Motivation Review - AdaRES_project / Y57
## 1. Goal
```json
{json.dumps(goal, ensure_ascii=False, indent=2)}
```
历史规范冠军Y36 .5110，目标.5150，差.0040；Y36新配对评估正在执行，不预写结果。
## 2. Story Status
同一Y57均值混叠假设，续做上轮完整256通道方案，不创建新ID或重开历史拒绝路线。
## 3. Experiment Assessment
上轮Y52 .509708428366/Y54 .506342952986/W2 .510107995189/B1 .502221764356为全量FIRST，
不是本轮新结果；Y52/Y54对W2总体区间跨0。原8场景标量全输入条件信号为负，完整保留。
## 4. Gap Analysis
完整256通道对全输入gain .02372823且CI跨0，对同容量shift gain .02439002；两个下降场景
保留。八TRAIN场景与自适应补测不是因果/总体/跨seed收益或新颖性证明。
## 5. Design Rationale
按`{prior_design}`保留769原输入，增加全部256peer对角方差，宽度固定1025，
head hidden256、全部7层、原W2监督/seed1999/30轮/Q128不改。预测新增66048参数是形状计算，
不是资源PASS。先匹配控制、原生后向及配对资源，不用验证集调参。
## 6. Next Batch
只实现独立六文件包及唯一res YAML。CPU/原生/成本测试逐项执行；尚无Git提交、准入或训练。
"""
        assert not (ROOT / review_path).exists()
        (ROOT / review_path).write_text(review, encoding="utf-8", newline="\n")
        DEST.mkdir()
        files = []
        for path in sorted(PARENT.glob("*.py")):
            destination = DEST / path.name
            if path.name == "dec.py":
                destination.write_text(changed, encoding="utf-8", newline="\n")
            else:
                destination.write_bytes(path.read_bytes())
            files.append(ref(destination))
        parent_config = ROOT / "AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
        CONFIG.write_bytes(parent_config.read_bytes())
        files.append(ref(CONFIG))
        assert len(files) == 7
        for row in files:
            path = ROOT / row["path"]
            if path.suffix == ".py":
                code = ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 8))
                assert not any(isinstance(n, ast.ImportFrom) and "adares_" in (n.module or "")
                               for n in ast.walk(code))
        assert len(list((ROOT / "AdaRES").rglob(CONFIG.name))) == 1
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        assert cfg == yaml.safe_load(parent_config.read_text(encoding="utf-8"))
        bundle_path = "pipeline/analyses/y57_prototype_bundle_" + TOKEN + ".json"
        bundle = save(bundle_path, dict(
            complete=True, id="AdaRES-Y57", token=TOKEN,
            created_at=datetime.now().astimezone().isoformat(), files=files,
            parent_files=[ref(p) for p in sorted(PARENT.glob("*.py"))],
            config_values=cfg, package=PACKAGE, only_indicator_class_changed=True,
            all_other_AST_nodes_equal=True, predicted_parameter_delta=66048,
            formal_code_uncommitted=True, tests="pending", no_training_or_admission=True))
        frozen_path = ROOT / "pipeline/analyses/y57_native_task_ef2c7bd97d37400ea1b1a32da2909760_program.py"
        original = ast.parse(frozen_path.read_text(encoding="utf-8"))
        frozen = ast.literal_eval(next(n.value for n in original.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "FROZEN" for t in n.targets)))
        payload = dict(package=PACKAGE, bundle=bundle, files={
            (ROOT / r["path"]).name: dict(sha256=r["sha256"], base64=base64.b64encode(
                (ROOT / r["path"]).read_bytes()).decode("ascii")) for r in files},
            parent_sources=frozen["sources"], checkpoint=frozen["checkpoint"],
            checkpoint_sha256=frozen["checkpoint_sha256"], scenes=frozen["scenes"],
            commit=frozen["commit"], config_values=cfg,
            parent_class=old_head, config_name=CONFIG.name)
        prefix = "PAYLOAD=" + repr(payload) + "\n"
        setup_path = ROOT / "scripts/setup_y57_prototype_5103.py"
        for name in ("cpu", "native", "resources"):
            body_path = ROOT / ("scripts/probe_y57_" + name + "_5103.py")
            program = prefix + setup_path.read_text(encoding="utf-8") + "\n" + body_path.read_text(encoding="utf-8")
            ast.parse(program, feature_version=(3, 8))
            path = "pipeline/analyses/y57_" + name + "_" + TOKEN + "_program.py"
            assert not (ROOT / path).exists()
            (ROOT / path).write_text(program, encoding="utf-8", newline="\n")
            plan = dict(complete=True, id="AdaRES-Y57", token=TOKEN,
                preregistered_at=datetime.now().astimezone().isoformat(), tested_bundle=bundle,
                program=ref(path), body=ref(body_path), setup=ref(setup_path),
                sample="Original8TRAIN scenes/32expressions, four native batch2 pairs; no selection by outcome.",
                controls="Keep all769inputs, append full256variance. Repeat/joint permutation, "
                         "same1025width query-axis cyclic-shift1 variance, equal-peer/no-overlap/padding negatives.",
                native="Released-SPFormer initialization only, unchanged loss. Every changed-shape "
                       "parameter and all indicator tensors finite nonzero gradients; variance evidence "
                       "and new first-layer columns actually reached. No optimizer or saved weights.",
                resources="Exact66048parameter delta; largest input-point-count batch among fixed4, "
                          "2warmups/3native backward timings under24GiB. Parent/candidate same common "
                          "initial weights, batch1 native predicts on all8fixed scenes,2warmups/3timings.",
                limits="Pre-training activity/resources only; no task-gain requirement, efficacy, "
                       "global Pareto, novelty, Git authorization or admission. Retain all negatives.", result="pending")
            plan_ref = save(path.replace("_program.py", "_prereg.json"), plan)
            bind(candidate, ref(path), plan_ref, ref(body_path))
        bind(candidate, ref(review_path), bundle, ref(setup_path), ref(__file__))
        candidate.update(stage="implementation", tested_bundle=bundle,
            next_action="Execute frozen Y57 CPU controls, then physical native backward and paired "
                        "resources in THIS tick; every production gate remains unpassed until tested.")
        work.pop("design_cycle", None)
        lease()
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(bundle=bundle, files=7, stage="implementation", no_commit=True), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
