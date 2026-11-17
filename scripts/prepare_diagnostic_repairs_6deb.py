"""Preserve failed diagnostics and freeze narrowly scoped replacements."""

import argparse
import ast
from datetime import datetime
import json

import authorize_prepared_candidates as scoped

ROOT = scoped.ROOT
TOKEN = "6debf88fed634a13b772f27bdc244bbd"


class SchedulerKey(ast.NodeTransformer):
    count = 0

    def visit_Constant(self, node):
        if node.value == "lr_scheduler":
            self.count += 1
            return ast.copy_location(ast.Constant(value="scheduler"), node)
        return node


class FullSecondMoment(ast.NodeTransformer):
    changes = 0

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "dispersion"
                and isinstance(node.value, ast.Call)):
            self.changes += 1
            node.value = ast.Name(id="variance", ctx=ast.Load())
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np" and node.func.attr == "roll"):
            self.changes += 1
            node.keywords.append(ast.keyword(arg="axis", value=ast.Constant(value=0)))
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value == 771:
            self.changes += 1
            return ast.copy_location(ast.Constant(value=1026), node)
        return node


def save_program(name, source, plan):
    ast.parse(source, feature_version=(3, 8))
    compile(source, name, "exec")
    path = "pipeline/analyses/" + name + "_" + TOKEN + "_program.py"
    assert not (ROOT / path).exists()
    (ROOT / path).write_text(source, encoding="utf-8", newline="\n")
    plan.update(
        complete=True, token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(),
        program=scoped.ref(path), result="pending", no_admission_or_training=True)
    prereg = path.replace("_program.py", "_prereg.json")
    scoped.tick.save(ROOT / prereg, plan)
    return [scoped.ref(path), scoped.ref(prereg)]


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
        candidates = {c["id"]: c for c in work["candidates"]}
        terminal = "pipeline/analyses/terminal_bindings_" + TOKEN + "_program.py"
        transform = SchedulerKey()
        tree = transform.visit(ast.parse((ROOT / terminal).read_text(encoding="utf-8")))
        assert transform.count == 1
        refs = save_program(
            "terminal_bindings_v2", ast.unparse(ast.fix_missing_locations(tree)) + "\n",
            dict(
                original_program=scoped.ref(terminal),
                failed_log=scoped.ref("pipeline/analyses/terminal_bindings_" + TOKEN + ".log"),
                scope="Checkpoint reader key only: actual saved key is scheduler, not lr_scheduler. "
                      "No model/checkpoint/protocol change; original failure preserved."))
        for cid in ("AdaRES-Y52", "AdaRES-Y54"):
            candidates[cid]["evidence"].extend(refs)

        old = "pipeline/analyses/y56_broader_incidence_" + TOKEN + "_program.py"
        tree = ast.parse((ROOT / old).read_text(encoding="utf-8"))
        index = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "indices" for t in n.targets))
        replacement = ast.parse("""
seen_scenes = {row['scan_id'] for row in FROZEN['scenes']}
indices = []
assert len(dataset.scene_inputs) == len(dataset)
for dataset_index, chunk in enumerate(dataset.scene_inputs):
    scene = str(chunk[0]['scene_id'])
    assert all(str(row['scene_id']) == scene for row in chunk)
    if scene not in seen_scenes and len(chunk) == 4:
        seen_scenes.add(scene)
        indices.append(dataset_index)
    if len(indices) == 32:
        break
""").body
        tree.body[index:index + 1] = replacement
        dataset_sources = scoped.tick.load(ROOT / (
            "pipeline/analyses/y57_gt_binding_result_ef2c7bd97d37400ea1b1a32da2909760.json"))["dataset_sources"]
        guards = ast.parse(
            "for path, expected in " + repr(dataset_sources) + ".items():\n"
            "    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected\n").body
        tree.body[index:index] = guards
        old_log = "pipeline/analyses/y56_broader_incidence_" + TOKEN + ".log"
        refs = save_program(
            "y56_broader_unique", ast.unparse(ast.fix_missing_locations(tree)) + "\n",
            dict(
                id="AdaRES-Y56", original_program=scoped.ref(old), failed_log=scoped.ref(old_log),
                failure_scope="diagnostic sampling-unit assumption, not hypothesis",
                reason="Language chunks repeat scenes; index exclusion did not exclude old scenes. "
                       "The old post-capture32unique-scene assertion failed; no completed result.",
                sample="First language chunk of exactly4annotations for each of first32distinct "
                       "TRAIN scenes in metadata order excluding all original8scene IDs. Metadata "
                       "only, frozen before predictions; all7layers/Q128/batch2 unchanged.",
                disposition="If zero exact copies, complete scoped diagnostic without claiming "
                            "population impossibility, rejecting counterfactual algebra or implementing correction."))
        candidates["AdaRES-Y56"]["evidence"].extend([scoped.ref(old_log), *refs])
        candidates["AdaRES-Y56"].setdefault("diagnostic_revision_history", []).append(dict(
            scope="prediction_blind_sampling_unit",
            failure_evidence=[scoped.ref(old_log)], replacement=refs,
            preregistered_at=datetime.now().astimezone().isoformat(), result="pending"))

        old = "pipeline/analyses/y57_full_input_signal_" + TOKEN + "_program.py"
        transform = FullSecondMoment()
        tree = transform.visit(ast.parse((ROOT / old).read_text(encoding="utf-8")))
        assert transform.changes == 3, transform.changes
        refs = save_program(
            "y57_vector_signal", ast.unparse(ast.fix_missing_locations(tree)) + "\n",
            dict(
                id="AdaRES-Y57", previous_program=scoped.ref(old),
                scalar_result=scoped.ref("pipeline/analyses/y57_full_input_signal_" + TOKEN + ".json"),
                scope="Diagnostic measurement revision1, not a production implementation revision.",
                reason="Original hypothesis concerns per-channel second moments. Scalar L2 "
                       "discards orientation; fixed full-input scalar test gave mean MSE gain "
                       "-.0003759429 with interval crossing0. Do not tune ridge or select channels.",
                change="Use ALL256diagonal variance channels instead of their L2 norm, "
                       "corresponding exactly to original moment-aliasing witness. Same baseline "
                       "all769head inputs+confidence, same fixed unit ridge/LOSO/no clipping. "
                       "Same-capacity negative cyclic shift1 along QUERY axis only.",
                controls="Same8scenes/32expressions, final layer6 primary, same checkpoint/source/GT; "
                         "train-fold standardization, no feature/layer/alpha/seed scan.",
                prediction="Aligned full second moments may add information in this fixed "
                           "function class. Null/reverse preserved; no efficacy/admission threshold.",
                limitations="Adaptive follow-up after scalar null, not independent confirmation "
                            "or an exhaustive conditional independence test. No formal model yet."))
        candidate = candidates["AdaRES-Y57"]
        candidate["evidence"].extend([
            scoped.ref("pipeline/analyses/y57_full_input_signal_" + TOKEN + ".json"),
            scoped.ref("pipeline/analyses/y57_full_input_signal_" + TOKEN + ".log"), *refs])
        candidate.setdefault("diagnostic_revision_history", []).append(dict(
            scope="all_channel_second_moment_not_scalar", result="pending",
            prior_negative=scoped.ref("pipeline/analyses/y57_full_input_signal_" + TOKEN + ".json"),
            replacement=refs, preregistered_at=datetime.now().astimezone().isoformat()))
        candidate["next_action"] = (
            "Execute frozen y57_vector_signal same-ID all-channel diagnostic; retain scalar "
            "negative and all scene folds. No formal implementation/admission from association alone.")
        scoped.lease(args.token)
        scoped.tick.save(ROOT / "pipeline/research_work.json", work)
        print("Froze reader-key repair, scene-unit repair and all-channel moment diagnostic.")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
