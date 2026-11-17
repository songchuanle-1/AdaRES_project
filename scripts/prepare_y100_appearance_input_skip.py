"""Prepare one independent W2 full-appearance bypass around its visual-memory MLP."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y96_seed_query_skip import PARENT, NAMES
from prepare_y98_appearance_mask_skip import once
from y99_deployment_v2 import Y99, git

CID = "AdaRES-Y100"
PACKAGE = "adares_y100_appearance_input_skip"
MODULE = '''class AppearanceInputSkip(nn.Module):
    """Keep the complete native appearance MLP and bypass it with all raw channels."""

    def __init__(self, input_dim, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim, d_model))

    def forward(self, projected, appearance):
        return projected + appearance @ self.weight


'''
EDITS = (
    ("class DEC(nn.Module):", MODULE + "class DEC(nn.Module):"),
    ("        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)",
     "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n"
     "        self.appearance_input_skip = AppearanceInputSkip(self.input_proj_2d[0].in_features, d_model)"),
    ("        inst_feats_2d = self.input_proj_2d(feats_2d)",
     "        inst_feats_2d = self.appearance_input_skip(self.input_proj_2d(feats_2d), feats_2d)"),
)


def files():
    return [ROOT / "AdaRES/ipdn/model_paper5" / PACKAGE / n for n in NAMES] + [
        ROOT / "AdaRES/configs_song/model_paper5" / (PACKAGE + "_res.yaml")]


def bodies():
    prefix, marker, _ = (ROOT / "scripts/probe_y83_cpu.py").read_text(encoding="utf-8").partition("parameters=sum(")
    assert marker
    _, marker, anchor = (ROOT / "scripts/probe_y84_cpu.py").read_text(encoding="utf-8").partition("kwargs = dict(cfg.model.dec")
    assert marker
    anchor, marker, _ = ("kwargs = dict(cfg.model.dec" + anchor).partition("parent.zero_grad(set_to_none=True);")
    assert marker
    anchor = anchor.replace("visual_query_init.", "appearance_input_skip.")
    controls = (ROOT / "scripts/probe_y98_cpu_controls.py").read_text(encoding="utf-8").replace("AppearanceMaskSkip", "AppearanceInputSkip")
    core = (ROOT / "scripts/y100_observer_core.py").read_text(encoding="utf-8")
    tail = (ROOT / "scripts/probe_y96_cpu_tail.py").read_text(encoding="utf-8").replace("seed_query_skip", "appearance_input_skip")
    tail = once(tail, "        value = decoder(**feed)",
        "        state['raw_appearance_expected'] = feed['feats_2d']\n"
        "        state['raw_spatial_expected'] = feed['x']\n"
        "        state['batch_offsets_expected'] = feed['batch_offsets']\n        value = decoder(**feed)")
    for a, b in (
        ("additional_parameters=65536", "additional_parameters=262144"),
        ("original_sampled_seed_and_generator_consumers_verified", "original_appearance_projection_and_fusion_consumers_verified"),
        ("initial_and_first_coarse_consumers_verified", "all_seven_original_static_masks_verified"),
        ("nonzero_seed_skip_changes_initial_and_final_masks", "nonzero_appearance_input_skip_changes_initial_and_final_masks"),
        ("CPU_initialization_paths", "CPU_appearance_input_paths"),
    ):
        tail = tail.replace(a, b)
    cpu = prefix + controls + "\n" + anchor + "\n" + tail.replace("# SHARED_OBSERVER_CORE", core)
    native, marker, _ = (ROOT / "scripts/probe_y83_native_smoke.py").read_text(encoding="utf-8").partition("losses=[];")
    assert marker
    native = native.replace("'AdaRES-Y83'", repr(CID)).replace("164632197", "164891781")
    native += "\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n"
    native += "head=model.dec.indi_embedding;residual=model.dec.appearance_input_skip;original_prediction=model.dec.prediction_head\n"
    tail = (ROOT / "scripts/probe_y96_native_tail.py").read_text(encoding="utf-8")
    tail = once(tail, "    reset_state()\n", "    reset_state()\n    state['raw_appearance_expected'] = args[6]\n"
        "    state['raw_spatial_expected'] = args[0]\n    state['batch_offsets_expected'] = args[2]\n")
    tail = tail.replace("initialization_paths=path_rows, initialization_calls=2", "appearance_input_paths=path_rows, appearance_input_calls=2")
    tail = tail.replace("generator_layer_calls=10", "original_appearance_projection_calls=2, original_spatial_projection_calls=2, original_sum_norm_calls=2, first_language_memory_consumers=2, original_base_mask_calls=2")
    native += core + "\n" + tail
    for source in (cpu, native):
        ast.parse(source, feature_version=(3, 8))
        assert source.count(core) == 1
    return cpu, native


def prepare(token):
    lease(token, 800)
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert view["experiment_first"]["enabled"] and view["research"]["creation_budget"] >= 1
    assert git("rev-parse", "HEAD").decode().strip() == Y99
    work = tick.load(ROOT / "pipeline/research_work.json")
    assert not any(r["id"] == CID for r in work["candidates"])
    before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
    index = git("ls-files", "--stage", "-z")
    queue = monitor.load_queue(str(ROOT))
    for name, cid, pid in (("server-1", "AdaRES-Y99", 2980836), ("server-2", "AdaRES-Y98", 25962)):
        task = next(r for r in queue["gpu_queues"][name] if r["id"] == cid)
        assert task["status"] == "running" and task["pid"] == pid and task["launch_verified"]
    paths = files()
    parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source = (parent / "dec.py").read_text(encoding="utf-8")
    for old, new in EDITS:
        source = once(source, old, new)
    ast.parse(source, feature_version=(3, 8))
    cpu, native = bodies()
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == "dec.py" else (parent / path.name).read_bytes()
        if path.name == "__init__.py":
            raw = b"# AdaRES-Y100: full raw appearance bypass beside the original visual-memory MLP.\nfrom .ipdn import IPDN\n"
        path.write_bytes(raw)
    paths[-1].write_bytes((ROOT / "AdaRES/configs_song/model_paper5" / (PARENT + "_res.yaml")).read_bytes())
    assert list((ROOT / "AdaRES/configs_song").rglob(paths[-1].name)) == [paths[-1]]
    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)["sha256"])
    payload = dict(variant=dict(id=CID, package=PACKAGE, config=paths[-1].relative_to(ROOT / "AdaRES").as_posix(),
        python_files={p.name: encoded(p) for p in paths[:-1]}, config_data=encoded(paths[-1])),
        parent_config="configs_song/model_paper5/" + PARENT + "_res.yaml", goal_sha256=ref("pipeline/goal.json")["sha256"],
        parent_dec_lf_sha256=hashlib.sha256((parent / "dec.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / "AdaRES/tools/train.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    program = ROOT / ("pipeline/checks/y100_cpu_" + token + "_program.py")
    native_path = ROOT / ("pipeline/checks/y100_native_" + token + "_body.py")
    history = ROOT / ("pipeline/checks/y100_targeted_history_" + token + ".json")
    assert not any(p.exists() for p in (program, native_path, history))
    program.write_text("PAYLOAD=" + repr(payload) + "\n" + cpu, encoding="utf-8", newline="\n")
    native_path.write_text(native, encoding="utf-8", newline="\n")
    ids = {"AdaRES-Y68", "AdaRES-Y71", "AdaRES-Y72", "AdaRES-Y80", "AdaRES-Y84", "AdaRES-Y96", "AdaRES-Y97", "AdaRES-Y98", "AdaRES-Y99"}
    tick.save(history, dict(complete=True, scope="Targeted input/destination comparison, not a novelty proof.",
        matches=[{k: r.get(k) for k in ("id", "stage", "hypothesis", "matched_control")} for r in work["candidates"] if r["id"] in ids],
        parent_source=ref(parent / "dec.py"), distinction="Y100 adds full raw1024 appearance to ORIGINAL input_proj_2d output BEFORE sum_norm and original visual memory. Y98 adds appearance directly to static masks; Y71 uses the product of already projected modalities. W2 independent; no combination, raw-language-square retry, selected feature subset or reopening."))
    work["candidates"].append(dict(id=CID, profile="experiment_first", stage="implementation", comparison_baseline="AdaRES-W2",
        implementation_package=PACKAGE, config=payload["variant"]["config"],
        hypothesis="Original full1024 appearance enters visual memory through three linear layers and two ReLUs before sum_norm. A zero-initialized parallel linear bypass of that SAME complete appearance vector may ease visual-memory fitting while retaining the entire original MLP. Existing projection/downstream attention already carry this information; no missing information, defect, novelty or benefit is presumed.",
        matched_control="Independent W2: appearance=input_proj_2d(raw)+raw@zeroW1024x256 before SAME original sum_norm(spatial+appearance); +262144/total164891781/no RNG. Preserve all1024 raw appearance/all32 spatial/all points, complete original projection/normalization/static x_mask/seven static readouts/query generator/refinement/sampling/SCG/k/gamma/shared confidence/hard-gate algorithm/targets/loss/selector/releasedbackbone/data/seed1999/fixed30/native YAML. Single added structural connection, no feature subset/threshold/temperature/loss/epoch/validation scan.",
        historical_scope="Not Y98 raw appearance-to-static-mask, Y99 spatial-to-mask, Y71 projected cross-modal product, Y96 sampled-seed-query bypass, or Y97 raw-language-squares. Preserve every closed/rejected route and numerical hold; no combination/reopening/retry/relaxed controls.",
        prepared_native_body=ref(native_path), observer_core=ref("scripts/y100_observer_core.py"),
        next_action="Run first exact original CPU YAML/fullmodel/releasedbackbone/TRAIN collation and scalar float32/64 no-RNG/permutation/sign/row-local controls, seven-stage W2 anchors/input gradients, causal bypass and two finite synthetic decoder updates; native GPU TRAIN still requires authorized idle exclusive GPU0 and actual two-step native PASS before admission.",
        evidence=[ref(p) for p in paths] + [ref(program), ref(native_path), ref(history), ref(__file__), ref("scripts/y100_observer_core.py")]))
    tick.save(ROOT / "pipeline/research_work.json", work)
    assert all(ref(p) == r for p, r in before.items()) and git("ls-files", "--stage", "-z") == index
    print(json.dumps(dict(candidate=CID, program=ref(program), native_body=ref(native_path), CPU_executed=False, native_executed=False, admitted=False)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True)
    token = p.parse_args().token
    lease(token, 800)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        prepare(token)
    finally:
        writer.close()
