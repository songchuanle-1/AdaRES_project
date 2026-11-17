"""Prepare a direct full existing-appearance path into original static masks."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y96_seed_query_skip import PARENT, NAMES


CID = "AdaRES-Y98"
PACKAGE = "adares_y98_appearance_mask_skip"
MODULE = '''class AppearanceMaskSkip(nn.Module):
    """Retain spatial mask features and add all existing appearance channels."""

    def __init__(self, input_dim, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim, d_model))

    def forward(self, mask_base, appearance):
        return mask_base + appearance @ self.weight


'''
EDITS = (
    ("class DEC(nn.Module):", MODULE + "class DEC(nn.Module):"),
    ("        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)",
     "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n"
     "        self.appearance_mask_skip = AppearanceMaskSkip(self.input_proj_2d[0].in_features, d_model)"),
    ("        mask_feats = self.x_mask(x)",
     "        mask_feats = self.appearance_mask_skip(self.x_mask(x), feats_2d)"),
)


def files():
    return [ROOT / "AdaRES/ipdn/model_paper5" / PACKAGE / name for name in NAMES] + [
        ROOT / "AdaRES/configs_song/model_paper5" / (PACKAGE + "_res.yaml")]


def once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def bodies():
    prefix, marker, _ = (ROOT / "scripts/probe_y83_cpu.py").read_text(encoding="utf-8").partition("parameters=sum(")
    assert marker
    _, marker, anchor = (ROOT / "scripts/probe_y84_cpu.py").read_text(encoding="utf-8").partition("kwargs = dict(cfg.model.dec")
    assert marker
    anchor, marker, _ = ("kwargs = dict(cfg.model.dec" + anchor).partition("parent.zero_grad(set_to_none=True);")
    assert marker
    anchor = anchor.replace("visual_query_init.", "appearance_mask_skip.")
    core = (ROOT / "scripts/y98_observer_core.py").read_text(encoding="utf-8")
    tail = (ROOT / "scripts/probe_y96_cpu_tail.py").read_text(encoding="utf-8")
    tail = tail.replace("seed_query_skip", "appearance_mask_skip")
    tail = once(tail, "    changed = decoder(**inputs)",
        "    changed = decoder(**inputs)\n"
        "    assert torch.equal(parts(changed)[0]['scores'], parts(baseline)[0]['scores'])")
    tail = once(tail, "        value = decoder(**feed)",
        "        state['raw_appearance_expected'] = feed['feats_2d']\n"
        "        state['batch_offsets_expected'] = feed['batch_offsets']\n        value = decoder(**feed)")
    for old, new in (
        ("additional_parameters=65536", "additional_parameters=262144"),
        ("original_sampled_seed_and_generator_consumers_verified", "original_appearance_and_base_mask_consumers_verified"),
        ("initial_and_first_coarse_consumers_verified", "all_seven_static_mask_consumers_verified"),
        ("nonzero_seed_skip_changes_initial_and_final_masks", "nonzero_appearance_skip_changes_initial_and_final_masks"),
        ("CPU_initialization_paths", "CPU_appearance_mask_paths"),
    ):
        tail = tail.replace(old, new)
    cpu = prefix + (ROOT / "scripts/probe_y98_cpu_controls.py").read_text(encoding="utf-8")
    cpu += "\n" + anchor + "\n" + tail.replace("# SHARED_OBSERVER_CORE", core)
    native, marker, _ = (ROOT / "scripts/probe_y83_native_smoke.py").read_text(encoding="utf-8").partition("losses=[];")
    assert marker
    native = native.replace("'AdaRES-Y83'", repr(CID)).replace("164632197", "164891781")
    native += "\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n"
    native += "head=model.dec.indi_embedding;residual=model.dec.appearance_mask_skip;original_prediction=model.dec.prediction_head\n"
    tail = (ROOT / "scripts/probe_y96_native_tail.py").read_text(encoding="utf-8")
    tail = once(tail, "    reset_state()\n",
        "    reset_state()\n    state['raw_appearance_expected'] = args[6]\n"
        "    state['batch_offsets_expected'] = args[2]\n")
    for old, new in (
        ("initialization_paths=path_rows, initialization_calls=2", "appearance_mask_paths=path_rows, appearance_mask_calls=2"),
        ("generator_layer_calls=10", "original_appearance_projection_calls=2, original_base_mask_calls=2"),
        ("static_mask_features_preserved=True", "original_base_mask_and_static_reuse_preserved=True"),
    ):
        tail = tail.replace(old, new)
    native += core + "\n" + tail
    for source in (cpu, native):
        ast.parse(source, feature_version=(3, 8))
        assert source.count(core) == 1
    return cpu, native


def prepare(token):
    lease(token, 800)
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert view["experiment_first"]["enabled"] and view["research"]["creation_budget"] >= 1
    work = tick.load(ROOT / "pipeline/research_work.json")
    assert not any(row["id"] == CID for row in work["candidates"])
    for cid in ("AdaRES-Y94", "AdaRES-Y95", "AdaRES-Y96"):
        earlier = next(row for row in work["candidates"] if row["id"] == cid)
        assert earlier["stage"] == "blocked" and earlier["blocker_kind"] == "exclusive_gpu_slot"
        assert not earlier.get("native_smoke_result")
    paths = files()
    parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and not any(path.exists() for path in paths)
    source = (parent / "dec.py").read_text(encoding="utf-8")
    for old, new in EDITS:
        source = once(source, old, new)
    ast.parse(source, feature_version=(3, 8))
    cpu, native = bodies()
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == "dec.py" else (parent / path.name).read_bytes()
        if path.name == "__init__.py":
            raw = b"# AdaRES-Y98: full existing 2D appearance residual into static mask features.\nfrom .ipdn import IPDN\n"
        path.write_bytes(raw)
    paths[-1].write_bytes((ROOT / "AdaRES/configs_song/model_paper5" / (PARENT + "_res.yaml")).read_bytes())
    assert list((ROOT / "AdaRES/configs_song").rglob(paths[-1].name)) == [paths[-1]]

    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)["sha256"])

    payload = dict(
        variant=dict(id=CID, package=PACKAGE, config=paths[-1].relative_to(ROOT / "AdaRES").as_posix(),
            python_files={path.name: encoded(path) for path in paths[:-1]}, config_data=encoded(paths[-1])),
        parent_config="configs_song/model_paper5/" + PARENT + "_res.yaml",
        goal_sha256=ref("pipeline/goal.json")["sha256"],
        parent_dec_lf_sha256=hashlib.sha256((parent / "dec.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / "AdaRES/tools/train.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    program = ROOT / ("pipeline/checks/y98_cpu_" + token + "_program.py")
    native_path = ROOT / ("pipeline/checks/y98_native_" + token + "_body.py")
    history = ROOT / ("pipeline/checks/y98_targeted_history_" + token + ".json")
    assert not any(path.exists() for path in (program, native_path, history))
    program.write_text("PAYLOAD=" + repr(payload) + "\n" + cpu, encoding="utf-8", newline="\n")
    native_path.write_text(native, encoding="utf-8", newline="\n")
    ids = {"AdaRES-Y68", "AdaRES-Y71", "AdaRES-Y72", "AdaRES-Y80", "AdaRES-Y81", "AdaRES-Y85",
           "AdaRES-Y88", "AdaRES-Y92", "AdaRES-Y96", "AdaRES-Y97"}
    tick.save(history, dict(complete=True, scope="Targeted input/consumer comparison, not novelty proof.",
        matches=[{key: row.get(key) for key in ("id", "stage", "hypothesis", "matched_control")}
                 for row in work["candidates"] if row["id"] in ids], parent_source=ref(parent / "dec.py"),
        duplicated_route_not_created="The initial projected 3D/2D product is exactly existing Y71; it was "
            "recognized before any new scientific ID/code/test. Y98 instead adds all raw1024 existing "
            "appearance channels to the static mask base, not the initial visual fusion."))
    support = [ref("scripts/" + name) for name in ("y98_observer_core.py", "probe_y98_cpu_controls.py",
        "probe_y96_cpu_tail.py", "probe_y96_native_tail.py")]
    work["candidates"].append(dict(
        id=CID, profile="experiment_first", stage="implementation", comparison_baseline="AdaRES-W2",
        implementation_package=PACKAGE, config=payload["variant"]["config"],
        hypothesis="Original static mask features come from sparse spatial features, while the full existing "
            "2D appearance features influence the query/visual-memory path. A separate learned direct full "
            "appearance residual into static mask features may ease appearance-sensitive mask fitting. "
            "Query interactions already carry appearance; no missing information, defect, novelty or benefit "
            "is presumed.",
        matched_control="Independent W2. mask_static=x_mask(x)+raw_existing_2D@zeroW1024x256 once before "
            "the SAME original get_batches and all seven mask readouts; +262144/total164891781/no RNG. "
            "Retain the complete original x_mask and input_proj_2d/visual fusion, all1024 existing appearance "
            "channels/all original sparse features/points/query channels, original query norms/scores/shared "
            "confidence, coordinates/sampling/SCG/coarse-fine half-average/self-language sum/six refinements/"
            "hard-gate algorithm/targets-loss-selector/releasedbackbone/seed1999/fixed30/native YAML. "
            "No new data, feature/threshold/temperature/loss/epoch or validation scan.",
        historical_scope="Not Y71 initial projected 3D/2D product before sum_norm, Y72 recurrent refined-visual "
            "mask additions, Y81 current-query mask attention, Y68 language mask modulation, Y85 mask norm, "
            "Y88 query-side quadratic, Y92 mask-side quadratic or Y96 sampled-seed initial-query skip. "
            "No combination or reopening. Y97 original and revision1 numerical guards/failures remain untouched.",
        prepared_native_body=ref(native_path), observer_core=ref("scripts/y98_observer_core.py"),
        next_action="Execute original CPU YAML/fullmodel/releasedbackbone/TRAIN collation plus independent "
            "scalar/no-RNG/row-channel permutation/zero/sign/row-local/both-input controls, seven-stage "
            "W2/RNG/weight/input-gradient anchors, initial-mask response with exact initial-score preservation "
            "and exact bypass, all raw1024/source/base/batching/shared static-mask consumers, two finite "
            "synthetic updates. Actual sparse GPU task remains mandatory; earlier Y94/Y95/Y96 precede Y98.",
        evidence=[ref(path) for path in paths] + [ref(program), ref(native_path), ref(history), ref(__file__)] + support))
    tick.save(ROOT / "pipeline/research_work.json", work)
    print(json.dumps(dict(candidate=CID, program=ref(program), native_body=ref(native_path),
                          CPU_executed=False, native_executed=False, admitted=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    lease(args.token, 800)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        prepare(args.token)
    finally:
        writer.close()
