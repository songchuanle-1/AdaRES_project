"""Prepare an independent nonlinear full-token input before original language norm."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y96_seed_query_skip import NAMES, PARENT


CID = "AdaRES-Y97"
PACKAGE = "adares_y97_language_quadratic_input"
MODULE = '''class LanguageQuadraticInput(nn.Module):
    """Keep the original token projection and add a full-channel quadratic path."""

    def __init__(self, input_dim, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim, d_model))

    def forward(self, projected, raw_tokens):
        return projected + raw_tokens.square() @ self.weight


'''
EDITS = (
    ("class DEC(nn.Module):", MODULE + "class DEC(nn.Module):"),
    ("        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)",
     "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n"
     "        self.language_quadratic = LanguageQuadraticInput(self.lang_proj.in_features, d_model)"),
    ("        lang_feats = self.lang_proj(lang_feats)",
     "        lang_feats = self.language_quadratic(self.lang_proj(lang_feats), lang_feats)"),
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
    anchor = anchor.replace("visual_query_init.", "language_quadratic.")
    core = (ROOT / "scripts/y97_observer_core.py").read_text(encoding="utf-8")
    tail = (ROOT / "scripts/probe_y96_cpu_tail.py").read_text(encoding="utf-8")
    tail = tail.replace("seed_query_skip", "language_quadratic")
    tail = once(tail, "        value = decoder(**feed)",
        "        state['raw_expected'] = feed['lang_feats']\n"
        "        state['mask_expected'] = feed['lang_masks']\n        value = decoder(**feed)")
    tail = tail.replace("additional_parameters=65536", "additional_parameters=196608")
    tail = tail.replace("original_sampled_seed_and_generator_consumers_verified",
                        "original_raw_tokens_and_projection_consumers_verified")
    tail = tail.replace("initial_and_first_coarse_consumers_verified",
                        "original_language_norm_sampling_and_first_self_consumers_verified")
    tail = tail.replace("nonzero_seed_skip_changes_initial_and_final_masks",
                        "nonzero_language_input_changes_initial_and_final_masks")
    tail = tail.replace("CPU_initialization_paths", "CPU_language_paths")
    cpu = prefix + (ROOT / "scripts/probe_y97_cpu_controls.py").read_text(encoding="utf-8")
    cpu += "\n" + anchor + "\n" + tail.replace("# SHARED_OBSERVER_CORE", core)
    native, marker, _ = (ROOT / "scripts/probe_y83_native_smoke.py").read_text(encoding="utf-8").partition("losses=[];")
    assert marker
    native = native.replace("'AdaRES-Y83'", repr(CID)).replace("164632197", "164826245")
    native += "\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n"
    native += "head=model.dec.indi_embedding;residual=model.dec.language_quadratic;original_prediction=model.dec.prediction_head\n"
    tail = (ROOT / "scripts/probe_y96_native_tail.py").read_text(encoding="utf-8")
    tail = once(tail, "    reset_state()\n", "    reset_state()\n"
        "    state['raw_expected'] = args[3]\n    state['mask_expected'] = args[4]\n")
    tail = tail.replace("initialization_paths=path_rows, initialization_calls=2",
                        "language_paths=path_rows, language_adapter_calls=2")
    tail = tail.replace("generator_layer_calls=10", "original_projection_calls=2, original_language_norm_calls=2")
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
        candidate = next(row for row in work["candidates"] if row["id"] == cid)
        assert candidate["stage"] == "blocked" and candidate["blocker_kind"] == "exclusive_gpu_slot"
        assert not candidate.get("native_smoke_result")
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
            raw = b"# AdaRES-Y97: full-channel token quadratic input before original language norm.\nfrom .ipdn import IPDN\n"
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
    program = ROOT / ("pipeline/checks/y97_cpu_" + token + "_program.py")
    native_path = ROOT / ("pipeline/checks/y97_native_" + token + "_body.py")
    history = ROOT / ("pipeline/checks/y97_targeted_history_" + token + ".json")
    assert not any(path.exists() for path in (program, native_path, history))
    program.write_text("PAYLOAD=" + repr(payload) + "\n" + cpu, encoding="utf-8", newline="\n")
    native_path.write_text(native, encoding="utf-8", newline="\n")
    ids = {"AdaRES-Y23", "AdaRES-Y26", "AdaRES-Y33", "AdaRES-Y35", "AdaRES-Y53", "AdaRES-Y68",
           "AdaRES-Y74", "AdaRES-Y75", "AdaRES-Y79", "AdaRES-Y88", "AdaRES-Y92", "AdaRES-Y93", "AdaRES-Y96"}
    tick.save(history, dict(complete=True, scope="Targeted token-input and consumer comparison, not novelty proof.",
        matches=[{key: row.get(key) for key in ("id", "stage", "hypothesis", "matched_control")}
                 for row in work["candidates"] if row["id"] in ids], parent_source=ref(parent / "dec.py")))
    support = [ref("scripts/" + name) for name in ("y97_observer_core.py", "probe_y97_cpu_controls.py",
        "probe_y96_cpu_tail.py", "probe_y96_native_tail.py")]
    work["candidates"].append(dict(
        id=CID, profile="experiment_first", stage="implementation", comparison_baseline="AdaRES-W2",
        implementation_package=PACKAGE, config=payload["variant"]["config"],
        hypothesis="Original decoder input language uses a full768-to256 linear projection before its consumed "
            "LayerNorm. Adding a zero-initialized projection of the SAME raw token-channel squares may ease "
            "nonlinear token fitting before sampling and six refinements. Contextual embeddings/downstream "
            "networks already have nonlinear capacity; no missing information, defect, novelty or benefit is presumed.",
        matched_control="Independent W2. L=lang_norm(lang_proj(T)+T.square()@zeroW768x256), once for all original "
            "tokens/all768 channels; +196608/total164826245/no RNG. Keep original projection/bias and SAME active "
            "language norm/epsilon, original token masks/frozen text encoder modes, all input features/2D/backbone/"
            "sampling algorithm/query count/coordinates/SCG/k/gamma/hard-gate algorithm/static masks/six "
            "refinements/shared confidence/targets/loss/selector/seed1999/fixed30/native YAML. Sampling values "
            "can change only as a consequence of the intended learned token representation; no scans.",
        historical_scope="Y75 multiplies generated queries by pooled projected language; Y79 adds current-query "
            "feedback to recurrent language memory. This changes every original token's initial representation "
            "before the original norm, not pooling, a final confidence readbackY33/Y35, Y23 semantic-loss padding, "
            "rejected Y26 discarded qla norm or Y53 padding premise. Y88/Y92 mask-side quadratic paths and "
            "Y93/Y96 hidden/seed skips are not combined or reopened; text encoder remains frozen.",
        prepared_native_body=ref(native_path), observer_core=ref("scripts/y97_observer_core.py"),
        next_action="Run actual original CPU YAML/fullmodel/releasedbackbone/TRAIN collation plus independent "
            "scalar/no-RNG/token-batch-input-output-channel permutations/sign/zero/token-local controls, exact "
            "seven-stage W2/input-gradient anchors, initial-final causal intervention/exact bypass, same raw "
            "tokens/original projection/norm/sampling/first language-self consumers and two synthetic updates. "
            "Sparse GPU task still mandatory. Preserve Y92/Y93, earlierY94..Y96 and GPU1/Y57 pauses.",
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
