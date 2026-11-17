"""Prepare one independent all-query-mean bypass of local self attention."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


CID = "AdaRES-Y95"
PACKAGE = "adares_y95_query_set_mean"
PARENT = "adares_w2_pointmass_competition"
NAMES = ("__init__.py", "backbone.py", "dec.py", "ipdn.py", "loss.py", "sample_model.py")
MODULE = '''class QuerySetMeanSkip(nn.Module):
    """Broadcast existing all-query context beside the original local update."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, self_query, language_query):
        context = language_query.mean(dim=1, keepdim=True)
        return self_query + language_query + context @ self.weight


'''


def files():
    return [ROOT / "AdaRES/ipdn/model_paper5" / PACKAGE / name for name in NAMES] + [
        ROOT / "AdaRES/configs_song/model_paper5" / (PACKAGE + "_res.yaml")]


def once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def observer():
    source = (ROOT / "scripts/y87_observer_core.py").read_text(encoding="utf-8")
    source = once(source, "(args[0].double() * args[1].double()) @ module.weight.double()",
                  "args[1].double().mean(1, keepdim=True) @ module.weight.double()")
    source = source.replace("float64_sum_product_projection_reference", "float64_all_query_mean_projection_reference")
    source = once(source, "    assert state['indicators'] == list(range(7))",
                  "    assert state['coarse'] == 6\n    assert state['indicators'] == list(range(7))")
    source = once(source, "        indicators=[], norms=[])",
                  "        indicators=[], norms=[], coarse=0)")
    source = once(source, "def before_indicator(module, args):", '''
def before_coarse(module, args):
    index = state['coarse']
    assert module is model.dec.qia_layers[index] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def before_indicator(module, args):''')
    source = once(source, "    hooks.extend(module.register_forward_pre_hook(before_ffn)",
                  "    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)\n"
                  "    hooks.extend(module.register_forward_pre_hook(before_ffn)")
    source = once(source, "fusion_calls=6, original_norm_consumption_calls=6,",
                  "fusion_calls=6, coarse_gate_consumers=6, original_norm_consumption_calls=6,")
    ast.parse(source, feature_version=(3, 8))
    return source


def bodies(core):
    from prepare_y87_self_language_product import bodies as previous_bodies
    previous_cpu, previous_native = previous_bodies()
    prefix, marker, _ = previous_cpu.partition("parameters=sum(")
    assert marker
    _, marker, decoder = previous_cpu.partition("kwargs = dict(cfg.model.dec")
    assert marker
    decoder, marker2, _ = (marker + decoder).partition('"""Original self/language tensors')
    assert marker2
    decoder = decoder.replace("self_language_product", "query_set_skip")
    _, marker, tail = previous_cpu.partition("hooks=install_hooks();model.dec.prediction_head=checked_prediction")
    assert marker
    tail = marker + tail
    tail = tail.replace("original_self_language_and_norm_ffn_consumption_verified",
                        "original_self_language_and_norm_ffn_consumption_verified")
    cpu = prefix + (ROOT / "scripts/probe_y95_cpu_controls.py").read_text(encoding="utf-8")
    cpu += "\n" + decoder + "\n" + core + "\n" + tail
    native, marker, _ = previous_native.partition('"""Original self/language tensors')
    assert marker
    _, marker, tail = previous_native.partition("class FirstTwoBatches:")
    assert marker
    native += core + "\n" + marker + tail
    native = native.replace("'AdaRES-Y87'", repr(CID)).replace("self_language_product", "query_set_skip")
    for value in (cpu, native):
        ast.parse(value, feature_version=(3, 8))
    return cpu, native


def prepare(token):
    lease(token, 800)
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert view["experiment_first"]["enabled"] and view["research"]["creation_budget"] >= 1
    work = tick.load(ROOT / "pipeline/research_work.json")
    assert not any(row["id"] == CID for row in work["candidates"])
    earlier = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y94")
    assert earlier["stage"] == "blocked" and earlier["blocker_kind"] == "exclusive_gpu_slot"
    paths = files()
    parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and not any(path.exists() for path in paths)
    source = (parent / "dec.py").read_text(encoding="utf-8")
    for old, new in (
        ("class DEC(nn.Module):", MODULE + "class DEC(nn.Module):"),
        ("        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)",
         "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n"
         "        self.query_set_skip = QuerySetMeanSkip(d_model)"),
        ("            query = self.norm_qqa(query_self + query_qla)",
         "            query = self.norm_qqa(self.query_set_skip(query_self, query_qla))"),
    ):
        source = once(source, old, new)
    ast.parse(source, feature_version=(3, 8))
    core = observer()
    cpu, native = bodies(core)
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == "dec.py" else (parent / path.name).read_bytes()
        if path.name == "__init__.py":
            raw = b"# AdaRES-Y95: all-query-mean bypass before original recurrent normalization.\nfrom .ipdn import IPDN\n"
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
    program = ROOT / ("pipeline/checks/y95_cpu_" + token + "_program.py")
    native_path = ROOT / ("pipeline/checks/y95_native_" + token + "_body.py")
    core_path = ROOT / "scripts/y95_observer_core.py"
    history = ROOT / ("pipeline/checks/y95_targeted_history_" + token + ".json")
    assert not any(path.exists() for path in (program, native_path, core_path, history))
    program.write_text("PAYLOAD=" + repr(payload) + "\n" + cpu, encoding="utf-8", newline="\n")
    native_path.write_text(native, encoding="utf-8", newline="\n")
    core_path.write_text(core, encoding="utf-8", newline="\n")
    ids = {"AdaRES-Y25", "AdaRES-Y26", "AdaRES-Y33", "AdaRES-Y57", "AdaRES-Y65", "AdaRES-Y66",
           "AdaRES-Y79", "AdaRES-Y80", "AdaRES-Y81", "AdaRES-Y84", "AdaRES-Y87", "AdaRES-Y94"}
    tick.save(history, dict(complete=True, scope="Targeted consumer/mechanism comparison; not novelty proof.",
        matches=[{key: row.get(key) for key in ("id", "stage", "hypothesis", "matched_control")}
                 for row in work["candidates"] if row["id"] in ids], parent_source=ref(parent / "dec.py")))
    work["candidates"].append(dict(
        id=CID, profile="experiment_first", stage="implementation", comparison_baseline="AdaRES-W2",
        implementation_package=PACKAGE, config=payload["variant"]["config"],
        hypothesis="Original neighborhood-restricted query self attention refines language-conditioned queries. "
            "A parallel shared projection of their full query-set mean into the existing self/language sum "
            "may ease fitting a global query context while retaining the local attention path. Self attention "
            "already provides context; no missing information, defect, novelty or trained benefit is presumed.",
        matched_control="Independent W2. fused=self_query+query_qla+mean_all128(query_qla)@zeroW256x256, "
            "six calls before SAME original norm_qqa and original stage FFNs; +65536/no RNG/total164695173. "
            "All queries/channels stay undetached; global context is a separate path, not changed attention "
            "neighbors. Preserve original local self-attention/SCG exclusions/k/gamma, coarse/fine half-average, "
            "all original inputs/data/2D/backbone/sampling/static masks/head sharing/targets/loss/selector/"
            "seed1999/fixed30/native YAML. No threshold/temperature/loss/feature/epoch/validation scan.",
        historical_scope="Y66 uses mask-overlap-weighted raw peers before visual attention; this uses an unweighted "
            "full current query_qla mean beside the later local self-attention. Y79/Y80 update language/visual "
            "memories, Y81 updates mask features, Y84 uses scene visual mean once for initialization, and Y87 "
            "uses same-slot self/language products without global pooling. Y94 only biases self-attention "
            "geometry. None is combined or reopened. Rejected Y25 coarse-visual gate and Y26 discarded norm "
            "reconnection, final-confidence pooling and separately paused Y57 second moments remain untouched.",
        prepared_native_body=ref(native_path), observer_core=ref(core_path),
        next_action="Execute original CPU YAML/full model/released backbone/real TRAIN collation plus float32/64 "
            "scalar/no-RNG/query-channel permutations/replication/unchanged-focal changed-peer controls, exact "
            "seven-stage W2/input-gradient anchors, causal intervention/bypass and two synthetic decoder updates. "
            "Actual sparse GPU two-step task remains mandatory; preserve Y92/Y93, earlier Y94 and both pauses.",
        evidence=[ref(path) for path in paths] + [ref(program), ref(native_path), ref(core_path), ref(history),
            ref(__file__), ref("scripts/probe_y95_cpu_controls.py"), ref("scripts/prepare_y87_self_language_product.py"),
            ref("scripts/y87_observer_core.py")]))
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
