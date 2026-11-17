"""Prepare an independent bypass of the six complete original recurrent query FFNs."""
import argparse, ast, base64, hashlib, json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y96_seed_query_skip import PARENT, NAMES
from prepare_y98_appearance_mask_skip import once
from y100_deployment import Y100, git

CID = "AdaRES-Y101"
PACKAGE = "adares_y101_query_ffn_skip"
MODULE = '''class QueryFFNSkip(nn.Module):
    """Retain the original full FFN and bypass it with the same current query."""
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, original_ffn, current_query):
        return original_ffn + current_query @ self.weight


'''
EDITS = (
    ("class DEC(nn.Module):", MODULE + "class DEC(nn.Module):"),
    ("        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)",
     "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.query_ffn_skip = QueryFFNSkip(d_model)"),
    ("            query = self.qqa_ffn_layers[i](query)",
     "            query = self.query_ffn_skip(self.qqa_ffn_layers[i](query), query)"),
)


def files():
    return [ROOT / "AdaRES/ipdn/model_paper5" / PACKAGE / n for n in NAMES] + [ROOT / "AdaRES/configs_song/model_paper5" / (PACKAGE + "_res.yaml")]


def bodies():
    prefix, marker, _ = (ROOT / "scripts/probe_y83_cpu.py").read_text(encoding="utf-8").partition("parameters=sum(")
    assert marker
    _, marker, anchor = (ROOT / "scripts/probe_y84_cpu.py").read_text(encoding="utf-8").partition("kwargs = dict(cfg.model.dec")
    assert marker
    anchor, marker, _ = ("kwargs = dict(cfg.model.dec" + anchor).partition("parent.zero_grad(set_to_none=True);")
    assert marker
    anchor = anchor.replace("visual_query_init.", "query_ffn_skip.")
    controls = (ROOT / "scripts/probe_y96_cpu_controls.py").read_text(encoding="utf-8").replace("SeedQuerySkip", "QueryFFNSkip")
    controls = controls.replace("independent_scalar_seed_projection", "independent_scalar_full_query_projection").replace("seed_intervention_is_slot_local", "current_query_intervention_is_slot_local")
    core = (ROOT / "scripts/y101_observer_core.py").read_text(encoding="utf-8")
    tail = (ROOT / "scripts/probe_y96_cpu_tail.py").read_text(encoding="utf-8").replace("seed_query_skip", "query_ffn_skip")
    tail = once(tail, "    assert not torch.allclose(parts(changed)[0]['masks'], parts(baseline)[0]['masks'])",
        "    assert all(torch.equal(parts(changed)[0][k], parts(baseline)[0][k]) for k in ('masks','scores','indis'))\n    assert not torch.allclose(parts(changed)[1]['masks'], parts(baseline)[1]['masks'])")
    tail = once(tail, "        value = decoder(**feed)", "        state['batch_offsets_expected'] = feed['batch_offsets']\n        value = decoder(**feed)")
    for a, b in (
        ("original_sampled_seed_and_generator_consumers_verified", "all_six_complete_original_query_FFNs_verified"),
        ("initial_and_first_coarse_consumers_verified", "all_six_shared_current_query_bypasses_verified"),
        ("nonzero_seed_skip_changes_initial_and_final_masks", "initial_readout_exact_and_refined_masks_change"),
        ("CPU_initialization_paths", "CPU_query_FFN_paths"),
    ):
        tail = tail.replace(a, b)
    cpu = prefix + controls + "\n" + anchor + "\n" + tail.replace("# SHARED_OBSERVER_CORE", core)
    native, marker, _ = (ROOT / "scripts/probe_y83_native_smoke.py").read_text(encoding="utf-8").partition("losses=[];")
    assert marker
    native = native.replace("'AdaRES-Y83'", repr(CID)).replace("164632197", "164695173")
    native += "\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n"
    native += "head=model.dec.indi_embedding;residual=model.dec.query_ffn_skip;original_prediction=model.dec.prediction_head\n"
    tail = (ROOT / "scripts/probe_y96_native_tail.py").read_text(encoding="utf-8")
    tail = once(tail, "    reset_state()\n", "    reset_state()\n    state['batch_offsets_expected'] = args[2]\n")
    tail = tail.replace("initialization_paths=path_rows, initialization_calls=2", "query_FFN_paths=path_rows, query_FFN_skip_calls=12")
    tail = tail.replace("generator_layer_calls=10", "original_query_FFN_calls=12")
    native += core + "\n" + tail
    for source in (cpu, native):
        ast.parse(source, feature_version=(3, 8))
        assert source.count(core) == 1
    return cpu, native


def prepare(token):
    lease(token, 800)
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert view["experiment_first"]["enabled"] and view["research"]["creation_budget"] >= 1
    assert git("rev-parse", "HEAD").decode().strip() == Y100
    work = tick.load(ROOT / "pipeline/research_work.json")
    assert not any(r["id"] == CID for r in work["candidates"])
    before = {p: ref(p) for p in ("pipeline/queue.json", "pipeline/running.json", "pipeline/goal.json", "SERVERS.yaml")}
    index = git("ls-files", "--stage", "-z")
    q = monitor.load_queue(str(ROOT))
    for name, cid, pid in (("server-1", "AdaRES-Y99", 2980836), ("server-2", "AdaRES-Y100", 19305)):
        task = next(r for r in q["gpu_queues"][name] if r["id"] == cid)
        assert task["status"] == "running" and task["pid"] == pid and task["launch_verified"]
    paths = files()
    parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source = (parent / "dec.py").read_text(encoding="utf-8")
    for a, b in EDITS:
        source = once(source, a, b)
    ast.parse(source, feature_version=(3, 8))
    cpu, native = bodies()
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == "dec.py" else (parent / path.name).read_bytes()
        if path.name == "__init__.py":
            raw = b"# AdaRES-Y101: shared current-query bypass beside complete original query FFNs.\nfrom .ipdn import IPDN\n"
        path.write_bytes(raw)
    paths[-1].write_bytes((ROOT / "AdaRES/configs_song/model_paper5" / (PARENT + "_res.yaml")).read_bytes())
    assert list((ROOT / "AdaRES/configs_song").rglob(paths[-1].name)) == [paths[-1]]
    def encoded(p):
        return dict(base64=base64.b64encode(p.read_bytes()).decode(), sha256=ref(p)["sha256"])
    payload = dict(variant=dict(id=CID, package=PACKAGE, config=paths[-1].relative_to(ROOT / "AdaRES").as_posix(), python_files={p.name: encoded(p) for p in paths[:-1]}, config_data=encoded(paths[-1])),
        parent_config="configs_song/model_paper5/" + PARENT + "_res.yaml", goal_sha256=ref("pipeline/goal.json")["sha256"],
        parent_dec_lf_sha256=hashlib.sha256((parent / "dec.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / "AdaRES/tools/train.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    program = ROOT / ("pipeline/checks/y101_cpu_" + token + "_program.py")
    native_path = ROOT / ("pipeline/checks/y101_native_" + token + "_body.py")
    history = ROOT / ("pipeline/checks/y101_targeted_history_" + token + ".json")
    assert not any(p.exists() for p in (program, native_path, history))
    program.write_text("PAYLOAD=" + repr(payload) + "\n" + cpu, encoding="utf-8", newline="\n")
    native_path.write_text(native, encoding="utf-8", newline="\n")
    ids={"AdaRES-Y73","AdaRES-Y83","AdaRES-Y87","AdaRES-Y95","AdaRES-Y96","AdaRES-Y100"}
    tick.save(history, dict(complete=True, scope="Targeted source/input/insertion comparison, not novelty proof.",
        matches=[{k:r.get(k) for k in ('id','stage','hypothesis','matched_control')} for r in work['candidates'] if r['id'] in ids],
        parent_source=ref(parent/'dec.py'), distinction="Y101 bypasses each COMPLETE original current-query FFN with its same undetached pre-FFN query. Y73 instead routes original initial-query-minus-refined anchor; Y87/Y95 act BEFORE norm_qqa, Y83 changes its sharing, Y96 acts once at initial generator. Single connection independent W2; no combination, feature selection, numerical retry or reopening."))
    work['candidates'].append(dict(id=CID,profile="experiment_first",stage="implementation",comparison_baseline="AdaRES-W2",implementation_package=PACKAGE,config=payload['variant']['config'],
        hypothesis="Six original recurrent query FFNs transform current fused queries through nonlinear blocks, residual additions and output norms. A shared zero-initialized full-channel linear bypass of that SAME current pre-FFN query after each complete FFN may ease recurrent query fitting while retaining all original transformations. Existing residual paths already represent this information; no defect, missing information, novelty or benefit is presumed.",
        matched_control="Independent W2: query=original_qqa_FFN_i(current)+current@zeroW256x256 at six original caller sites AFTER complete FFN including its original residual/norm. One shared matrix +65536/total164695173/noRNG. No FFN removal or norm/gate modification. All256 current-query channels/all128 slots remain undetached; preserve original initial prediction/static masks7/full inputs/appearance/sampling/SCG/k/gamma/coarse-fine average/self-language sum/shared confidence/targets/loss/selector/releasedbackbone/data/seed1999fixed30/nativeYAML. No feature/threshold/temperature/loss/epoch/validation scan.",
        historical_scope="Not Y73 initial-query difference anchor, Y83 private fusion norm, Y87 self-language product, Y95 global query mean or Y96 initial seed bypass. Preserve all negative evidence, numerical holds, closed/banned mechanisms and dedicated restoration; no combination/reopening/retry or tolerance relaxation.",
        prepared_native_body=ref(native_path),observer_core=ref('scripts/y101_observer_core.py'),next_action="Run FIRST exact original CPU YAML/fullmodel/releasedbackbone/TRAIN collation; float32/64 scalar/permutation/slot-local/no-RNG controls, seven-stage W2/input-gradient anchors, exact initial readout and causal refined-mask bypass restoration, all six original FFNs/shared projections/consumer gradients and two finite synthetic decoder updates. Actual native sparse GPU task still requires fresh authorized idle exclusive GPU0 and native PASS before admission.",
        evidence=[ref(p) for p in paths]+[ref(program),ref(native_path),ref(history),ref(__file__),ref('scripts/y101_observer_core.py')]))
    tick.save(ROOT/'pipeline/research_work.json',work)
    assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
    print(json.dumps(dict(candidate=CID,program=ref(program),native_body=ref(native_path),CPU_executed=False,native_executed=False,admitted=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);token=p.parse_args().token
    lease(token,800);writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:prepare(token)
    finally:writer.close()
