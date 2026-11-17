"""Prepare an independent sampled-seed bypass around the original query generator."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


CID = "AdaRES-Y96"
PACKAGE = "adares_y96_seed_query_skip"
PARENT = "adares_w2_pointmass_competition"
NAMES = ("__init__.py", "backbone.py", "dec.py", "ipdn.py", "loss.py", "sample_model.py")
MODULE = '''class SeedQuerySkip(nn.Module):
    """Retain the original generated query and its undetached sampled seed."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, generated_query, sampled_seed):
        return generated_query + sampled_seed @ self.weight


'''
EDITS = (
    ("class DEC(nn.Module):", MODULE + "class DEC(nn.Module):"),
    ("        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)",
     "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n"
     "        self.seed_query_skip = SeedQuerySkip(d_model)"),
    ("            query = self.query_generator(sampled_seed)",
     "            query = self.seed_query_skip(self.query_generator(sampled_seed), sampled_seed)"),
    ("            query = self.query_generator(seed_sp)",
     "            query = self.seed_query_skip(self.query_generator(seed_sp), seed_sp)"),
)


def files():
    return [ROOT / "AdaRES/ipdn/model_paper5" / PACKAGE / name for name in NAMES] + [
        ROOT / "AdaRES/configs_song/model_paper5" / (PACKAGE + "_res.yaml")]


def bodies():
    prefix, marker, _ = (ROOT / "scripts/probe_y83_cpu.py").read_text(encoding="utf-8").partition("parameters=sum(")
    assert marker
    source = (ROOT / "scripts/probe_y84_cpu.py").read_text(encoding="utf-8")
    _, marker, anchor = source.partition("kwargs = dict(cfg.model.dec")
    assert marker
    anchor, end, _ = (marker + anchor).partition("parent.zero_grad(set_to_none=True);")
    assert end
    anchor = anchor.replace("visual_query_init.", "seed_query_skip.")
    core = (ROOT / "scripts/y96_observer_core.py").read_text(encoding="utf-8")
    cpu = prefix + (ROOT / "scripts/probe_y96_cpu_controls.py").read_text(encoding="utf-8")
    cpu += "\n" + anchor + "\n" + (ROOT / "scripts/probe_y96_cpu_tail.py").read_text(encoding="utf-8")
    cpu = cpu.replace("# SHARED_OBSERVER_CORE", core)
    native, marker, _ = (ROOT / "scripts/probe_y83_native_smoke.py").read_text(encoding="utf-8").partition("losses=[];")
    assert marker
    native = native.replace("'AdaRES-Y83'", repr(CID)).replace("164632197", "164695173")
    native += "\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n"
    native += "head=model.dec.indi_embedding;residual=model.dec.seed_query_skip;original_prediction=model.dec.prediction_head\n"
    native += core + "\n" + (ROOT / "scripts/probe_y96_native_tail.py").read_text(encoding="utf-8")
    for value in (cpu, native):
        ast.parse(value, feature_version=(3, 8))
        assert value.count(core) == 1
    return cpu, native


def prepare(token):
    lease(token, 800)
    view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
    assert view["experiment_first"]["enabled"] and view["research"]["creation_budget"] >= 1
    work = tick.load(ROOT / "pipeline/research_work.json")
    assert not any(row["id"] == CID for row in work["candidates"])
    for cid in ("AdaRES-Y94", "AdaRES-Y95"):
        earlier = next(row for row in work["candidates"] if row["id"] == cid)
        assert earlier["stage"] == "blocked" and earlier["blocker_kind"] == "exclusive_gpu_slot"
        assert not earlier.get("native_smoke_result")
    paths = files()
    parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and not any(path.exists() for path in paths)
    source = (parent / "dec.py").read_text(encoding="utf-8")
    for old, new in EDITS:
        assert source.count(old) == 1
        source = source.replace(old, new)
    ast.parse(source, feature_version=(3, 8))
    cpu, native = bodies()
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == "dec.py" else (parent / path.name).read_bytes()
        if path.name == "__init__.py":
            raw = b"# AdaRES-Y96: undetached sampled-seed bypass of the original query generator.\nfrom .ipdn import IPDN\n"
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
    program = ROOT / ("pipeline/checks/y96_cpu_" + token + "_program.py")
    native_path = ROOT / ("pipeline/checks/y96_native_" + token + "_body.py")
    history = ROOT / ("pipeline/checks/y96_targeted_history_" + token + ".json")
    assert not any(path.exists() for path in (program, native_path, history))
    program.write_text("PAYLOAD=" + repr(payload) + "\n" + cpu, encoding="utf-8", newline="\n")
    native_path.write_text(native, encoding="utf-8", newline="\n")
    ids = {"AdaRES-Y25", "AdaRES-Y26", "AdaRES-Y53", "AdaRES-Y55", "AdaRES-Y57",
           "AdaRES-Y73", "AdaRES-Y75", "AdaRES-Y77", "AdaRES-Y84", "AdaRES-Y93", "AdaRES-Y95"}
    tick.save(history, dict(complete=True, scope="Targeted insertion/input comparison, not novelty evidence.",
        matches=[{key: row.get(key) for key in ("id", "stage", "hypothesis", "matched_control")}
                 for row in work["candidates"] if row["id"] in ids], parent_source=ref(parent / "dec.py")))
    support = [ref("scripts/" + name) for name in ("y96_observer_core.py", "probe_y96_cpu_controls.py",
        "probe_y96_cpu_tail.py", "probe_y96_native_tail.py")]
    work["candidates"].append(dict(
        id=CID, profile="experiment_first", stage="implementation", comparison_baseline="AdaRES-W2",
        implementation_package=PACKAGE, config=payload["variant"]["config"],
        hypothesis="The original initial-query generator transforms each sampled post-SCG visual seed through "
            "three linear layers before initial prediction. A parallel learned linear residual from that SAME "
            "seed may ease initial-query fitting while retaining the complete nonlinear generator. The generator "
            "can already learn identity-like mappings; no missing information, defect, novelty or benefit is presumed.",
        matched_control="Independent W2. q_init=query_generator(sampled_seed)+sampled_seed@zeroW256x256, "
            "once after the original generator and before initial prediction; +65536/total164695173/no RNG. "
            "All256 seed channels/all128 selected queries stay undetached. Original sampling indices/post-SCG "
            "features/query positions, complete five-layer generator, static mask features, six refinements/"
            "coarse-fine average/self-language sum/SCG hard neighborhoods/k/gamma/shared confidence/full inputs/"
            "2D/released backbone/targets/loss/selector/seed1999/fixed30/native YAML remain. No scans.",
        historical_scope="Y73 reinjects the already-generated initial query into six later refinements; this "
            "bypasses only the original initial generator using each corresponding original sampled seed once. "
            "Y75 uses pooled language products, Y77 coordinates, Y84 pooled scene visual means, Y95 full-query "
            "means inside recurrent fusion, and Y93 confidence-MLP evidence residuals. None is combined or reopened. "
            "Rejected final-confidence readouts/Y25 gate/Y26 discarded norm/data premises and pausedY57 remain untouched.",
        prepared_native_body=ref(native_path), observer_core=ref("scripts/y96_observer_core.py"),
        next_action="Execute actual CPU original YAML/full model/released backbone/real TRAIN collation, independent "
            "float32/64 scalar/no-RNG/slot-batch-channel permutations/zero/generated-seed gradients, exact seven-stage "
            "W2/input-gradient anchors, initial intervention/exact bypass, original sampled SCG seed identity/all "
            "five generator layers/initial-coarse consumers and two synthetic updates. Sparse GPU native two-step "
            "task remains mandatory; preserve Y92/Y93, earlierY94/Y95, GPU1 andY57 pauses.",
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
