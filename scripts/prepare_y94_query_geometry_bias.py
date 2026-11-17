"""Prepare independent query-self geometry bias and honest pending native checks."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


CID = "AdaRES-Y94"
PACKAGE = "adares_y94_query_geometry_bias"
PARENT = "adares_w2_pointmass_competition"
NAMES = ("__init__.py", "backbone.py", "dec.py", "ipdn.py", "loss.py", "sample_model.py")
MODULE = '''
class BoundedQueryGeometry(nn.Module):
    """A shared bounded axis basis on existing sampled query positions."""
    def __init__(self, nhead):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(nhead, 3))

    def forward(self, query_pos, source_pos, batch_mask):
        clean = source_pos.masked_fill(batch_mask.unsqueeze(-1), 0.)
        distance = (query_pos.unsqueeze(2) - clean.unsqueeze(1)).abs()
        basis = (distance / (1. + distance)).square()
        bias = torch.einsum('bqma,ha->bhqm', basis, self.weight.tanh())
        bias = bias.masked_fill(batch_mask[:, None, None, :], 0.)
        return bias.reshape(-1, query_pos.shape[1], source_pos.shape[1])


class GeometrySelfAttentionLayer(SelfAttentionLayer):
    def forward(self, x, x_mask=None, attn_mask=None, pe=None, norm=True,
                att_weights=None, attn_bias=None):
        if attn_bias is None:
            return super().forward(x, x_mask, attn_mask, pe, norm, att_weights)
        query = key = self.with_pos_embed(x, pe)
        combined = attn_bias
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            excluded = attn_mask[:, None].expand(-1, self.nhead, -1, -1).reshape(
                x.shape[0] * self.nhead, x.shape[1], x.shape[1])
            combined = combined.masked_fill(excluded, float('-inf'))
        output, _, _ = self.attn(query, key, x, key_padding_mask=x_mask,
                                 attn_mask=combined, att_weights=att_weights)
        if self.glu:
            output = self.glu_p(output) * self.glu_g(output).sigmoid()
        output = self.dropout(output) + x
        return self.norm(output) if norm else output


'''
ATTENTION_CONTROL = '''
# Exercise the original custom self attention and retain its actual inner returns.
torch.manual_seed(1999); old_attention = parent_module.SelfAttentionLayer(16, 4)
torch.manual_seed(1999); new_attention = dec_module.GeometrySelfAttentionLayer(16, 4)
assert all(torch.equal(v, new_attention.state_dict()[k]) for k,v in old_attention.state_dict().items())
query = torch.randn(2, 5, 16)
padding = torch.zeros(2, 5, dtype=torch.bool); padding[0, -1:] = True
gate = torch.zeros(2, 5, 5, dtype=torch.bool); gate[:, :, 2] = True
zero = torch.zeros(8, 5, 5)
old_values, new_values = [], []
hooks = [old_attention.attn.register_forward_hook(lambda m,a,o: old_values.append(o)),
         new_attention.attn.register_forward_hook(lambda m,a,o: new_values.append(o))]
try:
    baseline_attention = old_attention(query, padding, gate)
    anchor_attention = new_attention(query, padding, gate, attn_bias=zero)
    assert torch.equal(baseline_attention, anchor_attention)
    assert all(torch.equal(a,b) for a,b in zip(old_values[-1], new_values[-1]))
    bias = torch.randn_like(zero) * .1
    changed_attention = new_attention(query, padding, gate, attn_bias=bias)
    excluded = gate | padding[:, None]
    assert torch.equal(new_values[-1][1][excluded], torch.zeros_like(new_values[-1][1][excluded]))
    expected = old_values[-1][2] + bias.reshape(2,4,5,5).mean(1)
    assert torch.equal(torch.isneginf(expected), torch.isneginf(new_values[-1][2]))
    finite = torch.isfinite(expected)
    torch.testing.assert_allclose(new_values[-1][2][finite], expected[finite], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(changed_attention, baseline_attention)
finally:
    for hook in hooks: hook.remove()

'''
CPU_UPDATES = '''
with torch.no_grad(): decoder.query_geometry_bias.weight.zero_()
decoder.train(True)
synthetic_steps = []
optimizer = torch.optim.SGD(decoder.parameters(), lr=1e-4)
for step_index in range(2):
    optimizer.zero_grad()
    feed = dict(inputs)
    for key in ('x', 'lang_feats', 'feats_2d'):
        feed[key] = inputs[key].clone().requires_grad_()
    value = decoder(**feed)
    labels = torch.arange(256).reshape(2,128).remainder(2).long()
    synthetic_loss = sum(torch.nn.functional.cross_entropy(part['indis'].reshape(-1,2), labels.reshape(-1))
        + part['masks'].square().mean() for part in parts(value))
    synthetic_loss.backward()
    grads = [p.grad for p in decoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(feed[key].grad is not None and torch.isfinite(feed[key].grad).all()
               for key in ('x', 'lang_feats', 'feats_2d'))
    grad = decoder.query_geometry_bias.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    before = decoder.query_geometry_bias.weight.detach().clone()
    optimizer.step()
    update = float((decoder.query_geometry_bias.weight.detach()-before).abs().max())
    assert update>0 and torch.isfinite(decoder.query_geometry_bias.weight).all()
    synthetic_steps.append(dict(step=step_index,loss=float(synthetic_loss.detach()),all_finite=True,
        input_gradients_finite=True,original_gradient_tensors=len(grads),
        new_gradient_max=float(grad.abs().max()),parameter_update_max=update))
'''
NATIVE_GEOMETRY = '''
def observe_geometry(module,args,value):
    query_pos,source_pos,padding=args
    raw=decoder_inputs[-1];offsets=raw[2]
    assert query_pos is source_pos and query_pos.shape[1:]==(128,3)
    assert padding.dtype==torch.bool and not bool(padding.any())
    assert query_pos.shape[0]==len(offsets)-1
    original_positions,_=model.dec.get_batches(raw[5],offsets)
    seeds=original_positions.gather(1,raw[1].long().unsqueeze(-1).expand(-1,-1,3))
    expected_queries=seeds.gather(1,sampled[-1].long().unsqueeze(-1).expand(-1,-1,3))
    assert torch.equal(query_pos,expected_queries)
    assert value.shape==(query_pos.shape[0]*8,128,128) and value.requires_grad and torch.isfinite(value).all()
    value.retain_grad();bias_inputs.append(args);bias_outputs.append(value)
    with torch.no_grad():
        actual=value.detach().reshape(query_pos.shape[0],8,128,128);references=[];observed=[]
        for b in (0,query_pos.shape[0]-1):
            for h in (0,7):
                for q in (0,127):
                    for k in (0,127):
                        delta=(query_pos[b,q].double()-source_pos[b,k].double()).abs()
                        references.append(((delta/(1+delta)).square()*module.weight.detach()[h].double().tanh()).sum())
                        observed.append(actual[b,h,q,k].double())
        expected=torch.stack(references);observed=torch.stack(observed)
        torch.testing.assert_close(observed,expected,rtol=3e-5,atol=2e-6)
        nonzero=bool(module.weight.detach().abs().max()>0)
        if not nonzero:assert torch.equal(value.detach(),torch.zeros_like(value.detach()))
        geometry_paths.append(dict(weight_nonzero=nonzero,bias_max=float(value.detach().abs().max()),
            reference_error_max=float((observed-expected).abs().max()),native_positions_verified=True,
            hard_masks_verified=False,consumed_by_all_query_self_attentions=False,
            expressions=query_pos.shape[0],heads=8,queries=128,
            reference_expression_rows=[0,query_pos.shape[0]-1],reference_head_rows=[0,7],reference_query_rows=[0,127]))
'''
NATIVE_LAYER = '''
    def checked_layer(*args,_original=original,_index=index,**kwargs):
        assert kwargs['attn_bias'] is bias_outputs[-1]
        gate=kwargs['attn_mask']
        _,neighbors=model.dec.get_scg_weight(bias_inputs[-1][0],k=model.dec.num_proposal_query[_index])
        assert torch.equal(gate,~neighbors) and gate.dtype==torch.bool and not gate.requires_grad
        assert args[0].requires_grad;args[0].retain_grad()
        query_self_calls.append(_index);query_inputs.append((args[0],))
        return _original(*args,**kwargs)
'''
NATIVE_ATTENTION = '''
    def checked_attention(*args,_original=original_att,_index=index,**kwargs):
        _,neighbors=model.dec.get_scg_weight(bias_inputs[-1][0],k=model.dec.num_proposal_query[_index])
        excluded=(~neighbors)[:,None].expand(-1,8,-1,-1).reshape_as(bias_outputs[-1])
        expected=bias_outputs[-1].masked_fill(excluded,float('-inf'))
        assert torch.equal(kwargs['attn_mask'],expected) and kwargs['key_padding_mask'] is None
        attention_checks.append(_index)
        return _original(*args,**kwargs)
'''


def files():
    return [ROOT / "AdaRES/ipdn/model_paper5" / PACKAGE / name for name in NAMES] + [
        ROOT / "AdaRES/configs_song/model_paper5" / (PACKAGE + "_res.yaml")]


def replace_once(text, before, after):
    assert text.count(before) == 1, before[:120]
    return text.replace(before, after)


def bodies():
    cpu_path = ROOT / "scripts/probe_y69_cpu_attempt2.py"
    native_path = ROOT / "scripts/probe_y69_native_smoke.py"
    cpu = cpu_path.read_text(encoding="utf-8")
    native = native_path.read_text(encoding="utf-8")
    cpu = cpu.replace("BoundedGeometryBias", "BoundedQueryGeometry")
    cpu = cpu.replace("coarse_geometry_bias", "query_geometry_bias")
    cpu = cpu.replace("decoder.qia_layers", "decoder.qqa_layers")
    start, _, remainder = cpu.partition("# Exercise the actual custom attention")
    assert start and remainder
    _, marker, end = remainder.partition("kwargs = dict(cfg.model.dec")
    assert marker
    cpu = start + ATTENTION_CONTROL + marker + end
    cpu = replace_once(cpu,
        "assert args[3] is None or (args[3].dtype == torch.bool and not args[3].requires_grad)",
        "gate=kwargs['attn_mask'];assert gate.dtype==torch.bool and not gate.requires_grad")
    cpu = replace_once(cpu, "model.eval(); assert all(not m.training for m in model.modules())",
                       CPU_UPDATES + "\nmodel.eval(); assert all(not m.training for m in model.modules())")
    cpu = cpu.replace("decoder_coarse_consumption_calls", "decoder_query_self_consumption_calls")
    cpu = replace_once(cpu, "parameters=parameters, additional_parameters=24, component_controls=controls,",
        "parameters=parameters, additional_parameters=24, component_controls=controls,\n"
        "    synthetic_CPU_decoder_optimizer_steps=2,CPU_decoder_steps=synthetic_steps,")
    native = native.replace("AdaRES-Y69", CID).replace("coarse_geometry_bias", "query_geometry_bias")
    native = native.replace("model.dec.qia_layers", "model.dec.qqa_layers")
    native = native.replace("coarse_calls", "query_self_calls").replace("source_query_inputs", "query_inputs")
    native = native.replace("coarse_stage_calls", "query_self_stage_calls")
    native = native.replace("consumed_by_all_coarse_attentions", "consumed_by_all_query_self_attentions")
    native = native.replace("source_and_query_gradients_finite", "query_gradients_finite")
    start, marker, tail = native.partition("def observe_geometry(module,args,value):")
    assert marker
    _, marker, tail = tail.partition("def observe_indicator(module,args,value):")
    assert marker
    native = start + NATIVE_GEOMETRY + marker + tail
    start, marker, tail = native.partition("    def checked_layer(")
    assert marker
    _, marker, tail = tail.partition("    layer.forward=checked_layer")
    assert marker
    native = start + NATIVE_LAYER + marker + tail
    start, marker, tail = native.partition("    def checked_attention(")
    assert marker
    _, marker, tail = tail.partition("    layer.attn.forward=checked_attention")
    assert marker
    native = start + NATIVE_ATTENTION + marker + tail
    for body in (cpu, native):
        ast.parse(body, feature_version=(3, 8))
    return cpu, native, [ref(cpu_path), ref(native_path)]


def prepare(token):
    lease(token, 800)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        view = tick.load(ROOT / "pipeline/headless/.tick_model_view.json")
        assert view["experiment_first"]["enabled"] and view["research"]["creation_budget"] >= 1
        work = tick.load(ROOT / "pipeline/research_work.json")
        assert not any(row["id"] == CID for row in work["candidates"])
        target = files()
        assert not target[0].parent.exists() and not any(path.exists() for path in target)
        parent = target[0].parent.parent / PARENT
        source = (parent / "dec.py").read_text(encoding="utf-8")
        for before, after in (
            ("class DEC(nn.Module):", MODULE + "class DEC(nn.Module):"),
            ("self.qqa_layers.append(SelfAttentionLayer(d_model, nhead, dropout))",
             "self.qqa_layers.append(GeometrySelfAttentionLayer(d_model, nhead, dropout))"),
            ("        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)",
             "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n"
             "        self.query_geometry_bias = BoundedQueryGeometry(nhead)"),
            ("        # scg_weights = self.get_scg_weight(query_pos)",
             "        query_padding = torch.zeros(query_pos.shape[:2],dtype=torch.bool,device=query_pos.device)\n"
             "        query_geometry = self.query_geometry_bias(query_pos,query_pos,query_padding)\n"
             "        # scg_weights = self.get_scg_weight(query_pos)"),
            ("self.qqa_layers[i](query_qla, attn_mask=~scg_mask)",
             "self.qqa_layers[i](query_qla, attn_mask=~scg_mask,attn_bias=query_geometry)"),
        ):
            source = replace_once(source, before, after)
        ast.parse(source, feature_version=(3, 8))
        cpu, native, templates = bodies()
        target[0].parent.mkdir()
        for path in target[:-1]:
            raw = source.encode() if path.name == "dec.py" else (parent / path.name).read_bytes()
            if path.name == "__init__.py":
                raw = b"# AdaRES-Y94: geometry bias in query self attention only.\nfrom .ipdn import IPDN\n"
            path.write_bytes(raw)
        target[-1].write_bytes((ROOT / "AdaRES/configs_song/model_paper5" / (PARENT + "_res.yaml")).read_bytes())
        assert len(list((ROOT / "AdaRES/configs_song").rglob(target[-1].name))) == 1
        def encoded(path):
            return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)["sha256"])
        payload = dict(
            variant=dict(id=CID,package=PACKAGE,config=target[-1].relative_to(ROOT/"AdaRES").as_posix(),
                python_files={path.name:encoded(path) for path in target[:-1]},config_data=encoded(target[-1])),
            parent_config="configs_song/model_paper5/"+PARENT+"_res.yaml",
            goal_sha256=ref("pipeline/goal.json")["sha256"],
            parent_dec_lf_sha256=hashlib.sha256((parent/"dec.py").read_bytes().replace(b"\r\n",b"\n")).hexdigest(),
            train_source_lf_sha256=hashlib.sha256((ROOT/"AdaRES/tools/train.py").read_bytes().replace(b"\r\n",b"\n")).hexdigest())
        program = ROOT / ("pipeline/checks/y94_cpu_"+token+"_program.py")
        native_program = ROOT / ("pipeline/checks/y94_native_"+token+"_body.py")
        history = ROOT / ("pipeline/checks/y94_targeted_history_"+token+".json")
        assert not any(path.exists() for path in (program,native_program,history))
        program.write_text("PAYLOAD="+repr(payload)+"\n"+cpu,encoding="utf-8",newline="\n")
        native_program.write_text(native,encoding="utf-8",newline="\n")
        history_ids = {"AdaRES-Y25","AdaRES-Y26","AdaRES-Y27","AdaRES-Y54","AdaRES-Y69","AdaRES-Y77",
                       "AdaRES-Y87","AdaRES-Y91","AdaRES-Y92","AdaRES-Y93"}
        tick.save(history,dict(complete=True,scope="Targeted causal location comparison, not novelty proof.",
            matches=[{key:row.get(key) for key in ("id","stage","hypothesis","matched_control")}
                     for row in work["candidates"] if row["id"] in history_ids]))
        work["candidates"].append(dict(
            id=CID,profile="experiment_first",stage="implementation",comparison_baseline="AdaRES-W2",
            implementation_package=PACKAGE,config=payload["variant"]["config"],
            hypothesis="Original query self-attention uses sampled-position SCG hard neighborhoods but no learned "
                "axis-specific position bias inside those same allowed neighborhoods. A shared bounded head-specific "
                "bias from original query-position pairs may ease relational query fitting. Features/SCG already "
                "encode geometry; no missing information, defect, novelty or efficacy is presumed.",
            matched_control="Independent W2; add zero/no-RNG weight[8,3] (+24,total164629661) once, "
                "bias=sum_a tanh(W[h,a])*(abs(qpos_i-qpos_j)/(1+abs(delta)))**2. Apply only to six "
                "original query-self attention logits, retaining exact SCG hard exclusions, neighborhoods/k/gamma, "
                "coarse/fine/language attention, original norms/FFNs/shared confidence/masks/scores/inputs/data/"
                "targets-loss-selector/seed1999/fixed30 YAML. No additional norm, pooling, supervision or scan.",
            historical_scope="Not Y69 coarse query-to-visual-memory attention geometry, Y27 final-confidence "
                "centroid-pair readout, Y77 initial-query position projection, Y87 self/language product or "
                "Y25 reopening of Boolean exclusions. Only original query-to-query self-attention changes; "
                "no combination or reopening of a held/rejected route.",
            prepared_native_body=ref(native_program),
            next_action="Execute actual CPU original YAML/full model/released backbone/real TRAIN collation plus "
                "bounded scalar/permutation/padding/no-RNG/custom-attention/exact seven-stage W2/input-gradient/"
                "causal-bypass and two synthetic decoder updates. Real sparse GPU native two-step task remains "
                "mandatory before admission; preserve current Y91/Y92 and pauses.",
            evidence=[ref(path) for path in target]+[ref(program),ref(native_program),ref(history),ref(__file__)]+templates))
        tick.save(ROOT / "pipeline/research_work.json",work)
        print(json.dumps(dict(candidate=CID,program=ref(program),native_body=ref(native_program),
                              CPU_executed=False,native_executed=False,admitted=False)))
    finally:
        writer.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--token",required=True)
    prepare(parser.parse_args().token)
