"""Prepare a direct language-conditioned mask-feature path, preserving live jobs."""
import argparse
import ast
import base64
from datetime import datetime, timedelta
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y69'
PACKAGE = 'adares_y69_coarse_geometry_bias'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')

MODULE = 'class BoundedGeometryBias(nn.Module):\n    """A shared head-specific bounded displacement basis for coarse attention."""\n\n    def __init__(self, nhead):\n        super().__init__()\n        self.weight = nn.Parameter(torch.zeros(nhead, 3))\n\n    def forward(self, query_pos, source_pos, batch_mask):\n        clean = source_pos.masked_fill(batch_mask.unsqueeze(-1), 0.)\n        distance = (query_pos.unsqueeze(2) - clean.unsqueeze(1)).abs()\n        basis = (distance / (1. + distance)).square()\n        bias = torch.einsum(\'bqma,ha->bhqm\', basis, self.weight.tanh())\n        bias = bias.masked_fill(batch_mask[:, None, None, :], 0.)\n        return bias.reshape(-1, query_pos.shape[1], source_pos.shape[1])\n\n\nclass GeometryCrossAttentionLayer(CrossAttentionLayer):\n    def forward(self, source, query, batch_mask=None, attn_mask=None, pe=None, att_weights=None, attn_bias=None):\n        if attn_bias is None:\n            return super().forward(source, query, batch_mask, attn_mask, pe, att_weights)\n        batch = query.shape[0]\n        query = self.with_pos_embed(query, pe)\n        combined = attn_bias\n        if attn_mask is not None:\n            assert attn_mask.dtype == torch.bool\n            excluded = attn_mask[:, None].expand(-1, self.nhead, -1, -1).reshape(\n                batch * self.nhead, query.shape[1], source.shape[1])\n            combined = combined.masked_fill(excluded, float(\'-inf\'))\n        output, output_weight, src_weight = self.attn(query, source, source,\n            key_padding_mask=batch_mask, attn_mask=combined, att_weights=att_weights)\n        self.dropout(output)\n        output = output + query\n        self.norm(output)\n        return output, output_weight, src_weight\n\n\n'


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / n for n in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(c['id'] == CID for c in work['candidates'])
    target = files(); assert not any(p.exists() for p in target) and not target[0].parent.exists()
    parent = target[0].parent.parent / PARENT
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    assert source.count('class DEC(nn.Module):') == 1
    source = source.replace('class DEC(nn.Module):', MODULE + 'class DEC(nn.Module):')
    marker = 'self.qia_layers.append(CrossAttentionLayer(d_model, nhead, dropout))'
    assert source.count(marker) == 1
    source = source.replace(marker, 'self.qia_layers.append(GeometryCrossAttentionLayer(d_model, nhead, dropout))')
    marker = '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n        self.coarse_geometry_bias = BoundedGeometryBias(nhead)')
    marker = '        # scg_weights = self.get_scg_weight(query_pos)'
    assert source.count(marker) == 1
    source = source.replace(marker, '        geometry_bias = self.coarse_geometry_bias(query_pos, sp_pos, batch_mask)\n' + marker)
    marker = 'self.qia_layers[i](inst_feats, query, batch_mask, attn_masks)'
    assert source.count(marker) == 1
    source = source.replace(marker, 'self.qia_layers[i](inst_feats, query, batch_mask, attn_masks, attn_bias=geometry_bias)')
    target[0].parent.mkdir()
    for path in target[:-1]:
        raw = source.encode('utf-8') if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y69: bounded head-specific coarse geometry bias.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    target[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    assert len(list((ROOT / 'AdaRES/configs_song').rglob(target[-1].name))) == 1
    def encoded(p): return dict(base64=base64.b64encode(p.read_bytes()).decode(), sha256=ref(p)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE,
        config=target[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={p.name: encoded(p) for p in target[:-1]}, config_data=encoded(target[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    body = ROOT / 'scripts/probe_y69_cpu.py'
    program = ROOT / ('pipeline/checks/y69_cpu_' + token + '_program.py'); assert not program.exists()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='A learned head-specific bounded relative-coordinate bias on the six coarse visual cross-attention logits may ease spatial mask refinement. Native fine attention already uses isotropic SCG distances, and seed/features can encode geometry; this is a direct coarse-attention parameterization comparison, not missing information, novelty or proven efficacy.',
        matched_control='Unchanged W2 plus shared weight[8,3], exact zero/no RNG (+24 parameters). bias[b,h,q,m]=sum_a tanh(weight[h,a])*(abs(qpos[b,q,a]-spos[b,m,a])/(1+abs(delta)))**2, computed once and used by six coarse visual attentions. Original detached Boolean masks remain exact hard exclusions; key padding remains excluded. Fine SCG/k/gamma, shared confidence head/targets/losses/selector, native all-channel input, released SPFormer, Q128/seed1999/fixed30 YAML all unchanged.',
        historical_scope='This changes coarse visual attention logits, not final confidence readbacks (Y28/Y33/Y40), layer score aggregation (Y9/Y30/Y38), higher-order coverage (Y50), the native hard gate (Y25), discarded norm reconnection (Y26), confidence feedback (Y65/Y66), output norms (Y67), or language mask-feature modulation (Y68). Native fine isotropic SCG is explicitly preserved; no rejected premise or ID is reopened.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        next_action='Execute CPU native YAML/full model/released backbone/real TRAIN collation, independent bounded displacement formula, zero output/input-gradient/RNG anchors, padding/translation/permutation/head-isolation/hard-exclusion controls, six-stage bias consumption and finite nonzero new gradients. Actual sparse TRAIN two-step native loss/backward/optimizer on an exclusive authorized GPU remains mandatory before admission.',
        evidence=[ref(p) for p in target] + [ref(program), ref(body), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, program=ref(program), native_executed=False, admitted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True); args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: prepare(args.token)
    finally: lock.close()
