"""Prepare one independent query-conditioned coarse/fine fusion comparison."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y70'
PACKAGE = 'adares_y70_query_scale_fusion'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
MODULE = '''class QueryScaleFusion(nn.Module):
    """Bounded per-query channel mixing anchored to the original equal average."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, query, coarse, fine):
        modulation = torch.tanh(query @ self.weight)
        return (coarse + fine) / 2 + .5 * modulation * (fine - coarse)


'''


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / n for n in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['experiment_first']['enabled'] and view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(c['id'] == CID for c in work['candidates'])
    targets = files()
    assert not targets[0].parent.exists() and not any(p.exists() for p in targets)
    parent = targets[0].parent.parent / PARENT
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    old = 'class DEC(nn.Module):'
    assert source.count(old) == 1
    source = source.replace(old, MODULE + old)
    old = '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)'
    assert source.count(old) == 1
    source = source.replace(old, old + '\n        self.query_scale_fusion = QueryScaleFusion(d_model)')
    old = '            query_qia = (coarse_query + fine_query)/2'
    assert source.count(old) == 1
    source = source.replace(old, '            query_qia = self.query_scale_fusion(query, coarse_query, fine_query)')
    targets[0].parent.mkdir()
    for p in targets[:-1]:
        raw = source.encode('utf-8') if p.name == 'dec.py' else (parent / p.name).read_bytes()
        if p.name == '__init__.py':
            raw = b'# AdaRES-Y70: query-conditioned coarse/fine fusion.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw)
        ast.parse(p.read_text(encoding='utf-8'), feature_version=(3, 8))
    targets[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    def encoded(p):
        return dict(base64=base64.b64encode(p.read_bytes()).decode(), sha256=ref(p)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE,
        config=targets[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={p.name: encoded(p) for p in targets[:-1]}, config_data=encoded(targets[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    body = ROOT / 'scripts/probe_y70_cpu.py'
    program = ROOT / ('pipeline/checks/y70_cpu_' + token + '_program.py')
    assert not program.exists()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    relevant = []
    for c in work['candidates']:
        text = ' '.join(str(c.get(k, '')) for k in ('hypothesis', 'matched_control', 'historical_scope')).lower()
        if any(phrase in text for phrase in ('coarse/fine fusion', 'coarse-fine fusion', 'coarse and fine', 'scale fusion')):
            relevant.append(dict(id=c['id'], stage=c['stage'], hypothesis=c.get('hypothesis', '')[:800]))
    history = ROOT / ('pipeline/checks/y70_targeted_history_' + token + '.json')
    assert not history.exists()
    tick.save(history, dict(complete=True, scope='Candidate hypothesis/matched_control/historical_scope text only; no full-history or literature novelty claim.',
        phrases=['coarse/fine fusion', 'coarse-fine fusion', 'coarse and fine', 'scale fusion'], matches=relevant,
        parent_equal_fusion_source=ref(parent / 'dec.py'), parent_fusion='query_qia = (coarse_query + fine_query)/2'))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='Native W2 averages coarse and fine visual query updates equally in all six refinement stages. A shared query-conditioned channel-wise bounded mixture may make the two spatial scales easier to combine. This changes representation fusion, not confidence targets or inference selection; neither missing information, novelty nor efficacy is presumed.',
        matched_control='Unchanged W2 plus shared W[256,256] exact zero/no RNG (+65536 parameters): fused=(coarse+fine)/2 + .5*tanh(query@W)*(fine-coarse). At W=0 all original outputs/input gradients and parent RNG/weights anchor to W2. The original preceding-stage query conditions six fusion calls; stage0 is unchanged. Original coarse/fine attention outputs, Boolean hard gate, fine SCG/k/gamma, seven shared confidence calls/targets/losses, selector, all channels, released SPFormer, Q128/seed1999/fixed30 native YAML remain unchanged.',
        historical_scope='Not Y25 continuous attention-prior gate or Y26 discarded LayerNorm reconnection. This consumes already-computed coarse/fine outputs immediately before original query-language attention. Not Y65/Y66 confidence/peer feedback into refinement, Y67 stage readout norms, Y68 language mask-feature modulation or Y69 coarse geometry logits. No rejected route is reopened; targeted history text is recorded without claiming exhaustive novelty.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        next_action='Execute CPU native YAML/full model/released backbone/actual TRAIN collation, independent fusion formula/convex bounds/query and channel permutation/equal-branch/zero-RNG and gradient controls; all seven decoder anchors and six ordered actual downstream fusion consumers. A real exclusive-GPU native TRAIN loss/backward/optimizer two-step check remains mandatory before admission.',
        evidence=[ref(p) for p in targets] + [ref(program), ref(body), ref(history), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, program=ref(program), native_executed=False, admitted=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--token', required=True); a = p.parse_args()
    lease(a.token, 180)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: prepare(a.token)
    finally: lock.close()
