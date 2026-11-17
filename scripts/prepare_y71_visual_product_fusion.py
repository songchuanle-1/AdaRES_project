"""Prepare one independent query-conditioned coarse/fine fusion comparison."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y71'
PACKAGE = 'adares_y71_visual_product_fusion'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
MODULE = '''class VisualProductFusion(nn.Module):
    """Zero-initialized explicit interaction between native projected 3D/2D features."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, spatial, appearance):
        return spatial + appearance + (spatial * appearance) @ self.weight


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
    source = source.replace(old, old + '\n        self.visual_product_fusion = VisualProductFusion(d_model)')
    old = '        inst_feats = self.sum_norm(inst_feats+inst_feats_2d)'
    assert source.count(old) == 1
    source = source.replace(old, '        inst_feats = self.sum_norm(self.visual_product_fusion(inst_feats, inst_feats_2d))')
    targets[0].parent.mkdir()
    for p in targets[:-1]:
        raw = source.encode('utf-8') if p.name == 'dec.py' else (parent / p.name).read_bytes()
        if p.name == '__init__.py':
            raw = b'# AdaRES-Y71: multiplicative interaction of native 3D/2D visual features.\nfrom .ipdn import IPDN\n'
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
    body = ROOT / 'scripts/probe_y71_cpu.py'
    program = ROOT / ('pipeline/checks/y71_cpu_' + token + '_program.py')
    assert not program.exists()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    relevant = []
    for c in work['candidates']:
        text = ' '.join(str(c.get(k, '')) for k in ('hypothesis', 'matched_control', 'historical_scope')).lower()
        if any(phrase in text for phrase in ('visual product fusion', '2d', '2-d', 'visual fusion', 'cross-modal', 'multiplicative fusion')):
            relevant.append(dict(id=c['id'], stage=c['stage'], hypothesis=c.get('hypothesis', '')[:800]))
    history = ROOT / ('pipeline/checks/y71_targeted_history_' + token + '.json')
    assert not history.exists()
    tick.save(history, dict(complete=True, scope='Candidate hypothesis/matched_control/historical_scope text only; no full-history or literature novelty claim.',
        phrases=['visual product fusion', '2d', '2-d', 'visual fusion', 'cross-modal', 'multiplicative fusion'], matches=relevant,
        parent_equal_fusion_source=ref(parent / 'dec.py'), parent_fusion='inst_feats = self.sum_norm(inst_feats+inst_feats_2d)'))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='Native W2 adds projected sparse-3D and existing 2D appearance features before sum_norm. An explicit learned channel-product residual before that same normalization may ease visual interaction fitting. Existing downstream attention/MLPs may represent such interactions; neither missing information, novelty nor efficacy is presumed.',
        matched_control='Unchanged W2 plus fused=s+a+(s*a)@W for native projected spatial/appearance features; W[256,256] exact zero/no RNG (+65536 parameters), one call before original sum_norm. Original all-channel inputs and existing 2D features, mask features, query sampling, coarse/fine equal fusion, hard masks/SCG/k/gamma, seven shared confidence calls/targets/losses/selector, released SPFormer, Q128/seed1999/fixed30 and native YAML are preserved. At zero, all parent weights/RNG/decoder outputs/input gradients anchor to W2.',
        historical_scope='This changes early native 3D/2D visual fusion, not Y70 query-conditioned coarse/fine updates, Y68 language mask-feature modulation, Y69 geometry bias, Y65/Y66 feedback, Y67 readout norms, or Y63 confidence bilinear readout. Original feature sets and hard attention gates remain unchanged; no historical rejected ID or missing-information premise is reopened. Bounded candidate-text history search is recorded without an exhaustive novelty claim.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        next_action='Run CPU native YAML/full model/released backbone/actual TRAIN collation, independent product formula and same-sum/different-factorization control, zero output/input-gradient/RNG anchors, row/channel permutations, zero factors and wrong correspondence, full seven-stage decoder anchors and actual fusion-to-sum_norm consumption/finite nonzero new gradients. A real exclusive-GPU native sparse TRAIN two-step loss/backward/optimizer remains mandatory before admission.',
        evidence=[ref(p) for p in targets] + [ref(program), ref(body), ref(history), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, program=ref(program), native_executed=False, admitted=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--token', required=True); a = p.parse_args()
    lease(a.token, 180)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: prepare(a.token)
    finally: lock.close()
