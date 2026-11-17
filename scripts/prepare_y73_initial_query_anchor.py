"""Prepare an independent initial-query residual comparison without moving HEAD."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y73'
PACKAGE = 'adares_y73_initial_query_anchor'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
MODULE = '''class QueryAnchorResidual(nn.Module):
    """Compare a direct residual from the original sampled query state."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, query, anchor_query):
        return query + (anchor_query - query) @ self.weight


'''


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / n for n in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['experiment_first']['enabled'] and view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(c['id'] == CID for c in work['candidates'])
    paths = files(); parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    replacements = [
        ('class DEC(nn.Module):', MODULE + 'class DEC(nn.Module):'),
        ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.query_anchor_residual = QueryAnchorResidual(d_model)'),
        ('\n        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)',
         '\n        anchor_query = query\n        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)'),
        ('            query = self.qqa_ffn_layers[i](query)',
         '            query = self.qqa_ffn_layers[i](query)\n            query = self.query_anchor_residual(query, anchor_query)')]
    for old, new in replacements:
        assert source.count(old) == 1
        source = source.replace(old, new)
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        data = source.encode('utf-8') if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            data = b'# AdaRES-Y73: direct residual from the original sampled query state.\nfrom .ipdn import IPDN\n'
        path.write_bytes(data)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    paths[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    assert list((ROOT / 'AdaRES/configs_song').rglob(paths[-1].name)) == [paths[-1]]
    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE,
        config=paths[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={p.name: encoded(p) for p in paths[:-1]}, config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    body = ROOT / 'scripts/probe_y73_cpu.py'
    program = ROOT / ('pipeline/checks/y73_cpu_' + token + '_program.py')
    history = ROOT / ('pipeline/checks/y73_targeted_history_' + token + '.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    phrases = ['initial query', 'initial-query', 'query anchor', 'query anchoring', 'anchor query',
               'query identity', 'query trajectory', 'query skip', 'stage-zero', 'stage0 query']
    matches = []
    for c in work['candidates']:
        text = ' '.join(str(c.get(k, '')) for k in ('hypothesis', 'matched_control', 'historical_scope')).lower()
        if any(p in text for p in phrases):
            matches.append(dict(id=c['id'], stage=c['stage'], hypothesis=c.get('hypothesis', '')[:800]))
    tick.save(history, dict(complete=True, scope='Candidate hypothesis/control/history text only; not exhaustive novelty or literature evidence.',
        phrases=phrases, matches=matches, parent_source=ref(parent / 'dec.py')))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='The original sampled query state is successively transformed by six contextual refinement stages. A shared zero-initialized residual from that same query slot original state may ease fitting a balance between initial anchor identity and later contextual updates. Existing residual paths already retain information; information loss, novelty and efficacy are not presumed.',
        matched_control='Unchanged W2 plus q_i=q_after_original_qqa_ffn_i+(q_initial-q_after_original_qqa_ffn_i)@W before each of six loop predictions. W[256,256] is exactly zero without RNG (+65536 parameters); original stage0 query tensor is the same undetached anchor for all calls. Original static mask features, query positions/sampling, all input channels/existing2D, coarse/fine equal averaging, Boolean hard gate/SCG/k/gamma, shared confidence heads/targets/losses/selector, released SPFormer, Q128/seed1999/fixed30/native YAML stay unchanged.',
        historical_scope='This adds initial sampled-query state reuse after each complete refinement; not Y65/Y66 confidence/peer feedback, Y70 within-stage coarse/fine query fusion, Y71 early visual fusion, Y72 current-visual mask features, Y68 static language modulation, Y69 geometry bias or Y67 readout norms. It does not change confidence evidence or reopen a rejected/banned ID. Targeted candidate-text search is not an exhaustive novelty claim.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        next_action='Run actual CPU native-YAML/full model/released backbone/TRAIN collation plus independent anchor-residual formula, zero output/input-gradient/RNG anchors, joint permutations, same-anchor/wrong-correspondence controls, full seven-stage decoder anchors and six original qqa_ffn/current initial-anchor identities and actual prediction consumers, finite new gradient and unchanged-stage0/bypass controls. Preserve current Y71 HEAD87019/Y72 retained branch and original runs. Actual native sparse TRAIN two finite loss/backward/optimizer steps remain mandatory before admission.',
        evidence=[ref(p) for p in paths] + [ref(program), ref(body), ref(history), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(candidate=CID, program=ref(program), native_executed=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True); args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try:
        prepare(args.token)
    finally:
        lock.close()
