"""Prepare one independent query-conditioned coarse/fine fusion comparison."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y72'
PACKAGE = 'adares_y72_visual_mask_refinement'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
MODULE = 'class VisualMaskRefinement(nn.Module):\n    """Residual mask-feature updates from the current native refined visual state."""\n\n    def __init__(self, d_model):\n        super().__init__()\n        self.weight = nn.Parameter(torch.zeros(d_model, d_model))\n\n    def forward(self, mask_features, refined_visual, batch_mask):\n        clean = refined_visual.masked_fill(batch_mask.unsqueeze(-1), 0.)\n        return mask_features + clean @ self.weight\n\n\n'



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
    source = source.replace(old, old + '\n        self.visual_mask_refinement = VisualMaskRefinement(d_model)')
    old = '            pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)'
    assert source.count(old) == 1
    source = source.replace(old, '            refined_mask_feats = self.visual_mask_refinement(mask_feats, inst_feats, batch_mask)\n            pred_scores, pred_masks, attn_masks = self.prediction_head(query, refined_mask_feats, batch_mask)')
    targets[0].parent.mkdir()
    for p in targets[:-1]:
        raw = source.encode('utf-8') if p.name == 'dec.py' else (parent / p.name).read_bytes()
        if p.name == '__init__.py':
            raw = b'# AdaRES-Y72: mask-feature refinement from the current native visual state.\nfrom .ipdn import IPDN\n'
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
    body = ROOT / 'scripts/probe_y72_cpu.py'
    program = ROOT / ('pipeline/checks/y72_cpu_' + token + '_program.py')
    assert not program.exists()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    relevant = []
    for c in work['candidates']:
        text = ' '.join(str(c.get(k, '')) for k in ('hypothesis', 'matched_control', 'historical_scope')).lower()
        if any(phrase in text for phrase in ('dynamic mask', 'refined visual', 'mask refinement features', 'visual-to-mask', 'stage mask')):
            relevant.append(dict(id=c['id'], stage=c['stage'], hypothesis=c.get('hypothesis', '')[:800]))
    history = ROOT / ('pipeline/checks/y72_targeted_history_' + token + '.json')
    assert not history.exists()
    tick.save(history, dict(complete=True, scope='Candidate hypothesis/matched_control/historical_scope text only; no full-history or literature novelty claim.',
        phrases=['dynamic mask', 'refined visual', 'mask refinement features', 'visual-to-mask', 'stage mask'], matches=relevant,
        parent_equal_fusion_source=ref(parent / 'dec.py'), parent_fusion='loop prediction_head(query, mask_feats, batch_mask) uses the unchanged shared mask feature tensor'))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='W2 refines visual instance states with language at every decoder stage but predicts all masks against the same static x_mask features. A shared zero-initialized residual from the current refined visual state into each later mask feature tensor may ease conditional mask fitting. Queries already carry contextual information; missing information, novelty and efficacy are not presumed.',
        matched_control='Unchanged W2 plus mask_features_i=mask_features+clean(refined_visual_i)@W before the six loop prediction heads; W[256,256] exact zero/no RNG (+65536 parameters), padded rows zeroed, stage0 unchanged. Each call consumes current original ila_ffn output without accumulating previous residuals. Original shared confidence heads/targets/losses, mask-to-attention hard-gate algorithm, original SCG/k/gamma, query sampling, equal coarse/fine fusion, static x_mask, full inputs/existing2D data, released SPFormer, Q128/seed1999/fixed30/native YAML remain unchanged.',
        historical_scope='This updates per-stage mask features from current refined visual states; not Y68 one static language modulation, Y71 early3D/2D fusion, Y70 coarse/fine query mixing, Y65/Y66 confidence/peer query feedback, Y69 geometry logits or Y67 readout norms. Not Y28/Y33/Y40 final-confidence visual readbacks and not Y25 continuous mask attention gate: original hard-gate rule remains. No rejected ID is reopened and targeted history text is not an exhaustive novelty claim.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        next_action='Execute CPU native YAML/full model/released backbone/actual TRAIN collation, independent residual formula/zero output-input-gradient-RNG anchors/padding and row-channel permutation/wrong-stage correspondence controls; all seven decoder anchors, six current ila_ffn-to-mask-feature paths and six actual prediction consumers, finite nonzero new gradients and stage0/bypass controls. Preserve existing Y71 exact tested HEAD; native sparse TRAIN two-step loss/backward/optimizer remains mandatory before admission.',
        evidence=[ref(p) for p in targets] + [ref(program), ref(body), ref(history), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, program=ref(program), native_executed=False, admitted=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--token', required=True); a = p.parse_args()
    lease(a.token, 180)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: prepare(a.token)
    finally: lock.close()
