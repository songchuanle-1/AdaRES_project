"""Prepare an independent learned previous-mask logit residual; preserve runs."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y78'
PACKAGE='adares_y78_mask_logit_residual'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE='class MaskLogitResidual(nn.Module):\n    """Learn a shared query-conditioned path from the immediately prior mask."""\n\n    def __init__(self, d_model):\n        super().__init__()\n        self.weight = nn.Parameter(torch.zeros(d_model, 1))\n\n    def forward(self, query, current_masks, previous_masks, padding):\n        previous = previous_masks.masked_fill(padding.unsqueeze(1), 0.0)\n        return current_masks + torch.tanh(query @ self.weight) * previous\n\n\n'
METHOD='    def refined_prediction_head(self, query, mask_feats, batch_mask, previous_masks):\n        pred_scores, raw_masks, _ = self.prediction_head(query, mask_feats, batch_mask)\n        pred_masks = self.mask_logit_residual(query, raw_masks, previous_masks, batch_mask)\n        if self.attn_mask:\n            attn_masks = (pred_masks.sigmoid() < 0.5).bool()\n            attn_masks = attn_masks | batch_mask.unsqueeze(1)\n            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False\n            attn_masks = attn_masks | batch_mask.unsqueeze(1)\n            attn_masks = attn_masks.detach()\n        else:\n            attn_masks = None\n        return pred_scores, pred_masks, attn_masks\n\n'


def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source=(parent/'dec.py').read_text(encoding='utf-8')
    edits=[('class DEC(nn.Module):',MODULE+'class DEC(nn.Module):'),
        ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.mask_logit_residual = MaskLogitResidual(d_model)'),
        ('            pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)',
         '            pred_scores, pred_masks, attn_masks = self.refined_prediction_head(query, mask_feats, batch_mask, pred_masks)'),
        ('    def prediction_head(self, query, mask_feats, batch_mask):',METHOD+'    def prediction_head(self, query, mask_feats, batch_mask):')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y78: learned immediately preceding mask-logit residual into six refined masks.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y78_cpu.py';program=ROOT/('pipeline/checks/y78_cpu_'+token+'_program.py');history=program.with_name('y78_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    phrases=['previous mask','mask logit residual','mask accumulator','ordered history','mask attention prior']
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis','')[:1000],matched_control=c.get('matched_control','')[:1200]) for c in work['candidates'] if c['id'] in ('AdaRES-Y25','AdaRES-Y30','AdaRES-Y38','AdaRES-Y65','AdaRES-Y72','AdaRES-Y73','AdaRES-Y74','AdaRES-Y76','AdaRES-Y77')]
    tick.save(history,dict(complete=True,scope='Targeted candidate-text comparison only; not exhaustive novelty or literature evidence.',phrases=phrases,matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='Each original refinement predicts fresh masks from its query and static mask features; the preceding mask enters only the detached Boolean attention gate. A learned direct residual from the immediately preceding mask logits may ease iterative support refinement. The query already carries history; missing information, novelty and trained efficacy are not presumed.',
        matched_control='Independent unchanged W2 plus adjusted_masks=raw_current_masks+tanh(original_current_query@w)*clean(previous_adjusted_masks), shared w[256,1] exactly zero/no RNG (+256). Six calls only, initial stage unchanged; padding positions of previous masks zeroed. Previous mask remains differentiable. Original score query/readout, static mask features, full input/existing2D/released backbone, all six refinements/SCG/k/gamma, seven shared confidence calls/targets/losses/argmax, Q128/seed1999/fixed30/native YAML and the exact native Boolean attention rule remain. No scalar sweep, threshold, loss/feature/epoch or validation selection.',
        historical_scope='Y30/Y38 rejected final confidence history readouts; this changes six current mask logits and subsequent refinement, not a history-readout feature or claim of temporal evidence necessity. Y25 soft training-only attention prior is absent: the existing detached Boolean gate rule is applied unchanged to the adjusted mask. Y65 confidence-to-query feedback, Y72 current-visual mask features and Y73 initial-query anchor remain distinct frozen experiments; none is reopened or combined. Targeted source/history comparison is not exhaustive novelty evidence.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Execute CPU native YAML/full model/released backbone/actual original TRAIN collation, independent scalar gate-residual formula, query/point/batch permutations and padding/mispaired-previous-mask controls, zero/RNG/parent weights/all-seven-stage output and input-gradient anchors, exact six original raw/current/previous/query/static-mask identities, unchanged Boolean-gate consumer, all seven shared confidence consumers, finite new gradients and bypass. Retain tested source and prepare actual native two-step checks; all three existing runs continue. No CPU-based GPU PASS or efficacy.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
