"""Prepare a language-conditioned initial-query interaction, preserving runs."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y77'
PACKAGE='adares_y77_geometry_query_init'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE='''class GeometryQueryInitialization(nn.Module):
    """One direct scene-centered sampled-position projection before initial prediction."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, d_model))

    def forward(self, query, query_pos, scene_pos, padding, point_mass):
        weights = point_mass.masked_fill(padding, 0.0)
        clean = scene_pos.masked_fill(padding.unsqueeze(-1), 0.0)
        totals = weights.sum(1, keepdim=True)
        center = (clean * weights.unsqueeze(-1)).sum(1) / totals.clamp_min(1.0)
        relative = query_pos - center.unsqueeze(1)
        relative = relative.masked_fill((totals <= 0).unsqueeze(-1), 0.0)
        return query + relative @ self.weight


'''


def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source=(parent/'dec.py').read_text(encoding='utf-8')
    edits=[('class DEC(nn.Module):',MODULE+'class DEC(nn.Module):'),
        ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.geometry_query_init = GeometryQueryInitialization(d_model)'),
        ('\n        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)',
         '\n        query = self.geometry_query_init(query, query_pos, sp_pos, batch_mask, point_mass)\n        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y77: original sampled-position offset from the point-mass scene centroid.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y77_cpu.py';program=ROOT/('pipeline/checks/y77_cpu_'+token+'_program.py');history=program.with_name('y77_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    phrases=['query position','sampled position','seed coordinate','geometry initialization','scene centroid']
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis','')[:800]) for c in work['candidates'] if c['id'] in ('AdaRES-Y27','AdaRES-Y31','AdaRES-Y53','AdaRES-Y55','AdaRES-Y69','AdaRES-Y73','AdaRES-Y75')]
    tick.save(history,dict(complete=True,scope='Targeted candidate-text comparison only; not exhaustive novelty or literature evidence.',phrases=phrases,matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='Original sampled seed coordinates determine native SCG neighborhoods, while the initial query content is generated from existing visual states. A direct learned projection of each original query position relative to the all-point-mass scene centroid may ease fitting spatially grounded initial masks/confidence. Visual features and SCG already encode geometry; missing information, novelty and trained benefit are not presumed.',
        matched_control='Unchanged W2 plus q_init=q_generated+(original_query_pos-point_mass_weighted_mean(all_original_valid_scene_sp_pos))@W once immediately before original initial prediction. W[3,256] exactly zero/no RNG (+768). All three existing coordinates and all valid original scene point masses contribute; zero-mass empty-scene branch is a no-op. Preserve original sampling indices/query positions, static mask features, all six original refinements/hard gates/SCG/k/gamma/coarse-fine fusion, seven shared confidence calls/targets/losses/selector, original full input/existing2D/released backbone/Q128/seed1999/fixed30/native YAML. This is an initial representation connection, with no frequency/threshold/loss-weight sweep or validation selection.',
        historical_scope='Unlike Y69 learned geometry on six coarse attention logits, Y27 mask-derived peer centroid confidence, Y31 mask-shape second moments, Y73 repeated initial-state anchors or Y75 pooled-language/query product. Does not reopen their frozen experiments or claim native coordinate/point-mass/padding bugs (Y53/Y55); no changed data/augmentation or geometry labels are introduced.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Execute actual CPU native YAML/full model/released pretrain/TRAIN collation and independent weighted-centroid/scalar projection, zero/RNG/input gradients, translation/permutation/padding/point-mass intervention, full seven-stage W2 anchors and exact original generated-query/scene-position/mass/SCG-query-position identities plus initial prediction consumer. Preserve active HEAD9bf and all live Y72/Y74/Y75. Retain tested source and prepare native two-step checks; no CPU-based GPU PASS or efficacy.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
