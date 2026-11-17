"""Prepare a language-conditioned initial-query interaction, preserving runs."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y84'
PACKAGE='adares_y84_visual_query_init'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE='class VisualQueryInitialization(nn.Module):\n    """Add all-valid point-mass-weighted scene visual context once to initial queries."""\n\n    def __init__(self, d_model):\n        super().__init__()\n        self.weight = nn.Parameter(torch.zeros(d_model, d_model))\n\n    def forward(self, query, scene_features, padding, point_mass):\n        weights = point_mass.masked_fill(padding, 0.0)\n        clean = scene_features.masked_fill(padding.unsqueeze(-1), 0.0)\n        totals = weights.sum(1, keepdim=True)\n        context = (clean * weights.unsqueeze(-1)).sum(1) / totals.clamp_min(1.0)\n        context = context.masked_fill(totals <= 0, 0.0)\n        return query + (context @ self.weight).unsqueeze(1)\n\n\n'


def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source=(parent/'dec.py').read_text(encoding='utf-8')
    edits=[('class DEC(nn.Module):',MODULE+'class DEC(nn.Module):'),
        ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.visual_query_init = VisualQueryInitialization(d_model)'),
        ('\n        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)',
         '\n        query = self.visual_query_init(query, inst_feats, batch_mask, point_mass)\n        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y84: all-valid scene visual context before the original initial prediction.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y84_cpu.py';program=ROOT/('pipeline/checks/y84_cpu_'+token+'_program.py');history=program.with_name('y84_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis',''),matched_control=c.get('matched_control','')) for c in work['candidates'] if c['id'] in ('AdaRES-Y55','AdaRES-Y68','AdaRES-Y73','AdaRES-Y75','AdaRES-Y77','AdaRES-Y80','AdaRES-Y81','AdaRES-Y82','AdaRES-Y83')]
    tick.save(history,dict(complete=True,scope='Targeted input/location and formula comparison only; not exhaustive novelty.',matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='Native initial queries use sampled local visual states before six refinements. A direct learned projection of point-mass-weighted mean scene visual features into initial queries may ease global-context fitting. Later cross-attention already provides global information; no missing information, novelty or efficacy is presumed.',
        matched_control='Independent unchanged W2 plus q_init=q_generated+weighted_mean(all valid original inst_feats, point_mass)@W once before initial prediction. Shared zero W256x256/no RNG/+65536. Use all256 channels and all valid original superpoints/masses; masked or zero-total support contributes zero. Preserve all original coordinates/sampling/SCG/k/gamma/hard gates/static masks/full input/existing2D/released backbone/shared confidence/targets/loss/selector/seed1999/fixed30/native YAML. No validation scans.',
        historical_scope='Y77 projects sampled relative coordinates rather than visual scene context; Y75 multiplies the initial query by pooled language; Y68 modulates mask features with language; Y80 feeds current query attention into recurrent visual states; Y81 feeds query attention into mask features. This candidate uses original visual mean once at initial query construction, without combining those routes. No reopening of rejected point-mass-ordering Y55 or any frozen rejected route; no defect or novelty claim.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Execute CPU original native YAML/full model/released backbone/TRAIN collation, independent weighted visual mean and projection controls, zero/RNG/empty/padding/permutation/mass-response/input-gradient checks, full seven-stage W2/output/input-gradient anchors and exact original generated-query/scene-feature/mass consumers plus two synthetic CPU decoder optimizer steps. Preserve active Y80/Y81/Y82 and Y57 pause. Only actual native GPU TRAIN two-step PASS permits admission; CPU results do not establish task efficacy.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
