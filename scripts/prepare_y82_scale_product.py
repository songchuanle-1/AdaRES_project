"""Prepare independent cross-scale query product fusion; preserve runs."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y82'
PACKAGE='adares_y82_scale_product'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE='class ScaleProductFusion(nn.Module):\n    """Cross-channel product residual on original coarse/fine query updates."""\n\n    def __init__(self, d_model):\n        super().__init__()\n        self.weight = nn.Parameter(torch.zeros(d_model, d_model))\n\n    def forward(self, coarse, fine):\n        return (coarse + fine) / 2 + (coarse * fine) @ self.weight\n\n\n'


def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source=(parent/'dec.py').read_text(encoding='utf-8')
    edits=[('class DEC(nn.Module):',MODULE+'class DEC(nn.Module):'),
        ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.scale_product = ScaleProductFusion(d_model)'),
        ('            query_qia = (coarse_query + fine_query)/2',
         '            query_qia = self.scale_product(coarse_query, fine_query)')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y82: coarse/fine query product residual in six original fusion calls.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y82_cpu.py';program=ROOT/('pipeline/checks/y82_cpu_'+token+'_program.py');history=program.with_name('y82_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis',''),matched_control=c.get('matched_control','')) for c in work['candidates'] if c['id'] in ('AdaRES-Y25','AdaRES-Y26','AdaRES-Y28','AdaRES-Y30','AdaRES-Y38','AdaRES-Y65','AdaRES-Y66','AdaRES-Y69','AdaRES-Y70','AdaRES-Y71','AdaRES-Y73','AdaRES-Y76','AdaRES-Y79','AdaRES-Y80','AdaRES-Y81')]
    tick.save(history,dict(complete=True,scope='Targeted architecture/source comparison only, not exhaustive novelty.',matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='Native W2 averages coarse and fine query updates. A zero-initialized learned projection of their elementwise product may ease cross-scale interaction fitting. Downstream networks can already represent interactions; novelty, missing information and trained efficacy are not presumed.',
        matched_control='Independent unchanged W2 plus fused=(coarse+fine)/2+(coarse*fine)@W in the six original fusion locations. One shared zero W256x256/no RNG/+65536; both original branch tensors and all128 queries/all channels remain undetached. Initial prediction, original static masks/Boolean gates/sampling/SCG/geometry/k/gamma, all original inputs/existing2D/backbone and seven shared confidence heads/targets/losses/selector, seed1999/fixed30/native YAML remain. No threshold/temperature/loss/feature/epoch or validation search.',
        historical_scope='Y70 uses query-conditioned bounded convex channel mixing, whereas Y82 uses the two original branch tensors directly in a symmetric product projection that need not stay between branches. Y71 multiplies original sparse3D and existing2D features before sum_norm, not coarse/fine query updates. Y82 is a separate W2 comparison, not combined with Y70/Y71, and does not reopen rejected final-confidence/history/ranking/gate/normalization routes. No exhaustive novelty claim.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Run CPU native YAML/full model/released backbone/original TRAIN collation, independent scalar product projection and zero/RNG/gradients/permutations/branch-swap/same-mean different-product controls, seven-stage W2 and input-gradient anchors, six original coarse/fine tensor identities and actual qla consumers, shared confidence/query_norm, nonzero/bypass and two synthetic CPU decoder optimizer steps. Retain exact tested files and native two-step plan; wait exclusive authorized GPU. Preserve Y78/Y79/Y80 and Y57 pause. No CPU-based native GPU PASS/admission/efficacy.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
