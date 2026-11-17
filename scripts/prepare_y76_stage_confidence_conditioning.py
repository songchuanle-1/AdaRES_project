"""Prepare a confidence-only stage input; preserve every existing experiment."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y76'
PACKAGE='adares_y76_stage_confidence_conditioning'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE='''class StageConfidenceConditioning(nn.Module):
    """Learn stage identity at the shared confidence query input only."""

    def __init__(self, d_model):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(7, d_model))

    def forward(self, query, stage):
        if type(stage) is not int or not 0 <= stage < 7:
            raise ValueError("Expected original decoder stage 0..6")
        return query + self.offset[stage]


'''

def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
    assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT
    assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source=(parent/'dec.py').read_text(encoding='utf-8')
    edits=[('class QueryCompetitionIndicator(nn.Module):',MODULE+'class QueryCompetitionIndicator(nn.Module):'),
        ('    def forward(self, query, pred_masks, batch_mask, point_mass):','    def forward(self, query, pred_masks, batch_mask, point_mass, stage):'),
        ('        normalized_query = self.query_norm(query)','        normalized_query = self.query_norm(self.stage_conditioning(query, stage))'),
        ('        self.head = nn.Sequential(','        self.stage_conditioning = StageConfidenceConditioning(d_model)\n        self.head = nn.Sequential('),
        ('\n        pred_indis = self.indi_embedding(query, pred_masks, batch_mask, point_mass)\n','\n        pred_indis = self.indi_embedding(query, pred_masks, batch_mask, point_mass, 0)\n'),
        ('\n            pred_indis = self.indi_embedding(query, pred_masks, batch_mask, point_mass)\n','\n            pred_indis = self.indi_embedding(query, pred_masks, batch_mask, point_mass, i + 1)\n')]
    for old,new in edits:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y76: stage identity at the shared confidence input only.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y76_cpu.py';program=ROOT/('pipeline/checks/y76_cpu_'+token+'_program.py');history=program.with_name('y76_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis','')[:800],matched_control=c.get('matched_control','')[:1500]) for c in work['candidates'] if c['id'] in ('AdaRES-Y60','AdaRES-Y67')]
    tick.save(history,dict(complete=True,scope='Targeted comparison with completed Y60 and Y67 only; no exhaustive novelty claim.',matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='W2 applies one shared confidence head to initial and six refined queries without an explicit stage input. A learned stage-specific additive query offset may condition that same function on refinement stage while retaining head sharing. Queries may already encode stage; no missing information, novelty or efficacy is presumed.',
        matched_control='Unchanged W2 plus query_for_confidence=query+offset[stage], stage0..6, offset[7,256] exactly zero/no RNG (+1792). Only the shared QueryCompetitionIndicator query normalization receives the offset. Masks, scores, sampling, six refinement paths, point-mass overlap, all channels, one shared confidence MLP and targets/losses, selector, original2D/released backbone/Q128/seed1999/fixed30/native YAML remain unchanged. No stage-count, loss-weight or validation-based sweep.',
        historical_scope='A parameter-sharing alternative to completed Y60 independent stage heads: retains exactly one head and supplies stage identity. Unlike Y67 per-stage mask/score readout affine normalization, only confidence changes; mask/score readout stays W2. Does not reopen either frozen experiment or rejected confidence routes, Y26 discarded norm or Y53 padding premise.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Run actual CPU native model/pretrain/TRAIN collation and independent stage-offset/zero/RNG/seven-stage output/input-gradient/confidence-only intervention controls. Retain seven tested files without moving active HEAD3c. Then prepare native two-step observations for exclusive authorized GPU; preserve Y72/Y73/Y74 and Y57 pause.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
