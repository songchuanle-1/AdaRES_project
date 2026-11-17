"""Prepare independent stage-private recurrent query normalization; preserve runs."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y83'
PACKAGE='adares_y83_stage_query_norm'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')



def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source='import copy\n'+(parent/'dec.py').read_text(encoding='utf-8')
    edits=[('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.refinement_query_norms = nn.ModuleList([copy.deepcopy(self.norm_qqa) for _ in range(self.num_layer-1)])'),
        ('            query = self.norm_qqa(query_self + query_qla)',
         '            query = (self.norm_qqa if i == 0 else self.refinement_query_norms[i-1])(query_self + query_qla)')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y83: private affine LayerNorm in six original self/language query fusion updates.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y83_cpu.py';program=ROOT/('pipeline/checks/y83_cpu_'+token+'_program.py');history=program.with_name('y83_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis',''),matched_control=c.get('matched_control','')) for c in work['candidates'] if c['id'] in ('AdaRES-Y26','AdaRES-Y58','AdaRES-Y60','AdaRES-Y67','AdaRES-Y73','AdaRES-Y76','AdaRES-Y79','AdaRES-Y80','AdaRES-Y81','AdaRES-Y82')]
    tick.save(history,dict(complete=True,scope='Targeted architecture/source comparison only; not exhaustive novelty.',matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='The active norm_qqa affine transform is shared across six recurrent self/language query fusion updates. Independent per-stage copies with identical LayerNorm statistics/epsilon may ease stage-specific representation fitting. Existing stage FFNs can compensate; trained efficacy, missing information and novelty are not presumed.',
        matched_control='Independent unchanged W2. Keep original norm_qqa at refinement stage0 and copy it without RNG into five private refinement_query_norms for stages1..5 (+2560). Only affine parameter sharing in LayerNorm(query_self+query_qla) changes. Initial state and all seven mask/score/confidence predictions match W2; native eps/dtype/full query channels/Q128, shared out_norm/confidence MLP, static mask, initial prediction, hard attention algorithm/sampling/SCG/k/gamma/geometry, original full input/existing2D/released backbone/targets/loss/selector/seed1999/fixed30/native YAML remain. No scans of parameters or validation-selected schedules/features/thresholds/losses.',
        historical_scope='Y67 unties output prediction_head out_norm across seven readouts; Y83 unties the already active norm_qqa inside six recurrent state updates while preserving shared out_norm. Y60 unties confidence heads and Y76 conditions the confidence input; neither is combined. Rejected Y26 reconnected a discarded qla LayerNorm, whereas this route leaves that qla path untouched and only unties an existing consumed operation. No reopening of Y26 or other rejected routes; no exhaustive novelty claim.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Run CPU original native model/released backbone/TRAIN collation and scalar affine LayerNorm controls, disjoint parameter/no RNG/initial anchor/input-gradient and aggregate shared-gradient checks, exact six original self/language-sum inputs and real FFN consumers, shared confidence/out_norm/hard gates, private intervention/restoration and two synthetic CPU decoder optimizer steps. Retain tested source and native task plan; wait exclusive authorized GPU. Preserve Y79/Y80/Y81 and Y57 pause. CPU never proves native GPU PASS/admission/efficacy.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
