"""Prepare independent normalization of all original visual mask-feature channels."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y85'
PACKAGE='adares_y85_mask_feature_norm'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')


def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source=(parent/'dec.py').read_text(encoding='utf-8')
    edits=[('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
        '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.mask_feature_norm = nn.LayerNorm(d_model)'),
        ('        mask_feats = self.x_mask(x)','        mask_feats = self.mask_feature_norm(self.x_mask(x))')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y85: LayerNorm over all original x_mask feature channels before batching.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y85_cpu.py';program=ROOT/('pipeline/checks/y85_cpu_'+token+'_program.py');history=program.with_name('y85_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis',''),matched_control=c.get('matched_control','')) for c in work['candidates'] if c['id'] in ('AdaRES-Y26','AdaRES-Y41','AdaRES-Y67','AdaRES-Y68','AdaRES-Y71','AdaRES-Y72','AdaRES-Y81','AdaRES-Y83','AdaRES-Y84')]
    tick.save(history,dict(complete=True,scope='Targeted normalization location/consumer comparison; not exhaustive novelty.',matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='Original query readouts use LayerNorm while x_mask visual embeddings retain per-superpoint mean and magnitude. A single learned affine LayerNorm of all visual mask-feature channels before batching may change mask-fitting geometry. Removing magnitude/mean may also harm fitting; no native defect, missing information, novelty or efficacy is presumed.',
        matched_control='Independent unchanged W2 except mask_feats=LayerNorm(x_mask(x)) once before original get_batches, used by all seven original mask/score readouts. New512 affine parameters, default original LayerNorm epsilon1e-5, no RNG. Keep all256channels/all original superpoints, original query/out_norm/confidence sharing, original coordinates/full input/existing2D/backbone/sampling/SCG/k/gamma/static feature reuse/hard-gate algorithm/targets-loss-selector/seed1999/fixed30/native YAML. Native masks and gates may differ as the intended transform; no threshold/loss/epoch/feature scan. Bypassing only the new norm must restore all W2 outputs and input gradients.',
        historical_scope='Y67 unties existing query out_norm across readouts; Y83 unties existing recurrent norm_qqa; this adds one shared normalization to visual x_mask outputs before batching. Rejected Y26 reconnected a discarded query-language norm, which remains untouched here. Y41 changes boundary/interior message normalization, not visual mask-feature channel statistics. No reopening or combination with those routes.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Execute CPU original native YAML/model/backbone/TRAIN collation, scalar float32/64 affine LayerNorm/no-RNG/permutation/shift controls; verify bypass restores seven-stage W2 and input gradients, active transform changes intended masks, original x_mask/norm/get_batches/prediction identities, shared confidence/query_norm/hard-gate consumers and two synthetic CPU optimizer steps. Preserve Y81/Y82/Y83 and Y57 pause. Actual native GPU task PASS is still required before admission; CPU is not efficacy.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
