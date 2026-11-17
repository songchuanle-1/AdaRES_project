"""Prepare one independent query-conditioned language-update gate, keeping HEAD."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y74'
PACKAGE='adares_y74_language_update_gate'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE='''class LanguageUpdateGate(nn.Module):
    """Condition the existing attended-language update on its incoming query."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, query_language, query_visual):
        gate = torch.tanh(query_visual @ self.weight)
        return query_language + gate * (query_language - query_visual)


'''

def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source=(parent/'dec.py').read_text(encoding='utf-8')
    replacements=[('class DEC(nn.Module):',MODULE+'class DEC(nn.Module):'),
        ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.language_update_gate = LanguageUpdateGate(d_model)'),
        ('            query_qla, _, _ = self.qla_layers[i](lang_query, query_qia, lang_masks)',
         '            query_qla, _, _ = self.qla_layers[i](lang_query, query_qia, lang_masks)\n            query_qla = self.language_update_gate(query_qla, query_qia)')]
    for old,new in replacements:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw=source.encode() if path.name=='dec.py' else (parent/path.name).read_bytes()
        if path.name=='__init__.py':raw=b'# AdaRES-Y74: query-conditioned attended-language update gate.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw);ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(path):return dict(base64=base64.b64encode(path.read_bytes()).decode(),sha256=ref(path)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y74_cpu.py';program=ROOT/('pipeline/checks/y74_cpu_'+token+'_program.py');history=program.with_name('y74_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    phrases=['language update','language gate','language gating','gated language','qla','query-language']
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis','')[:800]) for c in work['candidates']
        if c['id'] in ('AdaRES-Y26','AdaRES-Y53','AdaRES-Y68','AdaRES-Y70','AdaRES-Y73') or any(p in ' '.join(str(c.get(k,'')) for k in ('hypothesis','matched_control','historical_scope')).lower() for p in phrases)]
    tick.save(history,dict(complete=True,scope='Targeted candidate text and known related controls only; not exhaustive novelty or literature evidence.',phrases=phrases,matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='The six existing query-language cross-attention outputs combine an incoming visual query with an attended-language update. A shared query-conditioned channel-wise gate on that existing update may ease fitting cross-modal updates. Native attention already depends on the query; no missing information, novelty or trained benefit is presumed.',
        matched_control='Unchanged W2 plus q_after=q_language+tanh(q_visual@W)*(q_language-q_visual) immediately after each original qla output and before original qqa. W[256,256] is exactly zero without RNG (+65536). All six qla outputs and incoming q_visual tensors stay undetached; original qqa consumes the new tensor. Initial stage0, static mask features, original language padding, original discarded qla norm semantics, coarse/fine averaging/hard Boolean masks/SCG/k/gamma, seven shared confidence calls/targets/losses/selector, full original inputs/existing2D/released backbone/Q128/seed1999/fixed30/native YAML stay unchanged.',
        historical_scope='This gates the attended-language increment after native qla, not Y70 coarse/fine visual fusion, Y68 language-conditioned static mask features, Y73 original-query anchoring or Y71 early visual products. It does not reconnect the discarded LayerNorm rejected under Y26, assert the padding-leakage premise rejected under Y53, alter attention temperature or mask/feature subsets, or reopen a rejected ID.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Run the prepared CPU native-YAML full-model/released-backbone/real TRAIN collation and independent gate controls, full seven-stage W2/RNG/input-gradient anchors, six original qla-output/incoming-query identities and actual qqa consumers, finite new gradient/step response and bypass. Then retain exact tested version and prepare native two-step program. GPU native task loss/backward/optimizer and fresh idle authorized resources remain mandatory before admission; preserve all current training.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
