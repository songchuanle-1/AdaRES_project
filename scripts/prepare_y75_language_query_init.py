"""Prepare a language-conditioned initial-query interaction, preserving runs."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y75'
PACKAGE='adares_y75_language_query_init'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE='''class LanguageQueryInitialization(nn.Module):
    """One direct pooled-language interaction before the initial prediction."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, query, language, padding):
        clean = language.masked_fill(padding.unsqueeze(-1), 0.0)
        count = (~padding).sum(1, keepdim=True).clamp_min(1).to(language.dtype)
        context = clean.sum(1) / count
        return query + (query * context.unsqueeze(1)) @ self.weight


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
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.language_query_init = LanguageQueryInitialization(d_model)'),
        ('\n        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)',
         '\n        query = self.language_query_init(query, lang_feats, lang_masks)\n        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y75: language interaction in original sampled-query initialization.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y75_cpu.py';program=ROOT/('pipeline/checks/y75_cpu_'+token+'_program.py');history=program.with_name('y75_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    phrases=['query initialization','query initialisation','query generator','initial query','language-conditioned query','query-language']
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis','')[:800]) for c in work['candidates']
        if c['id'] in ('AdaRES-Y26','AdaRES-Y53','AdaRES-Y68','AdaRES-Y71','AdaRES-Y73','AdaRES-Y74') or any(p in ' '.join(str(c.get(k,'')) for k in ('hypothesis','matched_control','historical_scope')).lower() for p in phrases)]
    tick.save(history,dict(complete=True,scope='Targeted candidate-text comparison only; not exhaustive novelty or literature evidence.',phrases=phrases,matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='Native sampling already uses language, while the sampled-seed query generator produces the initial query before later query-language refinement. A direct product interaction between the generated query and all valid projected language tokens pooled once may ease language-conditioned initial mask fitting. Native attention and sampling already encode language; missing information, novelty and efficacy are not presumed.',
        matched_control='Unchanged W2 plus q_init=q_generated+(q_generated*mean_valid(original projected language))@W, once after original query generation and before the initial prediction. W[256,256] exactly zero/no RNG (+65536). Preserve original sampling indices/query positions, all valid tokens and native padding, static mask features, six original refinement stages, coarse/fine fusion/hard gates/SCG/k/gamma, seven shared confidence calls/targets/losses/selector, full input channels/existing2D/released SPFormer/Q128/seed1999/fixed30/native YAML. Original stage0 and all later predictions consume the initialized query; this is a direct initialization interaction, not a confidence-only readout.',
        historical_scope='Unlike Y68 pooled-language modulation of static mask features, Y71 early two-visual-branch products, Y73 repeated original-query anchoring or Y74 gating six attended-language increments. Does not reconnect discarded qla norm (Y26), assert native padding leakage (Y53), change confidence evidence/attention temperatures/token or feature subsets, or reopen a rejected/banned ID.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Run actual CPU native YAML/full model/released backbone/TRAIN collation, independent pooled-language product controls, zero/RNG/input-gradient and all-seven-stage anchors, original query-generator/projected-language/padding identities and real initial prediction consumer, new gradient and bypass. Retain exact tested source and prepare actual native two-step checks; preserve live HEAD7da/three runs and prioritize Y74 when resources open.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
