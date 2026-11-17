"""Prepare an independent current-query-set to language feedback residual; preserve runs."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y79'
PACKAGE='adares_y79_query_language_feedback'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE='class QueryLanguageFeedback(nn.Module):\n    """A direct current-query-set attention residual into native language states."""\n\n    def __init__(self, d_model):\n        super().__init__()\n        self.weight = nn.Parameter(torch.zeros(d_model, d_model))\n\n    def forward(self, language, query, padding):\n        attention = torch.softmax(language @ query.transpose(-1, -2) / (query.shape[-1] ** 0.5), dim=-1)\n        context = attention @ query\n        delta = (context @ self.weight).masked_fill(padding.unsqueeze(-1), 0.0)\n        return language + delta\n\n\n'


def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source=(parent/'dec.py').read_text(encoding='utf-8')
    edits=[('class DEC(nn.Module):',MODULE+'class DEC(nn.Module):'),
        ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.query_language_feedback = QueryLanguageFeedback(d_model)'),
        ('            inst_feats = self.ila_ffn_layers[i](inst_feats)',
         '            inst_feats = self.ila_ffn_layers[i](inst_feats)\n            lang_query = self.query_language_feedback(lang_query, query, lang_masks)')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y79: current-query-set attention feedback into six native language states.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y79_cpu.py';program=ROOT/('pipeline/checks/y79_cpu_'+token+'_program.py');history=program.with_name('y79_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    phrases=['query-to-language feedback','current query set to language','language attends to queries','opposite direction to query-language update gate']
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis','')[:1000],matched_control=c.get('matched_control','')[:1200]) for c in work['candidates'] if c['id'] in ('AdaRES-Y25','AdaRES-Y26','AdaRES-Y30','AdaRES-Y33','AdaRES-Y38','AdaRES-Y40','AdaRES-Y65','AdaRES-Y66','AdaRES-Y68','AdaRES-Y73','AdaRES-Y74','AdaRES-Y75','AdaRES-Y77','AdaRES-Y78')]
    tick.save(history,dict(complete=True,scope='Targeted candidate-text comparison only; not exhaustive novelty or literature evidence.',phrases=phrases,matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='Native language states refine through language self-attention and visual cross-attention before providing query-language updates, without a direct current instance-query-set input. An added attention residual from all current queries into language states may ease iterative instance-aware language fitting. Existing visual/query paths can carry equivalent context; missing information, novelty and trained efficacy are not presumed.',
        matched_control='Independent unchanged W2 plus language_next=language_ffn_output+clean((softmax(language_ffn_output@current_query.T/sqrt(d))@current_query)@W), shared W[256,256] exactly zero/no RNG (+65536), six calls after original language/visual FFNs and before original query coarse/fine and query-language attention. Padded language residual rows zeroed, all128 original queries/all language channels retained and undetached. Original initial prediction, static masks, sampling/positions, six refinements/coarse-fine mixture/hard Boolean gates/SCG/k/gamma, seven shared confidence targets/losses/argmax, all inputs/existing2D/released backbone/Q128/seed1999/fixed30/native YAML remain. Standard fixed sqrt(d) in this new attention; no temperature, threshold, feature, loss, epoch or validation search.',
        historical_scope='This adds the query-set to language direction, unlike Y74 gating the language-to-query output or Y75 a pooled-language product in initial query formation. Y65 confidence feedback/Y66 peer context into queries, Y68 mask-feature modulation and Y78 prior-mask logits remain separate frozen experiments. It is not Y33/Y40 final confidence readback/alignment, Y30/Y38 final history readouts, Y26 discarded normalization reconnection or a Y53 padding-leakage claim; no rejected ID is reopened or combined. Targeted source/history comparison is not exhaustive novelty evidence.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Execute CPU native YAML/full model/released backbone/actual TRAIN collation, independent scalar softmax-context-projection, zero/RNG/input-gradient and query-set/token/batch/channel permutation controls, padded-row interventions/all-padding/wrong-query-batch and equal-query-mean different-distribution controls, all-seven-stage W2 anchors and exact six current-query/language-FFN/padding identities, original qla and next-lla consumption, original shared confidence/query_norm/static-mask identities, finite new gradients and two synthetic CPU decoder optimizer steps. Retain tested source and prepare exact native TRAIN two-step checks; no CPU-based admission/GPU PASS or efficacy. Preserve active HEAD498e and Y75/Y76/Y77.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
