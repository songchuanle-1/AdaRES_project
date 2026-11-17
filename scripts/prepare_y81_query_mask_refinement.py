"""Prepare independent refined-query-set attention into six later mask features; preserve runs."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID='AdaRES-Y81'
PACKAGE='adares_y81_query_mask_refinement'
PARENT='adares_w2_pointmass_competition'
NAMES=('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE='class QueryMaskRefinement(nn.Module):\n    """Attend from the original static mask base to current refined queries."""\n\n    def __init__(self, d_model):\n        super().__init__()\n        self.weight = nn.Parameter(torch.zeros(d_model, d_model))\n\n    def forward(self, visual, query, padding):\n        attention = torch.softmax(visual @ query.transpose(-1, -2) / (query.shape[-1] ** 0.5), dim=-1)\n        context = attention @ query\n        delta = (context @ self.weight).masked_fill(padding.unsqueeze(-1), 0.0)\n        return visual + delta\n\n\n'


def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def prepare(token):
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work=tick.load(ROOT/'pipeline/research_work.json');assert not any(c['id']==CID for c in work['candidates'])
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    source=(parent/'dec.py').read_text(encoding='utf-8')
    edits=[('class DEC(nn.Module):',MODULE+'class DEC(nn.Module):'),
        ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.query_mask_refinement = QueryMaskRefinement(d_model)'),
        ('            pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)',
         '            refined_mask_feats = self.query_mask_refinement(mask_feats, query, batch_mask)\n            pred_scores, pred_masks, attn_masks = self.prediction_head(query, refined_mask_feats, batch_mask)')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=source.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y81: current refined-query-set attention into six later mask feature tensors.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw);ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_y81_cpu.py';program=ROOT/('pipeline/checks/y81_cpu_'+token+'_program.py');history=program.with_name('y81_targeted_history_'+token+'.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
    phrases=['query-to-visual feedback','current query set to visual','visual attends to queries','opposite direction to query-visual update gate']
    matches=[dict(id=c['id'],stage=c['stage'],hypothesis=c.get('hypothesis','')[:1000],matched_control=c.get('matched_control','')[:1200]) for c in work['candidates'] if c['id'] in ('AdaRES-Y25','AdaRES-Y26','AdaRES-Y30','AdaRES-Y33','AdaRES-Y38','AdaRES-Y40','AdaRES-Y65','AdaRES-Y66','AdaRES-Y68','AdaRES-Y71','AdaRES-Y72','AdaRES-Y73','AdaRES-Y74','AdaRES-Y75','AdaRES-Y77','AdaRES-Y78','AdaRES-Y79','AdaRES-Y80')]
    tick.save(history,dict(complete=True,scope='Targeted candidate-text comparison only; not exhaustive novelty or literature evidence.',phrases=phrases,matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='Native W2 predicts refined masks from each current query against expression-independent static mask features. A shared attention residual from the full current refined query set into each later mask feature tensor may ease instance-conditioned mask fitting. Existing language/query paths can carry equivalent context; missing information, novelty and trained efficacy are not presumed.',
        matched_control='Independent unchanged W2 plus mask_features_i=mask_base+clean((softmax(mask_base@current_refined_queries.T/sqrt(d))@current_refined_queries)@W), shared W[256,256] exactly zero/no RNG (+65536). Six calls before original loop prediction heads; original initial prediction unchanged, original static mask base reused without accumulating updates, padded residual rows zero. All128 queries and all channels retained and undetached. Original sampling/geometry/coarse-fine equal fusion/Boolean gate algorithm/SCG/k/gamma, seven shared confidence heads/targets/losses/argmax, all inputs/existing2D/released backbone/Q128/seed1999/fixed30/native YAML remain. Fixed standard sqrt(d) scaling only; no temperature/threshold/loss/feature/epoch/validation search.',
        historical_scope='Y72 maps current visual states into mask features; Y81 conditions mask features on the full current refined query set via attention. Y68 performs one pooled-language multiplication of static mask features, Y79 changes the language stream, Y80 changes visual states before coarse/fine attention, and Y78 adds previous mask logits. These remain separate frozen experiments, not combined or reopened. This is not a rejected final confidence/history/visual readout, soft training-only attention gate, discarded-normalization reconnection or padding-leakage route. Targeted source/history comparison is not exhaustive novelty evidence.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        next_action='Execute CPU native YAML/full model/released backbone/actual TRAIN collation, independent scalar attention-context-projection and zero/RNG/input-gradient/permutation/padding/mispaired-query/equal-mean different-distribution controls, all-seven-stage W2 anchors, six exact original static mask base/current refined query/padding identities and actual prediction consumers, unchanged original hard-gate algorithm and six coarse consumers, seven shared confidence/query_norm consumers, nonzero/bypass and two synthetic CPU decoder optimizer steps. Retain exact tested files and prepare native two-step task plans. Preserve HEAD e0c and original Y77/Y78/Y79. No CPU-based GPU PASS/admission/efficacy.',
        evidence=[ref(p) for p in paths]+[ref(program),ref(body),ref(history),ref(__file__)]))
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(candidate=CID,program=ref(program),native_GPU_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
