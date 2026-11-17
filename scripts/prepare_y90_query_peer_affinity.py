"""Prepare an independent bounded learned compatibility in existing peer aggregation."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick

CID = 'AdaRES-Y90'
PACKAGE = 'adares_y90_query_peer_affinity'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py')
MODULE = '''class QueryPeerAffinity(nn.Module):
    """Neutral bounded compatibility for existing overlap-weighted query peers."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, query):
        score = torch.einsum('bid,de,bje->bij', query, self.weight, query)
        return 1 + torch.tanh(score / (query.shape[-1] ** .5))


'''
OLD_AGGREGATION = '''        competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
        normalized_query = self.query_norm(query)
'''
NEW_AGGREGATION = '''        normalized_query = self.query_norm(query)
        weighted_overlap = overlap * self.peer_affinity(normalized_query)
        competition = weighted_overlap / weighted_overlap.sum(-1, keepdim=True).clamp_min(1e-6)
'''
EDITS = [('class QueryCompetitionIndicator(nn.Module):',MODULE+'class QueryCompetitionIndicator(nn.Module):'),
    ('        self.evidence_norm = nn.LayerNorm(3 * d_model + 1)',
     '        self.evidence_norm = nn.LayerNorm(3 * d_model + 1)\n        self.peer_affinity = QueryPeerAffinity(d_model)'),
    (OLD_AGGREGATION,NEW_AGGREGATION)]


def files():
    return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/name for name in NAMES]+[
        ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]


def bodies():
    cpu,anchor,_ = (ROOT/'scripts/probe_y83_cpu.py').read_text(encoding='utf-8').partition('parameters=sum(')
    assert anchor
    core = (ROOT/'scripts/y90_observer_core.py').read_text(encoding='utf-8')
    cpu += (ROOT/'scripts/probe_y90_cpu_controls.py').read_text(encoding='utf-8')
    cpu += '\n'+core+'\n'+(ROOT/'scripts/probe_y90_cpu_tail.py').read_text(encoding='utf-8')
    native,anchor,_ = (ROOT/'scripts/probe_y83_native_smoke.py').read_text(encoding='utf-8').partition('losses=[];')
    assert anchor
    native = native.replace("'AdaRES-Y83'","'AdaRES-Y90'").replace('164632197','164695173')
    native += '\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n'
    native += 'head=model.dec.indi_embedding;affinity=head.peer_affinity;original_prediction=model.dec.prediction_head\n'
    native += core+'\n'
    tail = (ROOT/'scripts/probe_y88_native_tail.py').read_text(encoding='utf-8')
    tail = tail.replace('projection.weight','affinity.weight').replace('projection_paths','affinity_paths')
    tail = tail.replace('projection_calls=14, original_mask_consumers=14,\n    original_score_consumers=14, shared_head_calls=14, query_norm_consumption_calls=14,',
        'peer_gate_calls=14, original_overlap_consumers=14,\n    weighted_context_consumers=14, shared_MLP_calls=14, shared_query_norm_calls=14, original_evidence_norm_calls=14,')
    tail = tail.replace('static_mask_features_preserved=True,',
        'static_mask_features_preserved=True, original_competition_overlap_and_max_preserved=True,')
    native += tail
    for text in (cpu,native):
        ast.parse(text,feature_version=(3,8))
    return cpu,native


def prepare(token):
    view = tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
    assert view['experiment_first']['enabled'] and view['research']['creation_budget']>=1
    work = tick.load(ROOT/'pipeline/research_work.json')
    assert not any(candidate['id']==CID for candidate in work['candidates'])
    paths = files()
    parent = paths[0].parent.parent/PARENT
    assert not paths[0].parent.exists() and not any(path.exists() for path in paths)
    source = (parent/'dec.py').read_text(encoding='utf-8')
    for old,new in EDITS:
        assert source.count(old)==1
        source = source.replace(old,new)
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name=='dec.py' else (parent/path.name).read_bytes()
        if path.name=='__init__.py':
            raw = b'# AdaRES-Y90: bounded learned query-pair compatibility for original confidence peers.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw)
        ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes())
    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(),sha256=ref(path)['sha256'])
    payload = dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),
        python_files={path.name:encoded(path) for path in paths[:-1]},config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    cpu,native = bodies()
    program = ROOT/('pipeline/checks/y90_cpu_'+token+'_program.py')
    native_path = ROOT/('pipeline/checks/y90_native_'+token+'_body.py')
    history = ROOT/('pipeline/checks/y90_targeted_history_'+token+'.json')
    assert not any(path.exists() for path in (program,native_path,history))
    program.write_text('PAYLOAD='+repr(payload)+'\n'+cpu,encoding='utf-8',newline='\n')
    native_path.write_text(native,encoding='utf-8',newline='\n')
    matches = [dict(id=candidate['id'],stage=candidate['stage'],hypothesis=candidate.get('hypothesis',''),
        matched_control=candidate.get('matched_control','')) for candidate in work['candidates']
        if candidate['id'] in ('AdaRES-Y36','AdaRES-Y37','AdaRES-Y57','AdaRES-Y61','AdaRES-Y63',
                              'AdaRES-Y64','AdaRES-Y66','AdaRES-Y86','AdaRES-Y89')]
    tick.save(history,dict(complete=True,scope='Targeted consumer/mechanism comparison, not novelty proof.',
        matches=matches,parent_source=ref(parent/'dec.py')))
    work['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',
        hypothesis='Original confidence peers are pooled using only row-normalized soft mask overlap. A shared bounded learned compatibility of the existing normalized focal/peer query pair may ease fitting which overlapping peers influence confidence. Existing query and shared MLP can implicitly encode interactions; no defect, missing information, novelty or trained benefit is presumed.',
        matched_control='Independent W2, Aij=1+tanh(qnorm_i @ zeroW256x256 @ qnorm_j /sqrt(256)); context row-normalizes original off-diagonal overlap*A. Shared W/no RNG/+65536. Retain original overlap/max-overlap, masks/scores/queries, all Q128 peers/full channels/point masses/shared head and norms, original data/2D/releasedbackbone/sampling/SCG/gates/targets-loss-selector/seed1999/fixed30/native YAML. Initial all-output/input-gradient anchors exact; nonzero affinity/bypass controls, exact input/consumer/gradient/two updates required. Individual-stage row-constant gates may cancel; require aggregate nonzero learned context response over seven readouts, not fabricated per-stage efficacy. No temperature/threshold/degree/weight/validation scan.',
        historical_scope='Y63 adds focal/context bilinear logits after aggregation; Y90 changes only pair weighting before the SAME existing peer mean. Y64 adds directional containment context; Y66 feeds peer context into recurrence. Y36/Y37 spatial boundary/pair-distance summaries remain separate. User-paused Y57 peer-second-moment route and all banned/rejected routes are not resumed or combined. Y89 private-normalization preparation remains separate.',
        comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        prepared_native_body=ref(native_path),observer_core=ref('scripts/y90_observer_core.py'),
        next_action='Run original CPU native YAML/model/backbone/TRAIN loading, scalar bilinear/bounded/directed/no-RNG/permutation controls, seven-stage W2/input-gradient anchors, confidence-only intervention/bypass and exact overlap/affinity/context/head consumers with two synthetic updates. Retain exactly tested version and GPU0 native plans. Preserve original Y87/Y88 and earlier prepared Y89; no CPU-based admission or efficacy.',
        evidence=[ref(path) for path in paths]+[ref(program),ref(native_path),ref(history),ref(__file__),
            ref('scripts/y90_observer_core.py'),ref('scripts/probe_y90_cpu_controls.py'),ref('scripts/probe_y90_cpu_tail.py')]))
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(candidate=CID,program=ref(program),native_body=ref(native_path),native_executed=False)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token',required=True)
    args = parser.parse_args()
    lease(args.token,180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        prepare(args.token)
    finally:
        writer.close()
