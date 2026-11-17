"""Review same-ID physical revision and exact uncommitted seven-file handoff."""

from datetime import datetime,timedelta
import subprocess,statistics,json
from research_y56_dd8e import ROOT,TOKEN,lease,ref,save,bind,tick,monitor
import closure

lease(120)
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
    bundle=tick.load(ROOT/candidate['tested_bundle']['path'])
    assert all(ref(r['path'])==r for r in bundle['files'])
    diag=ref('pipeline/analyses/y57_anchor_gpu1_'+TOKEN+'.json')
    cpu=ref('pipeline/analyses/y57_anchor_cpu_'+TOKEN+'.json')
    native=ref('pipeline/analyses/y57_anchor_physical_gpu1_'+TOKEN+'.json')
    result=tick.load(ROOT/native['path'])
    assert result['complete'] and result['physical_revised_package'] and result['parameter_delta']==66048
    assert len(result['rows'])==4 and all(r['finite_backward'] and r['all7_anchor_paths_reached'] for r in result['rows'])
    assert all(r['controls'][6]['matched_anchor_shift_signal']>r['controls'][6]['original_arithmetic_bound'] for r in result['rows'])
    assert all(r['peak_reserved_bytes']<24*2**30 for r in result['stress'])
    test=subprocess.run(['python','-B',str(ROOT/'scripts/test_y57_anchor_dd8e.py')],capture_output=True,text=True,check=True)
    structure=save('pipeline/analyses/y57_anchor_structure_'+TOKEN+'.json',dict(complete=True,tests=5,passed=True,
        output=test.stdout+test.stderr,tested_bundle=candidate['tested_bundle'],observed_at=datetime.now().astimezone().isoformat()))
    profiles=result['paired_native_profile']
    parent={r['scene']:r['median_seconds'] for r in profiles if r['role']=='parent'}
    ratios=[r['median_seconds']/parent[r['scene']] for r in profiles if r['role']=='candidate']
    decision=monitor.log_decision(str(ROOT),'manual_y57_connection_revision',
        'Y57晚层二阶矩原数值控制失败，初始证据保留连接是否可修复当前实现？',
        '原ID预注册一次连接修订；原4/4失败保留，初始完整256矩跨7层连接的独立实现/CPU/native/资源复测通过，待精确七文件本地提交确认。',
        '原769输入、1025宽度、原数值界、30轮seed1999和所有禁令不改；不是效益证明，不以Y56提交授权覆盖Y57。',native['path'])
    decision_ref=ref(decision)
    review=save('pipeline/analyses/y57_anchor_gate_review_'+TOKEN+'.json',dict(
        complete=True,id='AdaRES-Y57',revision=1,observed_at=datetime.now().astimezone().isoformat(),
        tested_bundle=candidate['tested_bundle'],files=bundle['files'],goal_sha256=ref('pipeline/goal.json')['sha256'],
        gates=dict(diagnosis=dict(passed=True,evidence=[diag],scope='Actual layerwise contrast collapse; original4failures retained.'),
            controls=dict(passed=True,evidence=[diag,cpu,native],scope='Same width/full256/same shift1 and original128eps bound; full joint permutation/repeat.'),
            native_backward=dict(passed=True,evidence=[native],scope='Physical revised package/all7sites/every head tensor/initial anchor finite nonzero native task gradients.'),
            resources=dict(passed=True,evidence=[native],scope='Delta66048 unchanged vs original Y57;5bounded largest-input runs below24GiB,8scene paired timing; not global Pareto.'),
            independence=dict(passed=True,evidence=[structure,candidate['tested_bundle']],scope='Independent six-file package/unique YAML; only indicator and two local DEC connections change.'),
            constraints=dict(passed=True,evidence=[structure,ref('pipeline/designs/exp_design_Y56_Y57_6debf88fed634a13b772f27bdc244bbd.md')],
                scope='All769parent inputs/full256moments, original loss, seed1999/fixed30/SPFormer/Q128/argmax; no tuning/new scale.')),
        decision=decision_ref,median_paired_latency_ratio=statistics.median(ratios),paired_latency_ratios=ratios,
        maximum_stress_reserved_bytes=max(r['peak_reserved_bytes'] for r in result['stress']),
        final_control_signals=[r['controls'][6]['matched_anchor_shift_signal'] for r in result['rows']],
        original_control_bound=1.52587890625e-5,all_original_negative_evidence_preserved=True,
        scientific_prechecks_ready=True,formal_uncommitted=True,admission=False,trained_efficacy='unknown',
        limitations=['8TRAIN scene clusters only; layers/queries not independent samples.',
                     'Connection effect/native gradients are not trained efficacy, harm, or novelty.',
                     'Paired timing is bounded and has background hardware variation; final Pareto not established.',
                     'No commit permission for Y57 inferred from the seven-file Y56 answer.']))
    revision=candidate['revision_history'][-1]
    outputs=[diag,cpu,native,structure,review]
    revision['evidence'].extend([r for r in outputs if r not in revision['evidence']])
    revision['result_evidence']=outputs
    revision['result']='passed'
    for row in outputs:bind(candidate,row)
    for name in ('y57_anchor_cpu','y57_anchor_physical_gpu1'):
        bind(candidate,ref('pipeline/analyses/'+name+'_'+TOKEN+'.log'))
    bind(candidate,decision_ref,ref('scripts/test_y57_anchor_dd8e.py'),ref(__file__))
    now=datetime.now().astimezone()
    candidate.update(stage='blocked',blocker_kind='human_authorization',
        blocker='Revised Y57 scientific prechecks completed; exact7tested local Git commit requires its own scoped user confirmation. Y56-only approval does not cover Y57.',
        retry_after=(now+timedelta(hours=4)).isoformat(),
        unblock_condition='Owning human approves local Git commit ONLY of exact7revised Y57 files, or supplies their exact commit; then revalidate receipt/current goal/live GPU and native progress. No push or changed protocol.',
        next_action='Ask once for revised Y57 exact seven-file local commit; do not repeat Y56 permission or old numerical hold. Preserve original failures; training efficacy unknown.',
        wake_paths=['pipeline/authorizations/AdaRES-Y57_commit.json'],
        scientific_status='revision1_physical_gates_reviewed_pending_scoped_commit',
        gate_summary={k:v['scope'] for k,v in tick.load(ROOT/review['path'])['gates'].items()})
    closure.validate_revisions(ROOT,candidate)
    work['design_cycle']=dict(stop_reason='external_resource_blocked',attempted_ids=['AdaRES-Y56','AdaRES-Y57'],
        evidence=[review,ref('pipeline/analyses/y56_dispatch_'+TOKEN+'.json')],
        resume_condition='Y56 continues verified fixed30 training; exact Y57 local-commit user confirmation wakes its existing scientific admission. No claim of GPU shortage, route exhaustion, or scientific benefit.')
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(review=review,tested_bundle=candidate['tested_bundle'],stage='blocked',
        final_signals=[r['controls'][6]['matched_anchor_shift_signal'] for r in result['rows']],
        median_latency_ratio=statistics.median(ratios),files=[r['path'] for r in bundle['files']]),indent=2))
finally:
    lock.close()
