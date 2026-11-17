"""Register a bounded Y57 connection revision based on actual layer contrast loss."""

import ast
from datetime import datetime
from research_y56_dd8e import ROOT,TOKEN,lease,ref,save,bind,tick,monitor

lease(300)
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
    assert candidate['stage']=='blocked' and candidate['blocker_kind']=='numerical_control_failure'
    assert not candidate.get('revision_history')
    old='pipeline/analyses/y57_cpu_5103a354430e47b3b7a04162dcaae8af_program.py'
    tree=ast.parse((ROOT/old).read_text(encoding='utf-8'))
    payload=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets)))
    payload['commit']='200486ee12bcf3d7e82393d009d06214dfe83e38'
    setup=ROOT/'scripts/setup_y57_prototype_5103.py'
    body=ROOT/'scripts/probe_y57_anchor_dd8e.py'
    path='pipeline/analyses/y57_anchor_'+TOKEN+'_program.py'
    program='PAYLOAD='+repr(payload)+'\n'+setup.read_text(encoding='utf-8')+'\n'+body.read_text(encoding='utf-8')
    ast.parse(program,feature_version=(3,8))
    assert not (ROOT/path).exists()
    (ROOT/path).write_text(program,encoding='utf-8',newline='\n')
    failure=ref('pipeline/analyses/y57_activity_audit_5103a354430e47b3b7a04162dcaae8af.json')
    stamp=datetime.now().astimezone().isoformat()
    plan=save('pipeline/analyses/y57_anchor_prereg_'+TOKEN+'.json',dict(
        complete=True,id='AdaRES-Y57',revision=1,preregistered_at=stamp,failure=failure,
        defect='Actual released-init early-layer shift2.3e-3 but final4.7e-6..6.2e-6: late peer moments '
               'lose query-specific contrast while all tensor gradients remain finite; numerical failure is retained.',
        change='Keep original769inputs and full256second moment vector. Bind it at the initial query layer '
               'before subsequent mixing; use that SAME position-aligned vector at all7indicator sites. '
               'No layer search, new scale/temperature, feature-channel selection, width or loss change.',
        prediction='Same final-layer query-shift control exceeds UNCHANGED128*eps bound for the anchored vector; '
                   'all native head tensors and initial anchor receive finite nonzero task gradients.',
        controls='Same4original TRAIN batch2 pairs/32expressions: original/current-moment failure retained, '
                 'anchor and independent repeat. Final same-width shift1/full256, original bound unchanged.',
        program=ref(path),body=ref(body),setup=ref(setup),
        pending_physical_realization_and_resources=True,result='pending'))
    candidate['revision_history']=[dict(revision=1,failure_scope='implementation',
        reason='Late-layer per-query second-moment contrast collapses; original numerical control remains failed.',
        change='Move SAME full256peer moment evidence readout to initial query layer, retain across7confidence sites.',
        prediction='Preserved initial contrast reaches final native loss and passes same original shifted-moment bound.',
        preregistered_at=stamp,failure_evidence=[failure],result='pending',evidence=[plan],result_evidence=[])]
    bind(candidate,plan,ref(path),ref(body),ref(__file__))
    prior={k:candidate[k] for k in ('blocker_kind','blocker','retry_after','unblock_condition') if k in candidate}
    candidate.setdefault('blocker_history',[]).append(dict(**prior,reviewed_at=stamp,
        disposition='Original failure retained; a newly preregistered same-ID local connection revision is now executable.',
        revision_preregistration=plan))
    for key in ('blocker_kind','blocker','retry_after','unblock_condition'):candidate.pop(key,None)
    candidate.update(stage='implementation',next_action='Execute preregistered initial-moment connection diagnostic '
        'on idle authorized GPU1; preserve original failure. If justified, implement SAME Y57 independently and retest all affected gates.',
        failure_scope='implementation')
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Preregistered SAME Y57 revision1; original numerical failure and bound retained.')
finally:
    lock.close()
