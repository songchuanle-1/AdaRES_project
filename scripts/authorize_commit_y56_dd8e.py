"""Apply the explicit seven-file user answer, then reuse scoped commit guards."""

from datetime import datetime
from research_y56_dd8e import ROOT, TOKEN, lease, ref, bind, tick, monitor
import authorize_prepared_candidates as scoped

ANSWER_REFERENCE = '["request_user_input_async","call_2e4f12e5ed594680a80c79a99b16f025",0]'
BUNDLE = 'pipeline/analyses/y56_bundle_rev2_' + TOKEN + '.json'
scoped.SPECS = {'AdaRES-Y56': ('adares_y56_unique_competitors', BUNDLE)}
scoped.MESSAGE_ID = ANSWER_REFERENCE

lease(300)
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT/'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'human_authorization'
    bundle_ref, files = scoped.checked_bundle('AdaRES-Y56', candidate)
    review = tick.load(ROOT/('pipeline/analyses/y56_gate_review_'+TOKEN+'.json'))
    assert review['scientific_prechecks_ready'] and all(g['passed'] for g in review['gates'].values())
    head = scoped.git('rev-parse', 'HEAD').decode().strip()
    assert head == review['local_HEAD']
    path = 'pipeline/authorizations/AdaRES-Y56_commit.json'
    assert not (ROOT/path).exists()
    now = datetime.now().astimezone().isoformat()
    authorization = dict(complete=True, id='AdaRES-Y56', kind='explicit_human_scoped_commit_authorization',
        recorded_at=now, thread_id='01a07e32-78bf-7670-925b-370911ea48db',
        human_message_id=ANSWER_REFERENCE, human_message='授权这 7 个文件的本地提交',
        context='Direct owning-user answer to the exact seven-file Y56 question in this current task. '
                'Question explicitly excludes push and unrelated files; existing admission/GPU checks remain required.',
        files=files, tested_bundle=bundle_ref, starting_HEAD=head,
        permits=['Local Git add/commit of only the seven exact tested Y56 revision2 files.'],
        excludes=['Other files/projects, remote push, scheduler changes, changed gates/goals/protocols.',
                  'Scientific PASS or training permission inferred from this commit answer.'])
    tick.save(ROOT/path, authorization)
    prior = {k:candidate[k] for k in ('blocker_kind','blocker','retry_after','unblock_condition') if k in candidate}
    candidate.setdefault('blocker_history', []).append(dict(**prior, resolved_at=now, resolved_by=ref(path)))
    for k in ('blocker_kind','blocker','retry_after','unblock_condition','required_server'):
        candidate.pop(k, None)
    candidate.update(stage='validation', next_action='Direct human seven-file commit consent received; '
                     'commit exact tested Y56 bundle, then revalidate six gates and existing authorized dispatch.')
    bind(candidate, ref(path), ref(__file__))
    work.pop('design_cycle',None)
    tick.save(ROOT/'pipeline/research_work.json',work)
finally:
    lock.close()
scoped.apply('AdaRES-Y56', TOKEN, 'commit')
