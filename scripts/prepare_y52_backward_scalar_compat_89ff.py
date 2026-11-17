"""Freeze an output-only scalar cast, preserving all previous failed evidence."""
import ast
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
from prepare_y52_native_backward_89ff import ref, TOKEN
import foreground_tick as tick
import pipeline_monitor as monitor

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
    assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
    assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 900
    for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
        assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
    assert candidate['stage'] == 'validation'
    for row in candidate['evidence']:
        assert ref(row['path']) == row
    old_program = 'pipeline/analyses/y52_native_backward_buffer_compat_program_' + TOKEN + '.py'
    old_plan = 'pipeline/analyses/y52_native_backward_buffer_compat_prereg_' + TOKEN + '.json'
    failure_path = 'pipeline/analyses/y52_backward_buffer_compat_89ff_' + TOKEN + '.log'
    failure = (ROOT / failure_path).read_text(encoding='utf-8')
    assert 'Object of type Tensor is not JSON serializable' in failure
    source = (ROOT / old_program).read_text(encoding='utf-8')
    old = "expressions=sum(batch['scenes_len'])"
    assert source.count(old) == 1
    source = source.replace(old, "expressions=int(sum(batch['scenes_len']))")
    ast.parse(source, feature_version=(3, 8))
    program_path = 'pipeline/analyses/y52_native_backward_scalar_compat_program_' + TOKEN + '.py'
    plan_path = 'pipeline/analyses/y52_native_backward_scalar_compat_prereg_' + TOKEN + '.json'
    assert not (ROOT / program_path).exists() and not (ROOT / plan_path).exists()
    plan = dict(complete=True, id='AdaRES-Y52', token=TOKEN,
        preregistered_at=datetime.now().astimezone().isoformat(), original_program=ref(old_program),
        original_plan=ref(old_plan), preserved_failure=ref(failure_path),
        failure_scope='implementation: diagnostic JSON scalar, not model or gradient failure',
        change='Cast only the observed scalar sum(scenes_len) to Python int before serializing per-batch results. Every scientific operation, assertion, sample, state, matcher and optimizer remains unchanged.',
        prediction='Partial and final records serialize without a Tensor-type error; actual native/compatibility/gradient/restore assertions remain active and may still fail honestly.',
        candidate_model_revisions=0, result='pending')
    (ROOT / program_path).write_text(source, encoding='utf-8', newline='\n')
    plan['program'] = ref(program_path)
    tick.save(ROOT / plan_path, plan)
    for path in (failure_path, 'scripts/prepare_y52_backward_scalar_compat_89ff.py', program_path, plan_path):
        candidate['evidence'].append(ref(path))
    candidate['diagnostic_revision_history'].append(dict(scope='output_scalar_not_model',
        preregistered_at=plan['preregistered_at'], change=plan['change'], prediction=plan['prediction'],
        failure_evidence=[ref(failure_path)], evidence=[ref(plan_path)], result='pending'))
    candidate['next_action'] = 'Execute unchanged paired native backward/restore protocol with preregistered diagnostic scalar serialization fix. Preserve all failures; no model revision or scientific PASS yet.'
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(program=ref(program_path), plan=ref(plan_path),
        output_only_change=True, candidate_model_revisions=0), indent=2))
finally:
    lock.close()
