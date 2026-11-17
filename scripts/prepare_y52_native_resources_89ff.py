"""Freeze native paired predict and resource checks after actual backward/restore."""
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
    assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 400
    for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
        assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
    assert candidate['stage'] == 'validation'
    for row in candidate['evidence']:
        assert ref(row['path']) == row
    backward_path = 'pipeline/analyses/y52_backward_scalar_compat_89ff_' + TOKEN + '.json'
    result = tick.load(ROOT / backward_path)
    assert result['complete'] and result['actual_native_sparse_forward_and_backward']
    assert all(result['nonempty_optimizer_restore'].values()) and len(result['optimizer_steps']) == 2
    assert all(row['native_backward_finite'] for row in result['rows'])
    bound_program = 'pipeline/analyses/y52_native_matched_capture_binding_v2_' + TOKEN + '.py'
    tree = ast.parse((ROOT / bound_program).read_text(encoding='utf-8'))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets))
    payload = ast.literal_eval(assignment.value)
    setup = (ROOT / 'scripts/probe_y52_native_matched_capture.py').read_text(encoding='utf-8').split('rows = []\n', 1)[0]
    body_path = 'scripts/probe_y52_native_resources_89ff.py'
    source = 'PAYLOAD=' + repr(payload) + '\n' + setup + '\n' + (ROOT / body_path).read_text(encoding='utf-8')
    ast.parse(source, feature_version=(3, 8))
    program_path = 'pipeline/analyses/y52_native_resources_program_' + TOKEN + '.py'
    plan_path = 'pipeline/analyses/y52_native_resources_prereg_' + TOKEN + '.json'
    assert not (ROOT / program_path).exists() and not (ROOT / plan_path).exists()
    plan = dict(complete=True, id='AdaRES-Y52', token=TOKEN,
        preregistered_at=datetime.now().astimezone().isoformat(), backward_restore=ref(backward_path),
        bound_setup=ref(bound_program), body=ref(body_path),
        fixed_test='SAME frozen eight TRAIN scenes and32 expressions, four actual production batchsize2 pairs. Exact same parent/candidate initial state, queries128/singleargmax. Native predict2warmups/3measurements EACH model/pair; preserve repeat mask/IoU differences. Input-largest pair chosen only by point count for native training stress2warmups/3measures, finite backward and reserved<24GiB.',
        prediction='No added parameters; inference implementation is unchanged and should preserve paired outputs within observed repeat noise. Actual resource feasibility is measured, not accuracy/Pareto superiority.',
        limits='No validation or hyperparameter/sample selection, weight persistence, optimizer training, Git commit or admission. Largest means only among the frozen inputs, not global worst case.', result='pending')
    (ROOT / program_path).write_text(source, encoding='utf-8', newline='\n')
    plan['program'] = ref(program_path)
    tick.save(ROOT / plan_path, plan)
    for path in (backward_path, body_path, 'scripts/prepare_y52_native_resources_89ff.py', program_path, plan_path):
        if path not in {e['path'] for e in candidate['evidence']}:
            candidate['evidence'].append(ref(path))
    for row in candidate['diagnostic_revision_history']:
        if row['scope'] in ('checkpoint_loader_not_model', 'output_scalar_not_model'):
            row['result'] = 'Compatibility and output repairs completed in full native paired backward and exact nonempty-state restore; model files unchanged.'
            row.setdefault('result_evidence', []).append(ref(backward_path))
    candidate['next_action'] = 'Run preregistered actual paired parent/candidate predict and frozen-largest-pair native resource stress. Preserve released-init zero score activity and positive fixed-trained-parent gradient signal separately; no trained Y52 efficacy or automatic commit/admission.'
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(program=ref(program_path), plan=ref(plan_path), actual_backward_restore_bound=True), indent=2))
finally:
    lock.close()
