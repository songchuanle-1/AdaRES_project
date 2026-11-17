"""Preserve two exact prepared bundles after the scoped Y63 sibling commit."""

import ast
import copy
from datetime import datetime
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure

TOKEN = '10ecd3bb7a2b4b78a7382ea479609ebc'
lease(TOKEN, 120)
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue_before = [ref('pipeline/queue.json'), ref('pipeline/running.json')]
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidates = {c['id']: c for c in work['candidates'] if c['id'] in ('AdaRES-Y62', 'AdaRES-Y63')}
    assert set(candidates) == {'AdaRES-Y62', 'AdaRES-Y63'}
    assert all(c['stage'] == 'blocked' and c['blocker_kind'] == 'exclusive_gpu_slot'
               and c['blocked_from_stage'] == 'validation' and not c.get('admission_receipt')
               for c in candidates.values())
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()
    head = git('rev-parse', 'HEAD')
    y62, y63 = candidates['AdaRES-Y62'], candidates['AdaRES-Y63']
    assert head == y63['code_commit'] == '82fbd9fb1fc56da6afee5b145f59cfa533464707'
    assert git('rev-parse', head + '^') == y62['code_commit']
    changed = set(git('diff-tree', '--no-commit-id', '--name-only', '-r', head).splitlines())
    expected = {'AdaRES/ipdn/model_paper5/adares_y63_peer_bilinear/' + n
                for n in ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')}
    expected.add('AdaRES/configs_song/model_paper5/adares_y63_peer_bilinear_res.yaml')
    assert changed == expected
    original_plan_ref = copy.deepcopy(y62['native_smoke_plan'])
    old_plan = tick.load(closure.artifact(ROOT, original_plan_ref))
    old_program = closure.artifact(ROOT, old_plan['program'])
    old_text = old_program.read_text(encoding='utf-8')
    old_frozen = ast.literal_eval(ast.parse(old_text).body[0].value)
    assert validate_frozen(y62, old_plan, old_frozen) == 7
    assert old_plan['native_executed'] is False
    frozen = copy.deepcopy(old_frozen)
    frozen['code_commit'] = head
    plan = copy.deepcopy(old_plan)
    plan.update(code_commit=head, origin_plan=original_plan_ref,
        rebind_reason='Only seven independent Y63 files were added; exact Y62 tested bytes and goal are unchanged.',
        original_tested_commit=y62['code_commit'])
    native_path = ROOT / ('pipeline/checks/y62_smoke_rebound_' + TOKEN + '_program.py')
    plan_path = ROOT / ('pipeline/checks/y62_smoke_rebound_' + TOKEN + '_plan.json')
    assert not native_path.exists() and not plan_path.exists()
    assert validate_frozen(dict(y62, code_commit=head), plan, frozen) == 7
    native_path.write_text('FROZEN=' + repr(frozen) + '\n' + old_text.split('\n', 1)[1],
        encoding='utf-8', newline='\n')
    assert native_path.read_text(encoding='utf-8').split('\n', 1)[1] == old_text.split('\n', 1)[1]
    plan['program'] = ref(native_path)
    plan['planned_command'] = ('python -B scripts/run_experiment_first_probe.py y62_smoke ' +
        native_path.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id 0')
    tick.save(plan_path, plan)
    y63_plan = tick.load(closure.artifact(ROOT, y63['native_smoke_plan']))
    y63_program = closure.artifact(ROOT, y63_plan['program'])
    y63_frozen = ast.literal_eval(ast.parse(y63_program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(y63, y63_plan, y63_frozen) == 7
    assert y63_program.read_text(encoding='utf-8').split('\n', 1)[1] == (
        ROOT / 'scripts/probe_y63_native_smoke.py').read_text(encoding='utf-8')
    negative_controls = {}
    for name in ('candidate_id', 'old_commit'):
        altered = copy.deepcopy(y63_frozen)
        if name == 'candidate_id':
            altered['variant']['id'] = 'AdaRES-Y62'
        else:
            altered['code_commit'] = y62['code_commit']
        refused = False
        try:
            validate_frozen(y63, y63_plan, altered)
        except AssertionError:
            refused = True
        assert refused
        negative_controls[name + '_refused'] = True
    output = ROOT / ('pipeline/checks/prepared_sibling_alignment_' + TOKEN + '.json')
    assert not output.exists()
    tick.save(output, dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        code_commit=head, original_y62_commit=y62['code_commit'], exact_changed_files=sorted(changed),
        Y62_tested_files_unchanged=7, Y63_tested_files_verified=7,
        current_goal_matches=True, y62_original_plan=original_plan_ref,
        y62_rebound_plan=ref(plan_path), y63_native_plan=y63['native_smoke_plan'],
        negative_controls=negative_controls, native_smoke_executed=False,
        queue_changed=False, running_changed=False, production_sync=False,
        authorization_changed=False, model_revision=False))
    y62['code_commit'] = head
    y62['native_smoke_plan'] = ref(plan_path)
    y62['evidence'].extend([ref(native_path), ref(plan_path), ref(output), ref(__file__)])
    y63['evidence'].extend([ref(output), ref(__file__), ref('scripts/test_y63_preparation.py')])
    assert queue_before == [ref('pipeline/queue.json'), ref('pipeline/running.json')]
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print({'evidence': ref(output), 'exact_Y62_files_preserved': 7,
           'exact_Y63_files_verified': 7, 'code_commit': head, 'native_smoke_executed': False})
finally:
    lock.close()
