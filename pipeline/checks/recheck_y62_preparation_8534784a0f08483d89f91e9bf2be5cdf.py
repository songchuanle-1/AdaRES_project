"""Bind the existing Y62 plan to this tick's actual occupied GPU0 check."""

import ast
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure

TOKEN = '8534784a0f08483d89f91e9bf2be5cdf'
lease(TOKEN, 90)
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    before = [ref('pipeline/queue.json'), ref('pipeline/running.json')]
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y62')
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
    assert candidate['blocked_from_stage'] == 'validation'
    assert candidate['profile'] == 'experiment_first' and not candidate.get('admission_receipt')
    plan_path = closure.artifact(ROOT, candidate['native_smoke_plan'])
    plan = tick.load(plan_path)
    assert plan['id'] == candidate['id'] and plan['native_executed'] is False
    program_path = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program_path.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    resource_path = ROOT / ('pipeline/checks/gpu0_lifecycle_v3_' + TOKEN + '.json')
    resource = tick.load(resource_path)
    observed = datetime.fromisoformat(resource['observed_at'])
    now = datetime.now().astimezone()
    assert 0 <= (now - observed).total_seconds() < 300
    assert resource['complete'] and resource['command'].endswith('--token ' + TOKEN)
    assert resource['live']['hostname'] == 'dl01-MS-7C60'
    assert resource['live']['gpu_query_rc'] == resource['live']['compute_query_rc'] == 0
    assert resource['idle_authorized_gpus'] == []
    assert float(resource['hardware']['gpu0'].split(',')[0]) == 280
    active = [t for t in monitor.load_queue(str(ROOT))['gpu_queues']['server-2']
              if t['status'] in ('running', 'claiming')]
    assert len(active) == 1 and active[0]['id'] == 'AdaRES-Y61' and int(active[0]['gpu']) == 0
    process = resource['live']['tasks']['AdaRES-Y61']
    assert process['process_live'] and process['pid'] == active[0]['pid']
    assert process['cwd'] == server['remote_code_root'] and active[0]['config'] in process['cmdline']
    assert str(active[0]['pid']) in resource['live']['compute']
    output = ROOT / ('pipeline/checks/y62_preparation_recheck_' + TOKEN + '.json')
    assert not output.exists()
    tick.save(output, dict(complete=True, id=candidate['id'], token=TOKEN,
        observed_at=now.isoformat(), resource_observed_at=observed.isoformat(),
        exact_tested_files=7, code_commit=candidate['code_commit'],
        current_goal_source_commit_match=True, native_smoke_plan=ref(plan_path),
        frozen_program=ref(program_path), resource_check=ref(resource_path),
        idle_authorized_gpus=[], current_occupant='AdaRES-Y61', gpu0_power_limit_watts=280,
        native_smoke_executed=False, admitted=False, production_sync=False,
        queue_changed=False, running_changed=False, remote_mutations=False))
    additions = [ref(output), ref(__file__), ref(resource_path)]
    candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
    candidate['occupation_evidence'] = ref(output)
    candidate['retry_after'] = (observed + timedelta(hours=1)).isoformat()
    candidate['resource_work_kind'] = 'native_smoke'
    work['tick_attempt'] = dict(
        tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'], kind='resource_check', result='blocked',
        command='python -B pipeline/checks/recheck_y62_preparation_' + TOKEN + '.py',
        observation='Exact seven tested/committed Y62 files, frozen plan and current goal verified; '
            'the current tick observed exclusive GPU0 occupied by live Y61 at 280W. '
            'Native smoke/admission remain pending, with no production or running-job changes.',
        evidence=additions)
    assert before == [ref('pipeline/queue.json'), ref('pipeline/running.json')]
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(output), candidate_id=candidate['id'],
        tested_files_verified=7, result='blocked', native_smoke_executed=False)))
finally:
    lock.close()
