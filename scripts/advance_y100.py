"""Admit actual Y99 server2 evidence with the complete original Y94 protocol."""
import argparse
import ast
import copy
from datetime import datetime, timedelta
import json
import subprocess
import sys
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from advance_y82 import live, idle, unreserved
from run_y100_native_v2 import validate_binding, validate_terminal, runtime
from y100_admission_contract import validate_result
from y100_deployment import Y98, Y99, Y100, require_running_parent_handoff
import closure
import experiment_first
CID, NAME, GPU, POWER = ('AdaRES-Y100', 'server-2', 0, 280.0)
GRANT = 'pipeline/authorizations/server2_gpu1_pause_20261104.json'
GRANT_SHA = '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
PAUSE = 'pipeline/authorizations/server2_gpu1_pause_20261104.json'
PAUSE_SHA = '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
PARENT_FREEZE_SHA = 'bcd19eedca805c8491568b3f81cf9cefa8dc4b0def8f6a09dd87c58f5e29d51c'

def prepared(work):
    candidate = next((row for row in work['candidates'] if row['id'] == CID))
    assert candidate['code_commit'] == Y100
    assert (candidate['required_server'], candidate['required_gpu']) == (NAME, GPU)
    assert candidate['native_smoke_plan'] == candidate['native_smoke_plan_server2_gpu0']
    binding = tick.load(closure.artifact(ROOT, candidate['additive_version_binding']))
    validate_binding(binding, candidate, Y99)
    for dependency in binding['evidence']:
        closure.artifact(ROOT, dependency)
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    assert plan['execution_helper'] == ref('scripts/run_y100_native_v2.py')
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    runtime(NAME, candidate, frozen, monitor.load_servers(str(ROOT))[NAME])
    assert frozen['resource_authorization'] == ref(GRANT)
    assert ref(GRANT)['sha256'] == GRANT_SHA and ref(PAUSE)['sha256'] == PAUSE_SHA
    return (candidate, plan, program, frozen)

def make_freeze(parent, parentref):
    assert parentref['sha256'] == PARENT_FREEZE_SHA
    assert parent == tick.load(closure.artifact(ROOT, parentref))
    assert parent['complete'] is True and parent['id'] == 'AdaRES-Y98' and (parent['code_commit'] == Y98)
    assert parent['goal'] == ref('pipeline/goal.json')
    assert parent['resource_authorization'] == ref(GRANT)
    assert parent['evaluation']['methods'][0] == 'AdaRES-Y98'
    for key, value in dict(server=NAME, gpu=GPU, power_limit_watts=POWER).items():
        assert parent['resource_condition'][key] == value
    freeze = copy.deepcopy(parent)
    freeze.update(id=CID, frozen_at=datetime.now().astimezone().isoformat(), code_commit=Y100, parent_protocol=parentref, scientific_efficacy='Unknown until original full frozen evaluation.')
    freeze['evaluation']['methods'][0] = CID
    restored = copy.deepcopy(freeze['evaluation'])
    restored['methods'][0] = parent['id']
    assert restored == parent['evaluation'] and freeze['training'] == parent['training']
    assert freeze['resource_condition'] == parent['resource_condition']
    return freeze

def require_completed_owner():
    done = monitor.load_experiment(str(ROOT), 'AdaRES-Y98')
    assert done['status'] == 'done_complete' and done['server'] == NAME and (str(done['gpu']) == '0')
    validate_terminal(tick.load(closure.artifact(ROOT, done['terminal_evidence'])), NAME)
    return done

def admit(token):
    server = monitor.load_servers(str(ROOT))[NAME]
    closure.validate_adares_server(ROOT, server)
    servers = monitor.load_servers(str(ROOT))
    assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
    assert servers['server-6']['allowed_gpus'] == []
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, program, frozen = prepared(work)
    assert candidate['stage'] == 'validation' and (not candidate.get('admission_receipt'))
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == Y100
    require_running_parent_handoff(work, candidate)
    rawpath = closure.artifact(ROOT, candidate['native_smoke_result'])
    raw = tick.load(rawpath)
    validate_result(raw, candidate, frozen)
    done = require_completed_owner()
    queue = monitor.load_queue(str(ROOT))
    unreserved(queue, monitor.load_running(str(ROOT)), NAME, GPU)
    assert monitor.load_experiment(str(ROOT), CID) is None
    current = live(server, GPU)
    idle(current, POWER)
    cfg = yaml.safe_load((ROOT / 'AdaRES' / candidate['config']).read_bytes())
    assert cfg['train']['seed'] == 1999 and cfg['train']['epochs'] == 30
    assert cfg['dataloader']['train']['batch_size'] == 2
    assert cfg['model']['sampling_module']['num_proposal'] == 128
    tests = ROOT / ('pipeline/checks/adares-y100_admission_guards_' + token + '.log')
    assert not tests.exists()
    response = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts', '-p', 'test_y100_admission.py', '-v'], cwd=ROOT, capture_output=True, timeout=90)
    tests.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and b'skipped' not in tests.read_bytes(), str(tests)
    parentref = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y98'))['post_training_freeze']
    freeze = make_freeze(tick.load(closure.artifact(ROOT, parentref)), parentref)
    prefix = 'pipeline/checks/' + CID + '_' + token
    smoke_path, freeze_path = (ROOT / (prefix + '_native_smoke.json'), ROOT / (prefix + '_post_training_freeze.json'))
    receipt_path = ROOT / ('pipeline/admissions/' + CID + '.json')
    assert not any((path.exists() for path in (smoke_path, freeze_path, receipt_path)))
    command = 'python -B scripts/run_y100_native_v2.py --server server-2 --token ' + token
    smoke = copy.deepcopy(raw)
    smoke.update(command=command, server=NAME, physical_gpu=GPU, admission_resource_check=current, evidence=[ref(rawpath), ref(rawpath.with_suffix('.log')), ref(program), candidate['native_smoke_plan']])
    tick.save(smoke_path, smoke)
    tick.save(freeze_path, freeze)
    receipt = dict(profile='experiment_first', id=CID, candidate_id=CID, config=candidate['config'], code_commit=Y100, goal_sha256=raw['goal_sha256'], authorization=experiment_first.effective(ROOT)['authorization'], native_smoke=ref(smoke_path), post_training_plan=ref(freeze_path), resource_authorization=ref(GRANT), admitted_at=datetime.now().astimezone().isoformat(), gates={gate: dict(passed=True, evidence=[ref(smoke_path), ref(freeze_path), ref(program)]) for gate in experiment_first.GATES})
    experiment_first.validate_smoke(ROOT, receipt)
    decision = monitor.log_decision(str(ROOT), 'next_batch', CID + ' native TRAIN admission', 'Original native YAML/TRAIN two finite backward and optimizer steps passed; admit independent full1024 appearance bypass beside the original visual-memory projection MLP.', 'Full original server2 GPU0/280W frozen evaluation/cost, exact data and seed1999/fixed30 preserved. Efficacy unknown. Original Y99 and GPU1/Y57 pauses preserved.', ref(smoke_path)['path'])
    receipt['decision'] = ref(decision)
    tick.save(receipt_path, receipt)
    task = dict(id=CID, label=CID + ' fixed30', config=candidate['config'], status='waiting', gpu=GPU, server=NAME, depends_on=[], code_commit=Y100, admission_receipt=ref(receipt_path)['path'], scientific_candidate_id=CID, patience=30, min_epochs=30, fixed_epochs=30)
    closure.admission(ROOT, task)
    candidate.update(stage='admission', admission_receipt=task['admission_receipt'], native_smoke=ref(smoke_path), post_training_freeze=ref(freeze_path), next_action='Canonical gated dispatch and increasing native counters; preserve original Y99 and both pauses.')
    candidate['evidence'].extend([ref(smoke_path), ref(freeze_path), ref(receipt_path), ref(decision), ref(tests), ref(__file__), ref(PAUSE), done['terminal_evidence']])
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'], candidate_id=CID, kind='native_smoke', result='passed', command=command, observation='Actual native paths/backward/updates passed; queued with the full original server2 protocol.', evidence=[ref(smoke_path), ref(freeze_path)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    queue['gpu_queues'].setdefault(NAME, []).append(task)
    monitor.save_queue(str(ROOT), queue)
    print(json.dumps(dict(admitted=CID, receipt=ref(receipt_path), losses=raw['losses'])))

def dispatch(token):
    server = monitor.load_servers(str(ROOT))[NAME]
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0] and ref(PAUSE)['sha256'] == PAUSE_SHA
    queue = monitor.load_queue(str(ROOT))
    tasks = queue['gpu_queues'][NAME]
    task = next((task for task in tasks if task['id'] == CID))
    assert task['status'] == 'waiting'
    closure.admission(ROOT, task)
    assert not monitor.load_running(str(ROOT)).get(NAME, {}).get('0')
    assert not any((row['id'] != CID and row['status'] in ('waiting', 'claiming', 'running') and (str(row.get('gpu')) == '0') for row in tasks))
    others = {name: copy.deepcopy([row for row in rows if row['id'] != CID]) for name, rows in queue['gpu_queues'].items()}
    current = live(server, GPU)
    idle(current, POWER)
    events = []
    monitor._process_queue(queue, NAME, server['host'], server['user'], str(ROOT), set(), {'0'}, events, max_tasks_per_gpu=1, allow_external_overlay=False, remote_code_root=server['remote_code_root'], current_tasks=0, max_tasks_total=1)
    queue = monitor.load_queue(str(ROOT))
    task = next((row for row in queue['gpu_queues'][NAME] if row['id'] == CID))
    assert {name: [row for row in rows if row['id'] != CID] for name, rows in queue['gpu_queues'].items()} == others
    out = ROOT / ('pipeline/checks/adares-y100_dispatch_' + token + '.json')
    assert not out.exists()
    tick.save(out, dict(complete=True, token=token, id=CID, before=current, task=task, events=events, other_queue_entries_preserved=True, helper=ref(__file__)))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next((row for row in work['candidates'] if row['id'] == CID))
    confirmed = task['status'] == 'running' and task.get('launch_verified') is True
    candidate.update(stage='training' if confirmed else 'deployment', next_action='Observe exact native counter increase; reconcile uncertainty without restart.')
    candidate['evidence'].append(ref(out))
    if confirmed:
        original = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y98'))
        assert original['scientific_status'] == 'fixed30_complete_frozen_review_pending'
        assert original['stage'] == 'validation'
        original.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot', blocker='Original complete fixed30 checkpoint verified and archived; full original server-2 GPU0/280W frozen review waits while independently admitted Y100 owns the card.', retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(), unblock_condition='Fresh exclusive authorized idle server-2 GPU0 without compute or reservation.', next_action='Preserve complete original products, receipt and full original-device frozen review; no restart, migration, substitution or promotion from training metrics.')
        original['evidence'].append(ref(out))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(out), status=task['status'], pid=task.get('pid'), launch_verified=task.get('launch_verified'), events=events)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('admit', 'dispatch'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        dict(admit=admit, dispatch=dispatch)[args.operation](args.token)
    finally:
        writer.close()
