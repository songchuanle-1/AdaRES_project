"""Admit real retained Y94 native evidence on the original server-1 GPU0."""
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
from run_y94_bound_native_y93 import validate_terminal, runtime
from y94_contract import validate_result
from y94_deployment import Y92, Y93, Y94
import closure
import experiment_first
CID = 'AdaRES-Y94'
NAME, GPU, POWER = ('server-1', 0, 350.0)
GRANT = 'pipeline/authorizations/server1_gpu0_20261025.json'
GRANT_SHA = 'bef8248c4136533fc49d612dfe68ab06120c0831660e6dc85e7c7bc0b5694c01'
PAUSE = 'pipeline/authorizations/server2_gpu1_pause_20261104.json'
PAUSE_SHA = '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
PARENT_FREEZE_SHA = 'b866bf8a784b9b44ff63cb1dd7de74fd6288b0b84a1f2280a45c9eaa85a76f95'

def prepared(work):
    candidate = next((candidate for candidate in work['candidates'] if candidate['id'] == CID))
    assert candidate['code_commit'] == Y94
    assert (candidate['required_server'], candidate['required_gpu']) == (NAME, GPU)
    assert candidate['native_smoke_plan'] == candidate['native_smoke_plan_server1']
    binding = tick.load(closure.artifact(ROOT, candidate['additive_version_binding']))
    assert binding['complete'] is True and binding['code_commit'] == Y94
    assert binding['parent_commit'] == Y93
    for dependency in binding['evidence']:
        closure.artifact(ROOT, dependency)
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    runtime(NAME, candidate, frozen, monitor.load_servers(str(ROOT))[NAME])
    assert frozen['resource_authorization'] == ref(GRANT)
    assert ref(GRANT)['sha256'] == GRANT_SHA and ref(PAUSE)['sha256'] == PAUSE_SHA
    return (candidate, plan, program, frozen)

def make_freeze(parent, parentref):
    assert parent == tick.load(closure.artifact(ROOT, parentref))
    assert parent['id'] == 'AdaRES-Y92' and parent['code_commit'] == Y92
    assert parentref['sha256'] == PARENT_FREEZE_SHA
    assert parent['goal'] == ref('pipeline/goal.json')
    for key, value in dict(server=NAME, gpu=GPU, power_limit_watts=POWER).items():
        assert parent['resource_condition'][key] == value
    assert parent['evaluation']['methods'][0] == 'AdaRES-Y92'
    freeze = copy.deepcopy(parent)
    freeze.update(id=CID, frozen_at=datetime.now().astimezone().isoformat(), goal=ref('pipeline/goal.json'), code_commit=Y94, parent_protocol=parentref, resource_condition=dict(server=NAME, gpu=GPU, power_limit_watts=POWER, note='Existing power preserved; full paired evaluation/cost on this device; two batches are not a full-dataset memory bound.'), resource_authorization=ref(GRANT), scientific_efficacy='Unknown until original full frozen evaluation.')
    freeze['evaluation']['methods'][0] = CID
    restored = copy.deepcopy(freeze['evaluation'])
    restored['methods'][0] = 'AdaRES-Y92'
    assert restored == parent['evaluation'] and freeze['training'] == parent['training']
    return freeze

def verify(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, _, frozen = prepared(work)
    assert candidate['stage'] == 'validation' and candidate.get('native_smoke_result')
    path = closure.artifact(ROOT, candidate['native_smoke_result'])
    raw = tick.load(path)
    validate_result(raw, candidate, frozen)
    print(json.dumps(dict(result=ref(path), losses=raw['losses'], peak_reserved_bytes=raw['peak_reserved_bytes'], actual_native_verified=True)))

def admit(token):
    server = monitor.load_servers(str(ROOT))[NAME]
    closure.validate_adares_server(ROOT, server)
    servers = monitor.load_servers(str(ROOT))
    assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
    assert servers['server-6']['allowed_gpus'] == []
    assert ref(PAUSE)['sha256'] == PAUSE_SHA
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, program, frozen = prepared(work)
    assert candidate['stage'] == 'validation' and (not candidate.get('admission_receipt'))
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == Y94
    rawpath = closure.artifact(ROOT, candidate['native_smoke_result'])
    raw = tick.load(rawpath)
    validate_result(raw, candidate, frozen)
    done = monitor.load_experiment(str(ROOT), 'AdaRES-Y92')
    assert done['status'] == 'done_complete' and str(done['gpu']) == '0'
    validate_terminal(tick.load(closure.artifact(ROOT, done['terminal_evidence'])), NAME)
    queue = monitor.load_queue(str(ROOT))
    unreserved(queue, monitor.load_running(str(ROOT)), NAME, GPU)
    assert monitor.load_experiment(str(ROOT), CID) is None
    current = live(server, GPU)
    idle(current, POWER)
    cfg = yaml.safe_load((ROOT / 'AdaRES' / candidate['config']).read_bytes())
    assert cfg['train']['seed'] == 1999 and cfg['train']['epochs'] == 30
    assert cfg['dataloader']['train']['batch_size'] == 2 and cfg['model']['sampling_module']['num_proposal'] == 128
    tests = ROOT / ('pipeline/checks/adares-y94_admission_guards_' + token + '.log')
    assert not tests.exists()
    response = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts', '-p', 'test_y94_admission.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    tests.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0 and b'skipped' not in tests.read_bytes()
    parentref = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y92'))['post_training_freeze']
    parent = tick.load(closure.artifact(ROOT, parentref))
    freeze = make_freeze(parent, parentref)
    prefix = 'pipeline/checks/' + CID + '_' + token
    smoke_path = ROOT / (prefix + '_native_smoke.json')
    freeze_path = ROOT / (prefix + '_post_training_freeze.json')
    receipt_path = ROOT / ('pipeline/admissions/' + CID + '.json')
    assert not any((path.exists() for path in (smoke_path, freeze_path, receipt_path)))
    command = 'python -B scripts/run_y94_bound_native_y93.py --server server-1 --token ' + token
    smoke = copy.deepcopy(raw)
    smoke.update(command=command, server=NAME, physical_gpu=GPU, admission_resource_check=current, evidence=[ref(rawpath), ref(rawpath.with_suffix('.log')), ref(program), candidate['native_smoke_plan']])
    tick.save(smoke_path, smoke)
    tick.save(freeze_path, freeze)
    receipt = dict(profile='experiment_first', id=CID, candidate_id=CID, config=candidate['config'], code_commit=Y94, goal_sha256=raw['goal_sha256'], authorization=experiment_first.effective(ROOT)['authorization'], native_smoke=ref(smoke_path), post_training_plan=ref(freeze_path), resource_authorization=ref(GRANT), admitted_at=datetime.now().astimezone().isoformat(), gates={gate: dict(passed=True, evidence=[ref(smoke_path), ref(freeze_path), ref(program)]) for gate in experiment_first.GATES})
    experiment_first.validate_smoke(ROOT, receipt)
    decision = monitor.log_decision(str(ROOT), 'next_batch', CID + ' native TRAIN admission', 'Original native YAML/TRAIN two finite backward and optimizer steps passed; admit independent visual-side quadratic static mask readout.', 'Full original frozen evaluation, exact data, seed1999/fixed30 and350W preserved. Efficacy unknown; GPU1/Y57 remain paused.', ref(smoke_path)['path'])
    receipt['decision'] = ref(decision)
    tick.save(receipt_path, receipt)
    task = dict(id=CID, label=CID + ' fixed30', config=candidate['config'], status='waiting', gpu=GPU, server=NAME, depends_on=[], code_commit=Y94, admission_receipt=ref(receipt_path)['path'], scientific_candidate_id=CID, patience=30, min_epochs=30, fixed_epochs=30)
    closure.admission(ROOT, task)
    candidate.update(stage='admission', admission_receipt=task['admission_receipt'], native_smoke=ref(smoke_path), post_training_freeze=ref(freeze_path), next_action='Canonical gated dispatch; verify increasing native steps. Preserve original Y93 and both user pauses.')
    candidate['evidence'].extend([ref(smoke_path), ref(freeze_path), ref(receipt_path), ref(decision), ref(tests), ref(__file__), ref(PAUSE), done['terminal_evidence']])
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'], candidate_id=CID, kind='native_smoke', result='passed', command=command, observation='Actual original native two-step paths/backward/updates passed; queued with full original evaluation.', evidence=[ref(smoke_path), ref(freeze_path)])
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
    active = {str(row['gpu']) for row in tasks if row['status'] in ('running', 'claiming')}
    events = []
    monitor._process_queue(queue, NAME, server['host'], server['user'], str(ROOT), active, {'0'}, events, max_tasks_per_gpu=1, allow_external_overlay=False, remote_code_root=server['remote_code_root'], current_tasks=len(active), max_tasks_total=1)
    queue = monitor.load_queue(str(ROOT))
    task = next((row for row in queue['gpu_queues'][NAME] if row['id'] == CID))
    assert {name: [row for row in rows if row['id'] != CID] for name, rows in queue['gpu_queues'].items()} == others
    out = ROOT / ('pipeline/checks/adares-y94_dispatch_' + token + '.json')
    assert not out.exists()
    tick.save(out, dict(complete=True, token=token, id=CID, before=current, task=task, events=events, other_queue_entries_preserved=True, helper=ref(__file__)))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next((row for row in work['candidates'] if row['id'] == CID))
    confirmed = task['status'] == 'running' and task.get('launch_verified') is True
    candidate.update(stage='training' if confirmed else 'deployment', next_action='Observe exact native counter increase; reconcile any uncertainty without restart.')
    candidate['evidence'].append(ref(out))
    if confirmed:
        original = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y92'))
        assert original['scientific_status'] == 'fixed30_complete_frozen_review_pending'
        assert original['stage'] == 'validation'
        original.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot', blocker='Original fixed30 checkpoint verified and archived; original full frozen server-1 GPU0/350W review waits while independently admitted Y94 owns the card.', retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(), unblock_condition='Fresh exclusive authorized idle server-1 GPU0 with no compute PID or reservation.', next_action='Preserve original complete fixed30 products, receipt and full frozen server-1 GPU0/350W review. No restart, migration, substitution or promotion from training metrics.')
        original['evidence'].append(ref(out))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(out), status=task['status'], pid=task.get('pid'), launch_verified=task.get('launch_verified'), events=events)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('verify', 'admit', 'dispatch'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        dict(verify=verify, admit=admit, dispatch=dispatch)[args.operation](args.token)
    finally:
        writer.close()
