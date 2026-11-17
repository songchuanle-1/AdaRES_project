"""Freeze the actual Y72 CPU-tested bytes while preserving the Y71 handoff HEAD."""
import argparse
import ast
import base64
from datetime import datetime
import hashlib
import json
import subprocess
from prepare_y72_visual_mask_refinement import ROOT, CID, files, lease, monitor, ref, tick
from y68_terminal_contract import build_contract
from check_ef_execution import validate_frozen
import closure

HEAD = '87019e7c2f640f99cb82a3e5b396321fc5b54d25'


def save(token):
    lease(token, 120)
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    work_path = ROOT / 'pipeline/research_work.json'
    work = tick.load(work_path)
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    assert candidate['stage'] == 'implementation' and not candidate.get('code_commit')
    assert not candidate.get('admission_receipt')
    result_path = ROOT / ('pipeline/analyses/y72_cpu_' + token + '.json')
    cpu = tick.load(result_path)
    for key in ('complete', 'implementation_test_passed', 'full_model_CPU_constructed',
                'configured_TRAIN_data_collated', 'released_backbone_loaded',
                'native_YAML_semantically_unchanged', 'train_eval_train_roundtrip',
                'decoder_initial_RNG_and_parent_weights_exact', 'decoder_all_stages_anchor_exact',
                'decoder_input_gradients_anchor_close', 'decoder_current_visual_identity_verified',
                'decoder_backward_finite', 'decoder_refinement_gradient_finite_nonzero',
                'nonzero_refinement_preserves_initial_changes_final_masks',
                'bypassed_refinement_negative_control_exact'):
        assert cpu[key] is True, key
    assert cpu['id'] == CID and cpu['parameters'] == 164695173
    assert cpu['additional_parameters'] == 65536 and cpu['decoder_refinement_calls'] == 6
    assert cpu['decoder_indicator_calls'] == 7
    assert cpu['decoder_prediction_consumption_stages'] == list(range(7))
    assert {row['dtype'] for row in cpu['component_controls']} == {'torch.float32', 'torch.float64'}
    assert all(v is True for row in cpu['component_controls'] for k, v in row.items() if k != 'dtype')
    for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward',
                'native_sparse_forward', 'validation_used', 'admission'):
        assert cpu[key] is False, key
    program = ROOT / ('pipeline/checks/y72_cpu_' + token + '_program.py')
    assert ref(program)['sha256'] == '804df96bbede1531e0b12314979bc99863929d783053e60e27a6a93e045b7013'
    payload = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert payload['goal_sha256'] == cpu['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    for path in files():
        expected = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert path.read_bytes() == base64.b64decode(expected['base64'])
        assert ref(path)['sha256'] == expected['sha256']
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == HEAD
    assert not subprocess.check_output(['git', 'ls-files', '--', *[p.relative_to(ROOT).as_posix() for p in files()]], cwd=ROOT).strip()
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    queue = monitor.load_queue(str(ROOT))
    servers = monitor.load_servers(str(ROOT))
    originals = []
    for server, cid, pid in [('server-2', 'AdaRES-Y68', 20742), ('server-2', 'AdaRES-Y70', 52561),
                             ('server-1', 'AdaRES-Y69', 1135633)]:
        task = next(t for t in queue['gpu_queues'][server] if t['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] is True and task['pid'] == pid
        closure.admission(ROOT, task)
        originals.append(dict(id=cid, server=server, gpu=task['gpu'], pid=pid,
                              code_commit=task['code_commit']))
        if cid == 'AdaRES-Y68':
            build_contract(task, servers[server])
    paused = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    y71 = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y71')
    plan = tick.load(closure.artifact(ROOT, y71['native_smoke_plan']))
    native_program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(native_program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(y71, plan, frozen) == 7
    assert y71['code_commit'] == HEAD and y71['required_gpu'] == 0
    base = 'pipeline/checks/y72_cpu_preparation_' + token
    bundle_path = ROOT / (base + '_source_snapshot.json')
    version_path = ROOT / (base + '_version.json')
    previous_attempt = ROOT / (base + '_previous_attempt.json')
    assert not any(p.exists() for p in (bundle_path, version_path, previous_attempt))
    tick.save(previous_attempt, dict(tick_token=token, tick_attempt=work.get('tick_attempt')))
    tick.save(bundle_path, dict(complete=True, id=CID, tick_token=token,
        observed_at=datetime.now().astimezone().isoformat(), base_head=HEAD,
        code_commit=None, uncommitted=True, CPU_tested_payload=payload,
        exact_files=[ref(p) for p in files()], cpu_test=ref(result_path),
        cpu_log=ref(result_path.with_suffix('.log')), cpu_program=ref(program),
        limitation='Exact source-byte snapshot only; no scoped Git commit, native GPU PASS, admission or trained efficacy.'))
    tick.save(version_path, dict(complete=True, id=CID, tick_token=token,
        CPU_tested_source_snapshot=ref(bundle_path), base_head=HEAD, code_commit=None,
        uncommitted=True, head_preserved_for_Y71=True, Y71_bound_plan=ref(closure.artifact(ROOT, y71['native_smoke_plan'])),
        Y71_frozen_seven_files_verified=True, Y68_terminal_local_source_contract_verified=True,
        queue_before=queue_before, running_before=running_before, original_runs=originals,
        original_admissions_valid=True, Y57_paused=True, native_executed=False,
        admitted=False, pushed=False, production_sync=False, goal=ref('pipeline/goal.json'),
        evidence=[ref(result_path), ref(result_path.with_suffix('.log')), ref(program),
                  ref('scripts/probe_y72_cpu.py'), ref(__file__)]))
    additions = [ref(bundle_path), ref(version_path), ref(result_path), ref(result_path.with_suffix('.log')),
                 ref(program), ref(__file__), ref(previous_attempt)]
    candidate.update(source_manifest=ref(bundle_path), CPU_preparation=ref(version_path),
        implementation_test_passed=True, code_commit=None,
        version_status='CPU-tested immutable byte snapshot; scoped Git commit pending, HEAD preserved for Y71.',
        next_action='Do not replay CPU preparation. Recheck immutable Y72 source snapshot and goal; finish nearer Y71 '
                    'handoff at exact HEAD87019 first. Before advancing HEAD for a scoped seven-file Y72 commit, '
                    'preserve/rebind any still-live hash-bound observer/deployment contracts without changing old evidence. '
                    'Prepare Y72 native YAML/TRAIN/tools.train two-step program bound to the tested commit and exact authorized '
                    'runtime/data/path/power mapping; actual finite loss/backward/optimizer and six intended consumers on a fresh '
                    'exclusive idle/unreserved authorized GPU are mandatory. Only a real native PASS permits frozen original '
                    'full evaluation/cost protocol, receipt/admission and same-tick gated dispatch. Preserve Y68/Y69/Y70, '
                    'Y57 pause and original power. No additional human permission is needed for these scoped operations.')
    candidate['evidence'].extend(additions)
    work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id=CID,
        kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y72_cpu pipeline/checks/y72_cpu_' + token +
                '_program.py --token ' + token + ' --seconds 480',
        observation='Native YAML/full CPU model/released backbone/actual TRAIN collate162738points/8expressions passed; '
                    'float32/64 residual controls and all seven decoder anchors, six current-visual paths/consumers, finite new '
                    'gradient and stage0/bypass controls passed. Exact tested bytes frozen without moving Y71 HEAD87019. '
                    'CUDA/native sparse task loss/backward/optimizer, GPU plans, scoped commit and admission remain unexecuted; efficacy unknown.',
        evidence=additions)
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == HEAD
    tick.save(work_path, work)
    print(json.dumps(dict(version=ref(version_path), source_snapshot=ref(bundle_path),
        CPU_implementation_passed=True, native_GPU_passed=False, admitted=False, HEAD=HEAD)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 150)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        save(args.token)
    finally:
        lock.close()
