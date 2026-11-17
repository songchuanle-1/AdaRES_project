"""Retain the exact tested Y73 bytes on a scoped branch while keeping Y71 HEAD."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from prepare_y73_initial_query_anchor import ROOT, CID, files, lease, monitor, ref, tick
from save_y72_native_plans import git
from check_ef_execution import validate_frozen
from y68_terminal_contract import build_contract
import closure

HEAD = '87019e7c2f640f99cb82a3e5b396321fc5b54d25'
PARENT = 'c60c903941d6cd7ffcc343d7c13610f2ba416caa'
BRANCH = 'refs/heads/adares-y73-initial-query-anchor'


def save(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    assert candidate['stage'] == 'implementation' and not candidate.get('code_commit')
    result = ROOT / ('pipeline/analyses/y73_cpu_' + token + '.json')
    cpu = tick.load(result)
    for key in ('complete', 'implementation_test_passed', 'full_model_CPU_constructed',
                'configured_TRAIN_data_collated', 'released_backbone_loaded', 'native_YAML_semantically_unchanged',
                'train_eval_train_roundtrip', 'decoder_initial_RNG_and_parent_weights_exact',
                'decoder_all_stages_anchor_exact', 'decoder_input_gradients_anchor_close',
                'decoder_current_qqa_ffn_identity_verified', 'same_undetached_initial_query_identity_verified',
                'original_static_masks_and_padding_preserved', 'decoder_backward_finite',
                'decoder_residual_gradient_finite_nonzero', 'nonzero_residual_preserves_initial_changes_final_masks',
                'bypassed_residual_negative_control_exact'):
        assert cpu[key] is True, key
    assert cpu['id'] == CID and cpu['parameters'] == 164695173 and cpu['additional_parameters'] == 65536
    assert cpu['decoder_residual_calls'] == 6 and cpu['decoder_indicator_calls'] == 7
    assert cpu['decoder_prediction_consumption_stages'] == list(range(7))
    assert {r['dtype'] for r in cpu['component_controls']} == {'torch.float32', 'torch.float64'}
    assert all(v is True for r in cpu['component_controls'] for k, v in r.items() if k != 'dtype')
    for key in ('CUDA_initialized', 'native_smoke_passed', 'native_task_backward', 'native_sparse_forward', 'validation_used', 'admission'):
        assert cpu[key] is False
    program = ROOT / ('pipeline/checks/y73_cpu_' + token + '_program.py')
    assert ref(program)['sha256'] == 'c953bc51b16c906941a8281b00013d36c90b6898655227442b3bb96c74730665'
    payload = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert payload['goal_sha256'] == cpu['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    for path in files():
        value = payload['variant']['config_data'] if path.suffix == '.yaml' else payload['variant']['python_files'][path.name]
        assert ref(path)['sha256'] == value['sha256']
    paths = [p.relative_to(ROOT).as_posix() for p in files()]
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    assert git('rev-parse', 'refs/heads/adares-y72-visual-mask-refinement').decode().strip() == PARENT
    assert git('rev-parse', PARENT + '^').decode().strip() == HEAD
    assert not git('ls-files', '--', *paths).strip()
    assert subprocess.run(['git', 'show-ref', '--verify', '--quiet', BRANCH], cwd=ROOT).returncode == 1
    y71 = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y71')
    y72 = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y72')
    for c in (y71, y72):
        plan = tick.load(closure.artifact(ROOT, c['native_smoke_plan']))
        frozen = ast.literal_eval(ast.parse(closure.artifact(ROOT, plan['program']).read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(c, plan, frozen) == 7
    # The retained Y72 parent adds only its seven already-tested files; all common W2/native sources are unchanged.
    parent_delta = [row.split('\t') for row in git('diff', '--no-renames', '--name-status', HEAD, PARENT).decode().splitlines()]
    assert len(parent_delta) == 7 and all(status == 'A' and 'adares_y72_visual_mask_refinement' in path for status, path in parent_delta)
    for status, path in parent_delta:
        assert (ROOT / path).read_bytes() == git('show', PARENT + ':' + path)
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    queue = monitor.load_queue(str(ROOT)); servers = monitor.load_servers(str(ROOT))
    for server, cid, pid in [('server-2', 'AdaRES-Y68', 20742), ('server-2', 'AdaRES-Y70', 52561), ('server-1', 'AdaRES-Y69', 1135633)]:
        task = next(t for t in queue['gpu_queues'][server] if t['id'] == cid)
        assert task['status'] == 'running' and task['launch_verified'] is True and task['pid'] == pid
        closure.admission(ROOT, task)
        if cid == 'AdaRES-Y68': build_contract(task, servers[server])
    paused = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    base = 'pipeline/checks/y73_cpu_version_' + token
    outputs = {name: ROOT / (base + suffix) for name, suffix in (
        ('source', '_source_snapshot.json'), ('commit', '_commit.json'), ('version', '_version.json'), ('index', '_isolated.index'))}
    assert not any(p.exists() for p in outputs.values())
    index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not index.is_absolute(): index = ROOT / index
    index_before = hashlib.sha256(index.read_bytes()).hexdigest()
    tick.save(outputs['source'], dict(complete=True, id=CID, tick_token=token, CPU_tested_payload=payload,
        exact_files=[ref(p) for p in files()], cpu_test=ref(result), cpu_log=ref(result.with_suffix('.log')),
        cpu_program=ref(program), working_HEAD=HEAD, intended_parent_commit=PARENT, prior_tick_attempt=work.get('tick_attempt')))
    env = dict(os.environ, GIT_INDEX_FILE=str(outputs['index']))
    git('read-tree', PARENT, env=env)
    for path in files():
        oid = git('hash-object', '-w', '--stdin', data=path.read_bytes()).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644,' + oid + ',' + path.relative_to(ROOT).as_posix(), env=env)
    tree = git('write-tree', env=env).decode().strip()
    delta = [row.split('\t') for row in git('diff-tree', '-r', '--no-commit-id', '--name-status', PARENT, tree).decode().splitlines()]
    assert sorted(delta) == sorted([['A', path] for path in paths])
    lease(token, 120)
    commit = git('commit-tree', tree, '-p', PARENT, '-m', 'Add Y73 direct initial-query anchor residual').decode().strip()
    tick.save(outputs['commit'], dict(complete=True, id=CID, code_commit=commit, parent_commit=PARENT, exact_delta=delta,
        intended_branch=BRANCH, CPU_source_snapshot=ref(outputs['source']), working_HEAD_unchanged=HEAD))
    git('update-ref', BRANCH, commit, '0' * 40)
    for path in files():
        assert path.read_bytes() == git('show', commit + ':' + path.relative_to(ROOT).as_posix())
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    assert hashlib.sha256(index.read_bytes()).hexdigest() == index_before
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    tick.save(outputs['version'], dict(complete=True, id=CID, observed_at=datetime.now().astimezone().isoformat(),
        code_commit=commit, parent_commit=PARENT, retained_branch=BRANCH, working_HEAD=HEAD,
        main_index_unchanged=True, exact_files=[ref(p) for p in files()], source_snapshot=ref(outputs['source']),
        commit_record=ref(outputs['commit']), isolated_index=ref(outputs['index']), CPU_result=ref(result),
        Y71_and_Y72_frozen_sources_and_plans_verified=True, Y68_terminal_local_contract_preserved=True,
        original_admissions_valid=True, queue_and_running_preserved=True, Y57_paused=True,
        native_GPU_program_prepared=False, native_executed=False, admitted=False, production_sync=False,
        evidence=[ref(__file__), ref('scripts/prepare_y73_initial_query_anchor.py'), ref('scripts/probe_y73_cpu.py')],
        limitation='CPU implementation and exact branch version only; native sparse TRAIN two-step task loss/backward/optimizer and efficacy remain unobserved.'))
    additions = [ref(p) for p in outputs.values()] + [ref(result), ref(result.with_suffix('.log')), ref(program), ref(__file__)]
    candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(outputs['version']),
        implementation_test_passed=True,
        next_action='Do not replay preparation/CPU/version saving. Preserve Y71 working HEAD87019 and prioritize '
                    'its existing guarded handoff, then Y72 exact native plans. Prepare an original native '
                    'YAML/TRAIN/tools.train two-step program bound to this exact retained commit and verified '
                    'private runtime/original data/path/power mapping. Real finite native loss/backward/optimizer '
                    'and six current-qqa_ffn/initial-anchor paths and prediction consumers on a fresh exclusive '
                    'idle/unreserved authorized GPU are mandatory before original full evaluation/cost freeze, '
                    'receipt/admission and same-tick gated dispatch with native progress. Production selection '
                    'must safely preserve/rebind live source/observer/deployment contracts; never loosen old '
                    'evidence or restart protected runs. Y57 remains independently paused; scoped operations are authorized.')
    candidate['evidence'].extend(additions)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y73_cpu pipeline/checks/y73_cpu_' + token +
                '_program.py --token ' + token + ' --seconds 480',
        observation='Full CPU native-YAML model/released backbone/real TRAIN collate passed. Independent float32/64 '
                    'anchor-residual controls, seven-stage W2 output/RNG/input-gradient anchors, six original current '
                    'qqa_ffn and same undetached initial-query identities/actual prediction consumers, finite new '
                    'gradient and unchanged-stage0/bypass controls passed. Exact seven tested files saved on a '
                    'retained branch descending from Y72; main HEAD87019/index and original runs unchanged. '
                    'No native GPU program/task loss/backward/optimizer, admission, sync or efficacy result.', evidence=additions)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(version=ref(outputs['version']), code_commit=commit, retained_branch=BRANCH,
        working_HEAD=HEAD, main_index_unchanged=True, native_GPU_prepared=False, admitted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True); args = parser.parse_args()
    lease(args.token, 240)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try:
        save(args.token)
    finally:
        lock.close()
