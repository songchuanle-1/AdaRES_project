"""Retain exact tested Y103 additions and freeze guarded, unexecuted GPU0 tasks."""
import argparse, ast, base64, copy, hashlib, json, os
from datetime import datetime, timedelta
from pathlib import Path
import subprocess, sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y103_spatial_input_skip import CID, files
from y103_contract import validate_cpu
from y101_deployment import Y100, Y101, contract
from retain_y91_cpu_native import git
from retain_y100_cpu_native import Rewrite, tests
from prepare_y69_server1_alternative import adapted_body
from y82_server1_alternative import DATA_PROOF
from check_ef_execution import validate_frozen
import closure
HEAD = Y101
BRANCH = 'refs/heads/adares-y103-spatial-input-skip'
OWNERS = {'server-1': 'AdaRES-Y101', 'server-2': 'AdaRES-Y100'}
PAUSE = 'pipeline/authorizations/server2_gpu1_pause_20261104.json'

def retain(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next((r for r in work['candidates'] if r['id'] == CID))
    assert c['stage'] == 'implementation' and (not c.get('code_commit'))
    result = ROOT / ('pipeline/analyses/y103_cpu_' + token + '.json')
    validate_cpu(tick.load(result))
    guards = ROOT / ('pipeline/checks/y103_CPU_guards_' + token + '.log')
    tests('test_y103_preparation.py', guards, 10)
    program = ROOT / ('pipeline/checks/y103_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert payload['goal_sha256'] == tick.load(result)['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    for p in files():
        v = payload['variant']['config_data'] if p.suffix == '.yaml' else payload['variant']['python_files'][p.name]
        assert p.read_bytes() == base64.b64decode(v['base64']) and ref(p)['sha256'] == v['sha256']
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    paths = [p.relative_to(ROOT).as_posix() for p in files()]
    assert not git('ls-files', '--', *paths).strip()
    assert subprocess.run(['git', 'show-ref', '--verify', '--quiet', BRANCH], cwd=ROOT).returncode == 1
    before = {p: ref(p) for p in ('pipeline/queue.json', 'pipeline/running.json', 'pipeline/goal.json', 'SERVERS.yaml')}
    q = monitor.load_queue(str(ROOT))
    owners = []
    for name, cid, pid, commit in (('server-1', 'AdaRES-Y101', 2263581, Y101), ('server-2', 'AdaRES-Y100', 19305, Y100)):
        task = next((r for r in q['gpu_queues'][name] if r['id'] == cid))
        assert (task['status'], task['launch_verified'], task['pid'], task['code_commit']) == ('running', True, pid, commit)
        assert monitor.load_running(str(ROOT))[name]['0'] == cid
        closure.admission(ROOT, task)
        contract(commit, HEAD, cid)
        owners.append({k: task[k] for k in ('id', 'server', 'gpu', 'pid', 'code_commit', 'config', 'log', 'status', 'launch_verified')})
    from advance_y101 import prepared, validate_result
    parent, _, _, f = prepared(work)
    native = parent['native_smoke_result']
    validate_result(tick.load(closure.artifact(ROOT, native)), parent, f)
    base = ROOT / ('pipeline/checks/y103_cpu_version_' + token)
    source = base.with_name(base.name + '_source_snapshot.json')
    index = base.with_suffix('.index')
    out = base.with_suffix('.json')
    assert not any((p.exists() for p in (source, index, out)))
    main = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    main = main if main.is_absolute() else ROOT / main
    index_sha = hashlib.sha256(main.read_bytes()).hexdigest()
    tick.save(source, dict(complete=True, id=CID, token=token, CPU_tested_payload=payload, CPU_result=ref(result), CPU_log=ref(result.with_suffix('.log')), CPU_program=ref(program), exact_files=[ref(p) for p in files()], working_HEAD=HEAD))
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    git('read-tree', HEAD, env=env)
    for p in files():
        blob = git('hash-object', '-w', '--stdin', data=p.read_bytes()).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + p.relative_to(ROOT).as_posix(), env=env)
    tree = git('write-tree', env=env).decode().strip()
    delta = [r.split('\t') for r in git('diff-tree', '-r', '--no-commit-id', '--name-status', HEAD, tree).decode().splitlines()]
    assert sorted(delta) == sorted([['A', p] for p in paths])
    commit = git('commit-tree', tree, '-p', HEAD, '-m', 'Add independent Y103 whole spatial-input visual-memory bypass').decode().strip()
    git('update-ref', BRANCH, commit, '0' * 40)
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD and hashlib.sha256(main.read_bytes()).hexdigest() == index_sha
    assert all((p.read_bytes() == git('show', commit + ':' + p.relative_to(ROOT).as_posix()) for p in files()))
    tick.save(out, dict(complete=True, id=CID, code_commit=commit, parent_commit=HEAD, retained_branch=BRANCH, working_HEAD=HEAD, main_index_unchanged=True, CPU_tested_scientific_files_exact=True, original_Y99_Y100_preserved=True, original_owner_rows=owners, source_snapshot=ref(source), CPU_result=ref(result), CPU_program=ref(program), exact_delta=delta, isolated_index=ref(index), matching_current_owners=OWNERS, actual_parent_native=native, native_executed=False, admitted=False, production_sync=False, helper=ref(__file__), evidence=[ref(source), ref(result), ref(result.with_suffix('.log')), ref(program), ref(index), ref(guards), ref(__file__), native]))
    c.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(out), additive_version_binding=ref(out), implementation_test_passed=True, CPU_result=ref(result), CPU_program=ref(program), CPU_guard_tests=ref(guards))
    c['evidence'].extend([ref(out), ref(source), ref(index), ref(result), ref(result.with_suffix('.log')), ref(guards), ref(__file__)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert all((ref(p) == r for p, r in before.items()))
    print(json.dumps(dict(version=ref(out), code_commit=commit, HEAD_preserved=HEAD, CPU_guards=10)))

def create_runner():
    source = ROOT / 'scripts/run_y101_native.py'
    target = ROOT / 'scripts/run_y103_native.py'
    old_test = ROOT / 'scripts/test_y101_runtime_plan.py'
    new_test = ROOT / 'scripts/test_y103_runtime_plan.py'
    assert not target.exists() and (not new_test.exists())
    changes = {'run_y101_native': 'run_y103_native', 'original_Y99_Y100_preserved': 'original_Y101_Y100_preserved', 'Y101': 'Y103', 'y101': 'y103'}
    tree = Rewrite(changes).visit(ast.parse(source.read_text(encoding='utf-8')))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'PARENT' for t in n.targets)):
            n.value = ast.Constant(value=HEAD)
        if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'OWNERS' for t in n.targets)):
            n.value = ast.parse(repr(OWNERS), mode='eval').body
    text = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    text = text.replace('all six complete original FFNs/current-query bypasses/static masks/shared heads/hard gates/gradients', 'complete original spatial/appearance projections, whole spatial-input skip, sum_norm/visual memory/static7/shared heads/hard gates/gradients')
    ast.parse(text, feature_version=(3, 8))
    target.write_text(text, encoding='utf-8', newline='\n')
    tree = Rewrite(changes).visit(ast.parse(old_test.read_text(encoding='utf-8')))
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == 'test_matching_current_owner_full_completion_required':
            loop = next((x for x in n.body if isinstance(x, ast.For)))
            loop.iter = ast.parse(repr(tuple(OWNERS.items())), mode='eval').body
    text = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    ast.parse(text, feature_version=(3, 8))
    new_test.write_text(text, encoding='utf-8', newline='\n')
    return [ref(source), ref(target), ref(old_test), ref(new_test)]

def plan(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next((r for r in work['candidates'] if r['id'] == CID))
    assert c['stage'] == 'implementation' and c['implementation_test_passed'] and (not c.get('native_preparation'))
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    version = tick.load(closure.artifact(ROOT, c['source_manifest']))
    payload = tick.load(closure.artifact(ROOT, version['source_snapshot']))['CPU_tested_payload']
    validate_cpu(tick.load(closure.artifact(ROOT, c['CPU_result'])))
    base = ROOT / ('pipeline/checks/y103_native_' + token)
    out = base.with_suffix('.json')
    guards = base.with_name(base.name + '_runtime_guards.log')
    assert not out.exists() and (not guards.exists())
    authority = ref(PAUSE)
    assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
    closure.artifact(ROOT, DATA_PROOF)
    servers = monitor.load_servers(str(ROOT))
    assert servers['server-6']['allowed_gpus'] == []
    body = closure.artifact(ROOT, c['prepared_native_body']).read_text(encoding='utf-8')
    before = {p: ref(p) for p in ('pipeline/queue.json', 'pipeline/running.json')}
    refs = create_runner()
    plans = {}
    programs = []
    for name, power in (('server-2', 280.0), ('server-1', 350.0)):
        server = servers[name]
        closure.validate_adares_server(ROOT, server)
        assert server['allowed_gpus'] == [0]
        grant = authority if name == 'server-2' else ref('pipeline/authorizations/server1_gpu0_20261025.json')
        f = dict(variant=payload['variant'], code_commit=c['code_commit'], goal_sha256=payload['goal_sha256'], train_source_lf_sha256=payload['train_source_lf_sha256'], runtime_server=name, runtime_physical_gpu=0, runtime_power_limit_watts=power, resource_authorization=grant)
        program_body = body
        if name == 'server-1':
            f['runtime_path_mappings'] = copy.deepcopy(server['runtime_path_mappings'])
            program_body = adapted_body(body, power)
        suffix = name.replace('-', '')
        program = base.with_name(base.name + '_' + suffix + '_program.py')
        ppath = base.with_name(base.name + '_' + suffix + '_plan.json')
        assert not program.exists() and (not ppath.exists())
        value = dict(complete=True, id=CID, config=c['config'], profile='experiment_first', code_commit=c['code_commit'], goal_sha256=f['goal_sha256'], server=name, gpu=0, runtime_power_limit_watts=power, resource_authorization=grant, working_HEAD_preserved=HEAD, additive_parent=HEAD, retained_branch=BRANCH, native_executed=False, admitted=False, native_PASS_required=True, guard_tests_passed=10, Python38_compile=True, execution_helper=ref('scripts/run_y103_native.py'), planned_command='python -B scripts/run_y103_native.py --server ' + name + ' --token <CURRENT_TOKEN>', required_resource='Matching original ' + OWNERS[name] + ' natural fixed30/full finite exact original checkpoint verification/canonical archive and fresh exclusive idle/unreserved authorized GPU0; exact HEAD/runtime/data/350W280W. No GPU1, power change or substitution.', checks='Actual original YAML/sparse TRAIN two finite backward/optimizer steps; complete original spatial and appearance projections/full32 raw spatial bypass/sum_norm/visual memory/static7/shared heads/hard gates/full gradients/native modes/budget. CPU feasibility is not native PASS.')
        assert validate_frozen(c, value, f) == 7
        program.write_text('FROZEN=' + repr(f) + '\n' + program_body, encoding='utf-8', newline='\n')
        ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
        value['program'] = ref(program)
        tick.save(ppath, value)
        plans[name] = ref(ppath)
        programs.append(ref(program))
    c.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot', required_server='server-1', required_gpu=0, resource_work_kind='native_smoke', native_smoke_plan=plans['server-1'], native_smoke_plan_server1=plans['server-1'], native_smoke_plan_server2_gpu0=plans['server-2'], blocker='Original Y101/server1GPU0 and Y100/server2GPU0 remain reserved; first CPU feasibility passed, actual native sparse TRAIN still required.', retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(), wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'], unblock_condition='At exact HEADY101 after matching original Y101/server1 or Y100/server2 naturally fixed30/full finite exact checkpoint/canonical archive, then fresh exclusive authorized idle GPU0.', next_action='Y102 precedes Y103. Do not replay passed CPU/version/plan tests. At HEADY101 ONLY after matching original Y101/server1 or Y100/server2 naturally fixed30/full finite original checkpoint/canonical archive and fresh authorized exclusive idle GPU0 use run_y103_native.py; changed HEAD/owner requires NEW tested same-byte binding. Actual native PASS must continue through complete original selected-device freeze/admission/source selection/dispatch/increasing counters same tick. GPU1/Y57 pauses remain; efficacy unknown.')
    tick.save(ROOT / 'pipeline/research_work.json', work)
    tests('test_y103_runtime_plan.py', guards, 8)
    tick.save(out, dict(complete=True, id=CID, token=token, code_commit=c['code_commit'], CPU_version=c['source_manifest'], CPU_result=c['CPU_result'], CPU_program=c['CPU_program'], native_plans=plans, guard_tests=c['CPU_guard_tests'], guard_tests_passed=10, runtime_guard_tests=ref(guards), runtime_guard_tests_passed=8, native_observer_body=c['prepared_native_body'], native_observer_callbacks_executed_on_GPU=False, same_callbacks_actually_executed_on_CPU=True, original_data_proof=DATA_PROOF, original_Y99_Y100_preserved=True, native_executed=False, admitted=False, production_sync=False, helper=ref(__file__), evidence=refs + programs + list(plans.values()) + [ref(guards), c['CPU_guard_tests']]))
    c['native_preparation'] = ref(out)
    c['evidence'].extend([ref(out), ref(guards), authority, DATA_PROOF, ref('scripts/y103_contract.py'), ref('scripts/test_y103_preparation.py'), ref('scripts/test_y103_runtime_plan.py')] + list(plans.values()) + programs + refs)
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'], candidate_id=CID, kind='implementation_test', result='passed', command='python -B scripts/run_experiment_first_probe.py y103_cpu ' + c['CPU_program']['path'] + ' --token ' + token + ' --seconds 600', observation='FIRST original CPU YAML/fullmodel/releasedbackbone/TRAIN collation; scalarfloat32/64/no-RNG/permutations/sign/row-locality/seven-stage W2/input-gradient/causal initial-final masks/exact bypass and two finite synthetic updates passed. Same future native callbacks ran on CPU, whole raw spatial input/original projections/fusion/static mask and shared skip gradients verified;10 evidence+8 runtime guards passed. GPU native unexecuted; efficacy unknown.', evidence=[c['CPU_result'], c['source_manifest'], ref(out)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert all((ref(p) == r for p, r in before.items()))
    print(json.dumps(dict(preparation=ref(out), plans=plans, tests=10, runtime_tests=8, native_executed=False, admitted=False)))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=('retain', 'plan'))
    p.add_argument('--token', required=True)
    a = p.parse_args()
    lease(a.token, 400)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        {'retain': retain, 'plan': plan}[a.operation](a.token)
    finally:
        writer.close()
