"""Select only retained Y93 additions and observe Y92/Y93 without replaying runs."""
import argparse
import ast
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from observe_y69_server1_phase import parse_phase, phase_key, progressed, REMOTE
import closure
import y92_deployment as previous
Y91, Y92 = (previous.Y91, previous.Y92)
Y93 = 'daa93b4375f144f0e79da92f85fe1fe27e877e18'

def require_running_parent_handoff(work, candidate):
    original = next((row for row in work['candidates'] if row['id'] == 'AdaRES-Y92'))
    assert original['stage'] == 'training'
    plan = tick.load(closure.artifact(ROOT, original['terminal_handoff_plan']))
    assert plan['complete'] is True and plan['id'] == original['id']
    assert plan['tested_additive_native_evidence'] == candidate['native_smoke_result']
    assert plan['payload']['allowed_markers'] == [Y92, Y93]
    assert plan['CPU_guard_tests_passed'] == 30 and plan['additive_guard_tests_passed'] >= 8
    for entry in plan['dependencies']:
        closure.artifact(ROOT, entry)
VERSIONS = previous.VERSIONS + (Y93,)
PACKAGES = dict(previous.PACKAGES, **{'AdaRES-Y93': 'adares_y93_evidence_hidden_skip'})
ADDITIONS = dict(previous.ADDITIONS, **{Y93: 'AdaRES-Y93'})
git, adapt = (previous.git, previous.adapt)

def paths(cid):
    if cid != 'AdaRES-Y93':
        return previous.paths(cid)
    from prepare_y93_evidence_hidden_skip import NAMES
    package = PACKAGES[cid]
    return ['AdaRES/ipdn/model_paper5/' + package + '/' + name for name in NAMES] + ['AdaRES/configs_song/model_paper5/' + package + '_res.yaml']

def validate_delta(rows, cid):
    assert cid == 'AdaRES-Y93'
    assert len(rows) == 7 and all((len(row) == 2 and row[0] == 'A' for row in rows))
    assert {row[1] for row in rows} == set(paths(cid))

def contract(commit, head, cid):
    assert commit in VERSIONS and head in VERSIONS
    start, end = (VERSIONS.index(commit), VERSIONS.index(head))
    assert start <= end
    deployments = {}
    for version in VERSIONS[start:end + 1]:
        stop = VERSIONS.index(version)
        allowed = {path for item, candidate in ADDITIONS.items() if start < VERSIONS.index(item) <= stop for path in paths(candidate)}
        rows = [row.split('\t') for row in git('diff', '--no-renames', '--name-status', commit, version).decode().splitlines()]
        assert len(rows) == len(allowed) and all((status == 'A' for status, _ in rows))
        assert {path for _, path in rows} == allowed
        entry = dict(source_files={}, configs={})
        for _, path in rows:
            raw = git('show', version + ':' + path)
            assert (ROOT / path).read_bytes() == raw
            relative = path[len('AdaRES/'):]
            if path.endswith('.yaml'):
                entry['configs'][relative] = yaml.safe_load(raw)
            else:
                entry['source_files'][relative] = hashlib.sha256(raw).hexdigest()
        deployments[version] = entry
    for path in paths(cid):
        assert (ROOT / path).read_bytes() == git('show', commit + ':' + path) == git('show', head + ':' + path)
    return deployments

def select(token):
    from advance_y93 import prepared, validate_result
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, _, frozen = prepared(work)
    assert candidate['stage'] == 'validation' and (not candidate.get('admission_receipt'))
    validate_result(tick.load(closure.artifact(ROOT, candidate['native_smoke_result'])), candidate, frozen)
    require_running_parent_handoff(work, candidate)
    assert git('rev-parse', 'HEAD').decode().strip() == Y92
    assert git('rev-parse', Y93 + '^').decode().strip() == Y92
    contract(Y92, Y93, 'AdaRES-Y92')
    rows = [row.split('\t') for row in git('diff', '--no-renames', '--name-status', Y92, Y93).decode().splitlines()]
    validate_delta(rows, 'AdaRES-Y93')
    index_before = git('ls-files', '--stage', '-z')

    def entries(raw):
        return {row.split(b'\t', 1)[1]: row for row in raw.split(b'\x00') if row}
    before = entries(index_before)
    for _, path in rows:
        assert path.encode() not in before
        assert (ROOT / path).read_bytes() == git('show', Y93 + ':' + path)
    branch = git('symbolic-ref', 'HEAD').decode().strip()
    assert branch == 'refs/heads/master'
    out = ROOT / ('pipeline/checks/adares-y93_source_selection_' + token + '.json')
    snapshot, binary = (out.with_suffix('.index'), out.with_suffix('.index.bin'))
    assert not any((path.exists() for path in (out, snapshot, binary)))
    index_path = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    snapshot.write_bytes(index_before)
    binary.write_bytes(index_path.read_bytes())
    for _, path in rows:
        blob = git('rev-parse', Y93 + ':' + path).decode().strip()
        subprocess.run(['git', 'update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + path], cwd=ROOT, check=True)
    subprocess.run(['git', 'update-ref', '-m', 'AdaRES tested additive Y93', branch, Y93, Y92], cwd=ROOT, check=True)
    after = entries(git('ls-files', '--stage', '-z'))
    added = {path.encode() for _, path in rows}
    assert {path: row for path, row in after.items() if path not in added} == before
    assert all(((ROOT / path).read_bytes() == git('show', Y93 + ':' + path) for _, path in rows))
    tick.save(out, dict(complete=True, token=token, id='AdaRES-Y93', parent=Y92, target=Y93, only_added_files=rows, previous_index=ref(snapshot), previous_binary_index=ref(binary), other_index_entries_preserved=True, working_files_modified=False, helper=ref(__file__)))
    print(json.dumps(dict(selection=ref(out), HEAD=Y93)))

def observe(token, cid, label):
    assert cid in ('AdaRES-Y92', 'AdaRES-Y93') and label.replace('_', '').isalnum()
    queue = monitor.load_queue(str(ROOT))
    task = next((task for tasks in queue['gpu_queues'].values() for task in tasks if task['id'] == cid))
    assert task['status'] == 'running' and task['launch_verified'] is True
    name, commit, power = {'AdaRES-Y92': ('server-1', Y92, 350.0), 'AdaRES-Y93': ('server-2', Y93, 280.0)}[cid]
    assert (task['server'], str(task['gpu']), task['code_commit']) == (name, '0', commit)
    server = monitor.load_servers(str(ROOT))[name]
    closure.validate_adares_server(ROOT, server)
    closure.admission(ROOT, task)
    assert monitor.load_running(str(ROOT))[name]['0'] == cid
    head = git('rev-parse', 'HEAD').decode().strip()
    deployments = contract(commit, head, cid)
    for entry in deployments.values():
        entry['configs'] = {path: adapt(value, server) for path, value in entry['configs'].items()}
    payload = dict(root=server['remote_code_root'], task=task, source_files={path[len('AdaRES/'):]: ref(path)['sha256'] for path in paths(cid)[:-1]}, expected_config=adapt(yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_bytes()), server), deployments=deployments)
    body = REMOTE.replace('dl02-MS-7B73', server['expected_hostname'])
    body = body.replace("assert (root/'.codex_source_commit').read_text().strip()==task['code_commit']", "\nmarker=(root/'.codex_source_commit').read_text().strip()\nassert marker in P['deployments']\nfor relative,digest in P['deployments'][marker]['source_files'].items():\n    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest\nfor relative,value in P['deployments'][marker]['configs'].items():\n    assert yaml.safe_load((root/relative).read_text())==value\n")
    body = body.replace('print(json.dumps(dict(complete=True', "assert float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())==" + repr(power) + '\nprint(json.dumps(dict(complete=True')
    body = body.replace("source_marker=task['code_commit']", 'source_marker=marker')
    program = 'P=' + repr(payload) + '\n' + '\n\n'.join((inspect.getsource(function) for function in (parse_phase, phase_key, progressed))) + '\n' + body
    ast.parse(program, feature_version=(3, 8))
    out = ROOT / ('pipeline/checks/production_' + cid.lower() + '_' + token + '_' + label + '.json')
    assert not out.exists() and (not out.with_suffix('.log').exists())
    before_q, before_r = (ref('pipeline/queue.json'), ref('pipeline/running.json'))
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', server['user'] + '@' + server['host'], 'PYTHONDONTWRITEBYTECODE=1 ' + server['python'] + ' -B -'], input=program.encode(), capture_output=True, timeout=45)
    out.with_suffix('.log').write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0, 'Preserve failed observation; never restart'
    value = json.loads(response.stdout)
    assert value['native_progress_confirmed'] and progressed(value['before'], value['after'])
    value.update(id=cid, token=token, observed_local=datetime.now().astimezone().isoformat(), evidence=[ref(__file__), ref(out.with_suffix('.log'))])
    tick.save(out, value)
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next((candidate for candidate in work['candidates'] if candidate['id'] == cid))
    candidate['evidence'].append(ref(out))
    candidate['latest_training_observation'] = ref(out)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert (before_q, before_r) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
    print(json.dumps(dict(evidence=ref(out), phase=value['after']['latest'], pid=task['pid'], source_marker=value['source_marker'])))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('select', 'observe'))
    parser.add_argument('--id', choices=('AdaRES-Y92', 'AdaRES-Y93'), default='AdaRES-Y93')
    parser.add_argument('--token', required=True)
    parser.add_argument('--label', default='progress')
    args = parser.parse_args()
    lease(args.token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        select(args.token) if args.operation == 'select' else observe(args.token, args.id, args.label)
    finally:
        writer.close()
