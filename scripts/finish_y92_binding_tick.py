"""Refresh sequential observations and finish without aging evidence between calls."""
import argparse
import ast
from datetime import datetime
import json
import os
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure

SKILL = 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline'


class LabelsOnly(ast.NodeTransformer):
    def __init__(self):
        self.counts = {'_final_': 0, 'pipeline/checks/y92_binding_closeout_': 0}

    def visit_Constant(self, node):
        changes = {'_final_': '_closeout_retry_',
                   'pipeline/checks/y92_binding_closeout_': 'pipeline/checks/y92_binding_closeout_retry_'}
        if isinstance(node.value, str) and node.value in changes:
            self.counts[node.value] += 1
            node.value = changes[node.value]
        return node


def command(args):
    completed = subprocess.run([sys.executable, '-B'] + args, cwd=ROOT,
        capture_output=True, timeout=120)
    sys.stdout.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    assert completed.returncode == 0, 'Preserve the failure and reconcile this owned lease'


def finish(token):
    lease(token, 240)
    assert os.environ['CODEX_THREAD_ID'] == '01a07e32-78bf-7670-925b-370911ea48db'
    source = ROOT / 'scripts/record_y92_binding_closeout.py'
    destination = ROOT / 'scripts/record_y92_binding_closeout_v2.py'
    manifest = ROOT / ('pipeline/checks/y92_closeout_freshness_retry_' + token + '.json')
    assert not destination.exists() and not manifest.exists()
    original = source.read_text(encoding='utf-8')
    transformer = LabelsOnly()
    parsed = transformer.visit(ast.parse(original))
    assert transformer.counts == {'_final_': 1, 'pipeline/checks/y92_binding_closeout_': 1}
    restored = ast.parse(ast.unparse(ast.fix_missing_locations(parsed)))
    for item in ast.walk(restored):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            item.value = {
                '_closeout_retry_': '_final_',
                'pipeline/checks/y92_binding_closeout_retry_': 'pipeline/checks/y92_binding_closeout_',
            }.get(item.value, item.value)
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        ast.parse(original), include_attributes=False)
    destination.write_text(ast.unparse(ast.fix_missing_locations(parsed)) + '\n',
                           encoding='utf-8', newline='\n')
    ast.parse(destination.read_text(encoding='utf-8'), feature_version=(3, 8))
    assert ref(source)['sha256'] == hashlib_digest(original.encode())
    old_observations = [ref('pipeline/checks/production_adares-' + cid + '_' + token + '_final.json')
                        for cid in ('y90', 'y91')]
    old_occupation = [ref('pipeline/checks/gpu_occupation_' + name + '_final_' + token + '.json')
                      for name in ('server-1', 'server-2')]
    assert not (ROOT / ('pipeline/checks/y92_binding_closeout_' + token + '.json')).exists()
    tick.save(manifest, dict(complete=True, token=token,
        recorded_at=datetime.now().astimezone().isoformat(),
        original_helper=ref(source), retry_helper=ref(destination),
        original_command='python -B scripts/record_y92_binding_closeout.py --token ' + token,
        observed_failure='AssertionError: Refresh the actual observation under a new label',
        failure_location='record -> fresh(observed, observed_local), unchanged 0<=age<300 seconds guard',
        first_attempt_wrote_closeout=False, original_observations=old_observations,
        original_occupation=old_occupation, original_files_preserved=True,
        retry_changes_only_immutable_output_labels=True, normalized_AST_identical=True,
        scientific_or_numerical_controls_changed=False,
        CPU_or_native_scientific_tests_replayed=False, evidence=[ref(__file__)]))
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y92')
        candidate['evidence'].append(ref(manifest))
        tick.save(ROOT / 'pipeline/research_work.json', work)
    finally:
        writer.close()
    for cid in ('AdaRES-Y90', 'AdaRES-Y91'):
        command(['scripts/y91_deployment.py', 'observe', '--id', cid,
                 '--token', token, '--label', 'closeout_retry'])
    for name in ('server-1', 'server-2'):
        command([SKILL + '/gpu_occupation.py', 'AdaRES_project', '--token', token,
                 '--server', name, '--label', 'closeout_retry'])
    command(['scripts/record_y92_binding_closeout_v2.py', '--token', token])
    closeout_path = ROOT / ('pipeline/checks/y92_binding_closeout_retry_' + token + '.json')
    closeout = tick.load(closeout_path)
    assert closeout['complete'] is True and closeout['token'] == token
    notes = tick.load(ROOT / ('pipeline/reports/notes_' + token + '.json'))
    for cid, entry in closeout['observations'].items():
        phase = entry['phase']
        notes['actions'].append(cid + ' 收尾实测：' + phase['phase'] + ' '
            + str(phase['epoch']) + '/' + str(phase['total']) + '，步数 '
            + str(phase['step']) + '/' + str(phase['count'])
            + '；原进程、来源、原生 YAML 与递增计数均已核验。')
    notes['actions'].append('收尾证据：' + closeout_path.relative_to(ROOT).as_posix()
        + '；首次过期观测被原300秒守卫拒绝，保留原文件后顺序刷新，守卫未改。')
    notes_path = ROOT / ('pipeline/reports/notes_refreshed_' + token + '.json')
    assert not notes_path.exists()
    tick.save(notes_path, notes)
    command([SKILL + '/foreground_tick.py', 'AdaRES_project', 'report', '--token', token,
             '--notes', str(notes_path)])
    command([SKILL + '/foreground_tick.py', 'AdaRES_project', 'finish', '--token', token])
    receipt = tick.load(ROOT / ('pipeline/tick_receipts/' + token + '.json'))
    print(json.dumps(dict(closeout=ref(closeout_path), freshness_retry=ref(manifest),
        observations=closeout['observations'], receipt_status=receipt.get('status'),
        lease_release=receipt.get('lease_release')), ensure_ascii=False))


def hashlib_digest(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    finish(parser.parse_args().token)
