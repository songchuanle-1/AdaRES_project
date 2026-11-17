"""Prepare exact Y99 completion checks without loading checkpoints or stopping it."""
import argparse
import ast
import copy
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from finalize_y99_fixed30 import remote
import finalize_y99_fixed30 as archive

def prepare(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        server, queue, task, work, candidate = archive.current()
        before_queue = copy.deepcopy(queue)
        before_running = monitor.load_running(str(ROOT))
        archive.expected_mapping(before_running)
        base = ROOT / ('pipeline/checks/y99_terminal_preparation_' + token)
        output = base.with_suffix('.json')
        tests_path = base.with_name(base.name + '_tests.log')
        location_path = base.with_name(base.name + '_location.json')
        assert not any((p.exists() for p in (output, tests_path, location_path)))
        tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts', '-p', 'test_y99_terminal_handoff.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
        tests_path.write_bytes(tests.stdout + tests.stderr)
        assert tests.returncode == 0 and b'Ran 30 tests' in tests_path.read_bytes()
        import yaml
        config = yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_text(encoding='utf-8'))

        def adapt(value):
            if isinstance(value, dict):
                return {k: adapt(v) for k, v in value.items()}
            if isinstance(value, list):
                return [adapt(v) for v in value]
            if isinstance(value, str):
                for m in server.get('runtime_path_mappings', []):
                    old, new = (m['source'].rstrip('/'), m['target'].rstrip('/'))
                    if value == old or value.startswith(old + '/'):
                        value = new + value[len(old):]
            return value
        folder = ROOT / 'AdaRES/ipdn/model_paper5/adares_y99_spatial_mask_skip'
        files = {p.relative_to(ROOT / 'AdaRES').as_posix(): ref(p)['sha256'] for p in folder.glob('*.py')}
        assert len(files) == 6
        payload = dict(id=task['id'], pid=task['pid'], log=task['log'], config=task['config'], code_commit=task['code_commit'], root=server['remote_code_root'], work=server['work_dir'], basename=Path(task['config']).stem, source_files=files, expected_config=adapt(config))
        payload.update(archive.build_contract(task, server))
        discovery = archive.DISCOVER.replace('print(json.dumps(dict(complete=True', archive.IDENTITY + 'print(json.dumps(dict(complete=True')
        for body in (discovery, archive.REMOTE, archive.RESOURCE):
            ast.parse('P=' + repr(payload) + '\n' + body, feature_version=(3, 8))
        program_path = base.with_name(base.name + '_location_program.py')
        assert not program_path.exists()
        program_path.write_text('P=' + repr(payload) + '\n' + discovery, encoding='utf-8', newline='\n')
        location = remote(server, program_path.read_text(encoding='utf-8'), location_path)
        archive.validate_location(location, payload)
        assert location['complete'] and location['checkpoint_loaded'] is False
        assert location['remote_mutations'] is False and location['log'] == task['log']
        tick.save(location_path, location)
        assert monitor.load_queue(str(ROOT)) == before_queue
        assert monitor.load_running(str(ROOT)) == before_running
        dependencies = [ref(p) for p in ('scripts/finalize_y99_fixed30.py', 'scripts/test_y99_terminal_handoff.py', 'scripts/prepare_y99_terminal_handoff.py', 'scripts/y99_terminal_contract.py', 'scripts/y99_deployment_v2.py', 'scripts/create_y99_terminal_helpers.py', 'scripts/finalize_y61_fixed30.py', 'scripts/finalize_y58_y59_handoff.py', 'scripts/reconcile_y62_y63_server1.py', 'scripts/check_ef_execution.py')]
        tick.save(output, dict(complete=True, token=token, id=task['id'], observed_at=datetime.now().astimezone().isoformat(), payload=payload, declared_location=ref(location_path), CPU_guard_tests_passed=30, dependencies=dependencies, post_training_freeze=candidate['post_training_freeze'], admission=ref('pipeline/admissions/AdaRES-Y99.json'), full_terminal_verification_executed=False, checkpoint_loaded=False, archived=False, training_stopped=False, queue_changed=False, remote_mutations=False, required_before_archive='Exact PID and same-config train process absent; native Train/Val1..30, stable finite last30/best/model/optimizer/scheduler, unchanged source/YAML/receipt/freeze. Use explicit server-1 GPU0 and350W; retain all original checkpoints and other queue entries.', next_command='python -B scripts/finalize_y99_fixed30.py --token <fresh-owned-token>', evidence=[ref(__file__), ref(tests_path), ref(program_path), ref(location_path), ref(location_path.with_suffix('.log'))] + dependencies))
        candidate['terminal_handoff_plan'] = ref(output)
        candidate['evidence'].extend([ref(output), ref(__file__), ref(tests_path)] + dependencies[:2])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(__import__('json').dumps(dict(plan=ref(output), tests=30, location=location['checkpoint_directory'], archived=False, checkpoint_loaded=False)))
    finally:
        writer.close()
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--token', required=True)
    prepare(p.parse_args().token)
