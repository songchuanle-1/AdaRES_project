"""Preregister or record one CPU-only check of the frozen native observer."""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor


def ref(path):
    absolute = (ROOT / path).resolve()
    assert ROOT in absolute.parents and absolute.is_file()
    return dict(path=path, sha256=hashlib.sha256(absolute.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', required=True)
    parser.add_argument('--result')
    args = parser.parse_args()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == args.token and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 240
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert hashlib.sha256((ROOT / 'pipeline/headless' / name).read_bytes()).hexdigest() == owner[key]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y52')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        for row in candidate['evidence']:
            assert ref(row['path']) == row
        plan_path = 'pipeline/analyses/y52_capture_observer_cpu_prereg_' + args.token + '.json'
        program_path = 'pipeline/analyses/y52_capture_observer_cpu_program_' + args.token + '.py'
        if args.result:
            assert args.token in args.result
            result = tick.load(ROOT / args.result)
            assert result['complete'] and result['no_GPU']
            for key in ('actual_frozen_observer_function_executed_on_CPU', 'matched_index_order_preserved',
                        'original_input_identity_preserved', 'original_return_identity_preserved',
                        'inputs_unchanged', 'final_auxiliary_labels_correct',
                        'empty_matches_recorded_without_fake_zeros', 'inconsistent_expanded_reference_rejected'):
                assert result[key] is True, key
            for key in ('native_model_constructed', 'native_sparse_forward_executed',
                        'actual_TRAIN_match_frequency_measured', 'native_backward_executed',
                        'resources_or_restore_passed', 'scientific_gate_passed'):
                assert result[key] is False, key
            assert any(row['path'] == plan_path for row in candidate['evidence'])
            assert args.result not in {row['path'] for row in candidate['evidence']}
            candidate['evidence'].append(ref(args.result))
            candidate['attempted_alternatives'].append('Exact frozen native-capture observer passed CPU synthetic instrumentation controls: original input/return identity, match index order, final/aux labels, empty metadata and reference-mismatch rejection. No native model, real matching or scientific gate executed.')
            tick.save(ROOT / 'pipeline/research_work.json', work)
            print(json.dumps(dict(recorded=ref(args.result), scientific_gate_passed=False), indent=2))
            return
        assert not (ROOT / plan_path).exists() and not (ROOT / program_path).exists()
        frozen_path = 'pipeline/analyses/y52_native_matched_capture_prereg_5b882c0928b14a3bab6799e9d15a5bf4.json'
        frozen = tick.load(ROOT / frozen_path)
        assert ref(frozen['body']['path']) == frozen['body']
        bundle = tick.load(ROOT / frozen['prototype']['path'])
        for row in bundle['files']:
            assert ref(row['path']) == row
        loss = next(row for row in bundle['files'] if row['path'].endswith('/loss.py'))
        payload = {}
        for name, row in (('observer', frozen['body']), ('loss', loss)):
            payload[name] = dict(base64=base64.b64encode((ROOT / row['path']).read_bytes()).decode(), sha256=row['sha256'])
        body_path = 'scripts/probe_y52_capture_observer_cpu.py'
        source = 'PAYLOAD=' + repr(payload) + '\n' + (ROOT / body_path).read_text(encoding='utf-8')
        compile(source, program_path, 'exec')
        ast.parse(source, feature_version=(3, 8))
        plan = dict(complete=True, id='AdaRES-Y52', token=args.token,
            preregistered_at=datetime.now().astimezone().isoformat(),
            frozen_native_plan=ref(frozen_path), frozen_observer=frozen['body'],
            body=ref(body_path), preparer=ref('scripts/prepare_y52_capture_observer_cpu.py'),
            test='Execute ONLY the exact observer AST on two ragged synthetic expressions over final/aux calls, an empty-match case and an intentionally inconsistent expanded reference. Spy original delegate must receive the identical objects and its return must be passed through unchanged. Preserve actual supplied index order; reject bad reference before delegation. CUDA hidden.',
            purpose='New instrumentation reliability test for the newly prepared native capture; not repetition of model-construction or TRAIN-count controls, and not scientific mechanism evidence.',
            native_model_or_TRAIN_execution=False, scientific_gate_passed=False,
            limitation='A spy delegate cannot certify full Criterion behavior, real FPS/matching, backward, GPU resources or efficacy.')
        (ROOT / program_path).write_text(source, encoding='utf-8', newline='\n')
        tick.save(ROOT / plan_path, plan)
        for path in (body_path, 'scripts/prepare_y52_capture_observer_cpu.py', program_path, plan_path):
            candidate['evidence'].append(ref(path))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(plan=ref(plan_path), program=ref(program_path), native_GPU_executed=False), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
