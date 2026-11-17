"""Freeze a native matched-query diagnostic without running or passing its gates."""
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
    args = parser.parse_args()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None, 'Pipeline writer busy'
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == args.token and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 900
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert hashlib.sha256((ROOT / 'pipeline/headless' / name).read_bytes()).hexdigest() == owner[key]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y52')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        for row in candidate['evidence']:
            assert ref(row['path']) == row
        bundle_path = 'pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json'
        bundle = tick.load(ROOT / bundle_path)
        payload = dict(files={}, parent_files=[])
        for row in bundle['files']:
            assert ref(row['path']) == row
            payload['files'][row['path'].rsplit('/', 1)[-1]] = dict(
                base64=base64.b64encode((ROOT / row['path']).read_bytes()).decode(), sha256=row['sha256'])
        for row in bundle['source_files']:
            assert ref(row['path']) == row
            payload['parent_files'].append(dict(path=row['path'].removeprefix('AdaRES/'), sha256=row['sha256']))
        payload['capture_sha256'] = '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
        payload['train_source_sha256'] = ref('AdaRES/tools/train.py')['sha256']
        body_path = 'scripts/probe_y52_native_matched_capture.py'
        program_path = 'pipeline/analyses/y52_native_matched_capture_program_' + args.token + '.py'
        plan_path = 'pipeline/analyses/y52_native_matched_capture_prereg_' + args.token + '.json'
        assert not (ROOT / program_path).exists() and not (ROOT / plan_path).exists()
        body = (ROOT / body_path).read_text(encoding='utf-8')
        source = 'PAYLOAD=' + repr(payload) + '\n' + body
        compile(source, program_path, 'exec')
        ast.parse(source, feature_version=(3, 8))
        # Exercise the fail-closed entry before any torch/gorilla import or CUDA call.
        prefix = body.split('import gorilla', 1)[0]
        try:
            exec(compile(prefix, '<native-diagnostic-no-GPU-guard>', 'exec'), {'P': {'physical_gpu': None}})
        except AssertionError:
            guard_rejected_no_GPU = True
        else:
            raise AssertionError('Native diagnostic accepted missing exclusive GPU')
        plan = dict(complete=True, id='AdaRES-Y52', token=args.token,
            prepared_at=datetime.now().astimezone().isoformat(), prototype=ref(bundle_path),
            body=ref(body_path), preparer=ref('scripts/prepare_y52_native_matched_capture.py'),
            production_train_source=ref('AdaRES/tools/train.py'),
            frozen_test='SAME eight TRAIN scenes/32 expressions, actual batch_size2, seed1999 per dataset access; load released backbone with exact production loader. Observe actual final/auxiliary loss_masks matching inputs without changing them. Compare actual matched hard-superpoint IoU to exact expanded-point IoU and the frozen point-count target; report existing >0.5 support disagreements without selecting masks or thresholds.',
            controls='Full packaged source compiles for Python3.8 grammar; missing exclusive GPU is rejected before framework import. No candidate model file modified, no substitute sample/FPS/match indices, no checkpoint or learned tensors saved.',
            compilation_passed=True, no_GPU_guard_rejected=guard_rejected_no_GPU,
            native_forward_executed=False, native_backward_executed=False,
            resources_or_restore_passed=False, scientific_gate_passed=False,
            scope='Executable diagnostic preparation only, not a new scientific test, implementation revision, candidate, native gate PASS, commit permission or training admission.',
            next_action='Use existing run_adares_authorized_probe.py --gpu --gpu-id <fresh-idle-authorized-card> --seconds 600 with this frozen program ONLY after live idleness and exclusive lease checks. No execution if occupied.',
            limitation='An initial native match capture is only the first GPU diagnostic. Full paired backward/shared-query, real optimizer steps, complete restoration and resource controls still remain pending even if it runs.')
        (ROOT / program_path).write_text(source, encoding='utf-8', newline='\n')
        tick.save(ROOT / plan_path, plan)
        for path in (body_path, 'scripts/prepare_y52_native_matched_capture.py', program_path, plan_path):
            candidate['evidence'].append(ref(path))
        candidate['attempted_alternatives'].append('Native matched-query diagnostic prepared under current lease: frozen eight-scene TRAIN inputs, exact source binding, Python3.8 compile and fail-closed no-GPU entry only; not executed on GPU and no scientific gate PASS.')
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(plan=ref(plan_path), program=ref(program_path),
            Python38_compilation_passed=True, no_GPU_guard_rejected=True,
            native_GPU_executed=False, scientific_gate_passed=False), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
