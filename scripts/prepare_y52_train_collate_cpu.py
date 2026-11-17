"""Preregister and freeze a new real batch-size-two CPU dataflow control."""
import argparse
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--token', required=True)
args = parser.parse_args()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
    assert owner['token'] == args.token and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
    assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 600
    def ref(path):
        absolute = (ROOT / path).resolve()
        assert ROOT in absolute.parents
        return dict(path=path, sha256=hashlib.sha256(absolute.read_bytes()).hexdigest())
    for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
        assert hashlib.sha256((ROOT / 'pipeline/headless' / name).read_bytes()).hexdigest() == owner[key]
    bundle_path = 'pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json'
    bundle = tick.load(ROOT / bundle_path)
    payload = dict(files={})
    for row in bundle['files']:
        assert ref(row['path']) == row
        name = row['path'].rsplit('/', 1)[-1]
        if name in ('ipdn.py', 'loss.py', 'adares_y52_point_target_res.yaml'):
            payload['files'][name] = dict(base64=base64.b64encode((ROOT / row['path']).read_bytes()).decode(), sha256=row['sha256'])
    assert len(payload['files']) == 3
    body_path = 'scripts/probe_y52_train_collate_cpu.py'
    source = 'PAYLOAD=' + repr(payload) + '\n' + (ROOT / body_path).read_text(encoding='utf-8')
    program_path = 'pipeline/analyses/y52_train_collate_program_' + args.token + '.py'
    prereg_path = 'pipeline/analyses/y52_train_collate_prereg_' + args.token + '.json'
    assert not (ROOT / program_path).exists() and not (ROOT / prereg_path).exists()
    compile(source, program_path, 'exec')
    tick.save(ROOT / prereg_path, dict(complete=True, id='AdaRES-Y52', token=args.token,
        preregistered_at=datetime.now().astimezone().isoformat(), prototype=ref(bundle_path), body=ref(body_path),
        test='Use the SAME frozen eight TRAIN scenes/32 expressions, seed1999 per dataset access. Collate adjacent pairs with the ACTUAL production training collate (batch_size2). Compare frozen candidate expression point counts exactly with independent single-scene collates, repeat batch metadata, and native expanded-point IoU on three fixed empty/full/parity masks.',
        purpose='New actual cross-scene offset/annotation/count plumbing integration, beyond prior single-scene TRAIN replay and synthetic ragged-component controls. SAME frozen candidate, no implementation revision.',
        prediction='Single-scene and merged counts/annotation identity agree exactly, including mixed-label superpoints; point-IoU equals exact expanded reference. Failures preserved as implementation/instrumentation evidence, not automatic hypothesis falsification.',
        scope='CUDA hidden, CPU threads2; no model construction/weights/forward/backward, no validation selection. Native learned predictions, matcher, shared-query backward and resource/restore gates remain unrun.',
        timeout_seconds=300))
    (ROOT / program_path).write_text(source, encoding='utf-8', newline='\n')
    print(json.dumps(dict(preregistration=ref(prereg_path), program=ref(program_path)), indent=2))
finally:
    lock.close()
