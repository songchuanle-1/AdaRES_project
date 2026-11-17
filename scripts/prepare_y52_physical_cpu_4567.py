"""Bind an isolated CPU probe to the exact local prototype bytes."""
import base64
from datetime import datetime
import json
from research_tick_4567 import ROOT, TOKEN, lease, ref, tick

lease()
bundle = tick.load(ROOT / ('pipeline/analyses/y52_prototype_bundle_4567_' + TOKEN + '.json'))
assert bundle['complete'] and len(bundle['files']) == 7
payload = {}
for row in bundle['files']:
    assert ref(row['path']) == row
    if row['path'].endswith('.py'):
        payload[row['path'].rsplit('/', 1)[-1]] = dict(base64=base64.b64encode((ROOT / row['path']).read_bytes()).decode(), sha256=row['sha256'])
body = (ROOT / 'scripts/probe_y52_physical_cpu_4567.py').read_text(encoding='utf-8')
program = ROOT / ('pipeline/analyses/y52_physical_cpu_v2_program_4567_' + TOKEN + '.py')
assert not program.exists()
source = 'PAYLOAD=' + repr(dict(files=payload)) + '\n' + body
compile(source, str(program), 'exec')
program.write_text(source, encoding='utf-8', newline='\n')
freeze = ROOT / ('pipeline/analyses/y52_physical_cpu_v2_prereg_4567_' + TOKEN + '.json')
assert not freeze.exists()
tick.save(freeze, dict(complete=True, id='AdaRES-Y52', preregistered_at=datetime.now().astimezone().isoformat(),
    original_failure_log=ref('pipeline/analyses/y52_physical_cpu_4567_' + TOKEN + '.log'),
    original_program=ref('pipeline/analyses/y52_physical_cpu_program_4567_' + TOKEN + '.py'),
    candidate_bundle_unchanged=ref('pipeline/analyses/y52_prototype_bundle_4567_' + TOKEN + '.json'),
    program=ref(program.relative_to(ROOT).as_posix()),
    test_change='Observe each leaf and an identical candidate repeat. Score and auxiliary-score gradients still require exact equality; other shared-component gradients require finite values and error no greater than measured repeat noise or prospective 8*float32eps*max(1, gradient scale). Exact loss and all unaffected loss terms remain required.',
    scope='Instrumentation clarification only. Do not erase v1 strict-equality failure, count it as native PASS, or infer efficacy; native model gate remains unrun.'))
print(json.dumps(dict(program=ref(program.relative_to(ROOT).as_posix()), prototype_files=len(bundle['files'])), indent=2))
