"""Freeze additive edge controls without changing a gate or production source."""

import ast
from datetime import datetime
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    previous = ROOT / ('pipeline/analyses/y56_cpu_rev2_' + TOKEN + '_program.py')
    tree = ast.parse(previous.read_text(encoding='utf-8'))
    payload = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets)))
    setup = ROOT / 'scripts/setup_y56_rev2_dd8e.py'
    body = ROOT / 'scripts/probe_y56_gradient_properties_dd8e.py'
    program = 'PAYLOAD=' + repr(payload) + '\n' + setup.read_text(encoding='utf-8') + '\n' + body.read_text(encoding='utf-8')
    ast.parse(program, feature_version=(3, 8))
    path = 'pipeline/analyses/y56_gradient_properties_' + TOKEN + '_program.py'
    assert not (ROOT / path).exists()
    (ROOT / path).write_text(program, encoding='utf-8', newline='\n')
    plan = save(path.replace('_program.py', '_prereg.json'), dict(
        complete=True, id='AdaRES-Y56', revision=2, preregistered_at=datetime.now().astimezone().isoformat(),
        program=ref(path), body=ref(body), setup=ref(setup), bundle=candidate['tested_bundle'],
        prediction='All outputs and query/mask/every-parameter gradients bitwise equal immutable initial Y56 on18cases.',
        controls='float32/64, singleton/three_classes/query-only/mask-only/partial-joint/padding/empty/zero-mass/mixed.',
        new_production_revision=False, thresholds_changed=False, result='pending'))
    bind(candidate, ref(path), plan, ref(body), ref(__file__))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print('Frozen additive18CPU edge/input/parameter-gradient properties, before execution.')
finally:
    lock.close()
