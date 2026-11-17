"""Separate revision fixture directory; retain the failed harness assertion."""

import ast
from datetime import datetime
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    setup = ROOT / 'scripts/setup_y56_rev1_dd8e.py'
    failure = ref('pipeline/analyses/y56_cpu_rev1_' + TOKEN + '.log')
    for name in ('cpu', 'native', 'resources'):
        old = ROOT / ('pipeline/analyses/y56_' + name + '_rev1_' + TOKEN + '_program.py')
        tree = ast.parse(old.read_text(encoding='utf-8'))
        payload = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets)))
        body = ROOT / ('scripts/probe_y56_cpu_rev1_v2_dd8e.py' if name == 'cpu'
                       else 'scripts/probe_y56_' + name + '_dd8e.py')
        program = 'PAYLOAD=' + repr(payload) + '\n' + setup.read_text(encoding='utf-8') + '\n' + body.read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3, 8))
        path = 'pipeline/analyses/y56_' + name + '_rev1_v2_' + TOKEN + '_program.py'
        assert not (ROOT / path).exists()
        (ROOT / path).write_text(program, encoding='utf-8', newline='\n')
        plan = save(path.replace('_program.py', '_prereg.json'), dict(
            complete=True, id='AdaRES-Y56', revision=1, preregistered_at=datetime.now().astimezone().isoformat(),
            program=ref(path), body=ref(body), setup=ref(setup), bundle=candidate['tested_bundle'],
            retained_harness_failure=failure,
            repair='Use separate revision1 fixture directory rather than overwrite frozen initial files. '
                   'CPU old-class construction preserves RNG, restoring original seed fixture exactly.',
            production_source_or_scientific_gate_change=False, result='pending'))
        bind(candidate, ref(path), plan, ref(body))
    bind(candidate, failure, ref(setup), ref(__file__))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print('Frozen driver-only revision1 fixture isolation; initial fixtures and failure preserved.')
finally:
    lock.close()
