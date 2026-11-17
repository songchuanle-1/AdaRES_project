"""Freeze final-budget exact grouping realization and all affected gates."""

import ast
from datetime import datetime
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    setup = ROOT / 'scripts/setup_y56_rev2_dd8e.py'
    previous_bundle = candidate['tested_bundle']
    old_bundle = tick.load(ROOT / previous_bundle['path'])
    files = [ref(r['path']) for r in old_bundle['files']]
    bundle = save('pipeline/analyses/y56_bundle_rev2_' + TOKEN + '.json', dict(
        complete=True, id='AdaRES-Y56', revision=2, files=files,
        previous_bundle=previous_bundle, original_bundle=old_bundle['original_bundle'],
        original_source_archive=old_bundle['original_source_archive'],
        preregistration=ref('pipeline/analyses/y56_revision2_prereg_' + TOKEN + '.json'),
        created_at=datetime.now().astimezone().isoformat(), formal_uncommitted=True,
        semantics_unchanged=True, parameter_delta_expected=0))
    for name in ('cpu', 'native', 'resources'):
        old = ROOT / ('pipeline/analyses/y56_' + name + '_rev1_v2_' + TOKEN + '_program.py')
        tree = ast.parse(old.read_text(encoding='utf-8'))
        payload = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets)))
        import base64
        payload['files'] = {(ROOT / r['path']).name: dict(sha256=r['sha256'],
            base64=base64.b64encode((ROOT / r['path']).read_bytes()).decode('ascii')) for r in files}
        payload['bundle'] = bundle
        body = ROOT / ('scripts/probe_y56_cpu_rev1_v2_dd8e.py' if name == 'cpu'
                       else 'scripts/probe_y56_' + name + '_dd8e.py')
        program = 'PAYLOAD=' + repr(payload) + '\n' + setup.read_text(encoding='utf-8') + '\n' + body.read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3, 8))
        path = 'pipeline/analyses/y56_' + name + '_rev2_' + TOKEN + '_program.py'
        assert not (ROOT / path).exists()
        (ROOT / path).write_text(program, encoding='utf-8', newline='\n')
        plan = save(path.replace('_program.py', '_prereg.json'), dict(
            complete=True, id='AdaRES-Y56', revision=2, preregistered_at=datetime.now().astimezone().isoformat(),
            program=ref(path), body=ref(body), setup=ref(setup), bundle=bundle,
            unchanged_cohort_controls_and_gates=True, original_reference_equality_test=True, result='pending'))
        bind(candidate, ref(path), plan, ref(body))
    bind(candidate, bundle, ref(setup), ref(__file__), ref('scripts/test_y56_prototype_rev2_dd8e.py'))
    candidate['tested_bundle'] = bundle
    candidate['revision_history'][-1]['evidence'].append(bundle)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print('Frozen revision2 exact grouping and unchanged actual gate protocols; old evidence preserved.')
finally:
    lock.close()
