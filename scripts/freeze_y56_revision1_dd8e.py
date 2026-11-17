"""Bind exact prefilter revision to new immutable programs before execution."""

import ast
import base64
from datetime import datetime
import json
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    revision = candidate['revision_history'][-1]
    assert revision['revision'] == 1 and revision['result'] == 'pending'
    old_program = ROOT / ('pipeline/analyses/y56_cpu_' + TOKEN + '_program.py')
    tree = ast.parse(old_program.read_text(encoding='utf-8'))
    payload = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets)))
    original_decoder = base64.b64decode(payload['files']['dec.py']['base64']).decode('utf-8')
    payload['initial_class'] = ast.unparse(next(n for n in ast.parse(original_decoder).body
        if isinstance(n, ast.ClassDef) and n.name == 'QueryCompetitionIndicator'))
    bundle_original = tick.load(ROOT / candidate['tested_bundle']['path'])
    files = [ref(r['path']) for r in bundle_original['files']]
    bundle = save('pipeline/analyses/y56_bundle_rev1_' + TOKEN + '.json', dict(
        complete=True, id='AdaRES-Y56', revision=1, files=files,
        original_bundle=candidate['tested_bundle'], original_source_archive=ref(old_program),
        preregistration=ref('pipeline/analyses/y56_revision1_prereg_' + TOKEN + '.json'),
        reason='Exact raw-query injectivity prefilter; unchanged scientific semantics and all parameters.',
        created_at=datetime.now().astimezone().isoformat(), formal_uncommitted=True))
    payload['bundle'] = bundle
    payload['files'] = {(ROOT / r['path']).name: dict(sha256=r['sha256'],
        base64=base64.b64encode((ROOT / r['path']).read_bytes()).decode('ascii')) for r in files}
    setup = ROOT / 'scripts/setup_y56_prototype_dd8e.py'
    for name in ('cpu', 'native', 'resources'):
        body = ROOT / ('scripts/probe_y56_' + name + ('_rev1_dd8e.py' if name == 'cpu' else '_dd8e.py'))
        program = 'PAYLOAD=' + repr(payload) + '\n' + setup.read_text(encoding='utf-8') + '\n' + body.read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3, 8))
        path = 'pipeline/analyses/y56_' + name + '_rev1_' + TOKEN + '_program.py'
        assert not (ROOT / path).exists()
        (ROOT / path).write_text(program, encoding='utf-8', newline='\n')
        plan = save(path.replace('_program.py', '_prereg.json'), dict(
            complete=True, id='AdaRES-Y56', revision=1, preregistered_at=datetime.now().astimezone().isoformat(),
            program=ref(path), body=ref(body), bundle=bundle,
            same_cohort_and_original_controls=True, initial_reference_equality_added=name == 'cpu', result='pending'))
        bind(candidate, ref(path), plan, ref(body))
    bind(candidate, bundle, ref(__file__))
    candidate['tested_bundle'] = bundle
    revision['evidence'].append(bundle)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(bundle=bundle, revision=1, result='pending')))
finally:
    lock.close()
