"""Freeze additional task-facing controls and order-balanced timing audit."""

import ast
from datetime import datetime
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    previous = ROOT / ('pipeline/analyses/y56_native_rev2_' + TOKEN + '_program.py')
    tree = ast.parse(previous.read_text(encoding='utf-8'))
    payload = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets)))
    setup = ROOT / 'scripts/setup_y56_rev2_dd8e.py'
    body = ROOT / 'scripts/probe_y56_margin_timing_v2_dd8e.py'
    program = 'PAYLOAD=' + repr(payload) + '\n' + setup.read_text(encoding='utf-8') + '\n' + body.read_text(encoding='utf-8')
    ast.parse(program, feature_version=(3, 8))
    path = 'pipeline/analyses/y56_margin_timing_v2_' + TOKEN + '_program.py'
    assert not (ROOT / path).exists()
    (ROOT / path).write_text(program, encoding='utf-8', newline='\n')
    plan = save(path.replace('_program.py', '_prereg.json'), dict(
        complete=True, id='AdaRES-Y56', revision=2, preregistered_at=datetime.now().astimezone().isoformat(),
        program=ref(path), body=ref(body), setup=ref(setup), bundle=candidate['tested_bundle'],
        prediction='Existing logit change should also affect binary class margin/confidence, rather than '
                   'only common-mode logit shifts. Record actual native task gradient dot intervention and exact repeat.',
        diagnostic='Fixed discovery pair4, released init and separate strict trained-W2; all7layers, no forced copies.',
        timing='Same fixed10TRAIN scenes/common initial weights;2warmups/5measured alternating parent-candidate '
               'order within every scene/round. Two models resident in same single-GPU process, not production peak memory.',
        retained_harness_failure=ref('pipeline/analyses/y56_margin_timing_gpu1_' + TOKEN + '.log'),
        driver_repair='Project IPDN eval() delegates to train(False), which returns None; '
                      'use standalone eval() calls, not their return values. No source/model/test threshold change.',
        no_change_to_model_or_gates=True, no_efficacy_threshold=True, result='pending'))
    bind(candidate, ref(path), plan, ref(body), ref(__file__),
         ref('pipeline/analyses/y56_margin_timing_gpu1_' + TOKEN + '.log'))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print('Frozen task-facing readout and order-balanced native paired timing; no model/gate changes.')
finally:
    lock.close()
