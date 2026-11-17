"""Freeze an explanatory audit of the original unmet native-control criterion."""

import argparse
import ast
from datetime import datetime
from research_5103 import ROOT, TOKEN, lease, ref, save, bind, tick, monitor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--token', required=True)
    args = p.parse_args()
    assert args.token == TOKEN
    lease()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        old = 'pipeline/analyses/y57_native_v2_' + TOKEN + '_program.py'
        tree = ast.parse((ROOT / old).read_text(encoding='utf-8'))
        payload = ast.literal_eval(tree.body[0].value)
        source = 'PAYLOAD=' + repr(payload) + '\n' + (ROOT / 'scripts/setup_y57_prototype_5103.py').read_text(encoding='utf-8')
        source += '\n' + (ROOT / 'scripts/probe_y57_activity_audit_5103.py').read_text(encoding='utf-8')
        ast.parse(source, feature_version=(3, 8))
        path = 'pipeline/analyses/y57_activity_audit_' + TOKEN + '_program.py'
        assert not (ROOT / path).exists()
        (ROOT / path).write_text(source, encoding='utf-8', newline='\n')
        plan = dict(complete=True, id='AdaRES-Y57', token=TOKEN,
            preregistered_at=datetime.now().astimezone().isoformat(), original_program=ref(old),
            interrupted_log=ref('pipeline/analyses/y57_native_v2_' + TOKEN + '.log'),
            failure_log=ref('pipeline/analyses/y57_native_v2_retry_' + TOKEN + '.log'),
            program=ref(path), body=ref('scripts/probe_y57_activity_audit_5103.py'),
            sample='All same8TRAIN scenes/32expressions/four batch2 pairs/all7layers. No sample/threshold search.',
            test='Record original unchanged bounds and pass booleans for released initialization and '
                 'separate trained-W2 diagnostic state. Preserve all native loss gradients/new columns. '
                 'Strictly map only three changed shapes, common tensors exact, no saved weights/optimizer.',
            prediction='May distinguish weak initial feature variance/control signal from unreachable path. '
                       'No original control failure is converted to a PASS by this audit.', result='pending')
        prereg = save(path.replace('_program.py', '_prereg.json'), plan)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y57')
        bind(candidate, plan['failure_log'], plan['interrupted_log'], ref(path), prereg, plan['body'])
        candidate.update(stage='validation', scientific_status='original_native_shift_control_failed_pending_audit',
                         next_action='Run frozen released/trained-diagnostic explanatory audit; unchanged '
                                     'model and numerical bounds, no original control gate PASS yet.')
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print('Frozen explanatory audit; original failure preserved and no gate relaxed.')
    finally:
        lock.close()


if __name__ == '__main__':
    main()
