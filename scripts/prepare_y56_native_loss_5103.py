"""Freeze actual task-loss test after the prior real-copy discovery."""

import argparse
import ast
from datetime import datetime
from research_5103 import ROOT, TOKEN, lease, ref, save, bind, tick, monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    lease()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        old = 'pipeline/analyses/y56_native_batch2_222ed2a6e80c43f6b1de2d050be97fa0_program.py'
        source = (ROOT / old).read_text(encoding='utf-8')
        tree = ast.parse(source)
        boundary = next(n.lineno for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'records' for t in n.targets))
        body = 'scripts/probe_y56_native_loss_5103.py'
        program = ''.join(source.splitlines(keepends=True)[:boundary - 1]) + '\nimport gc\n' + (ROOT / body).read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3,8))
        path = 'pipeline/analyses/y56_native_loss_' + TOKEN + '_program.py'
        assert not (ROOT / path).exists()
        (ROOT / path).write_text(program, encoding='utf-8', newline='\n')
        plan = dict(complete=True, id='AdaRES-Y56', token=TOKEN,
            preregistered_at=datetime.now().astimezone().isoformat(), program=ref(path), body=ref(body), setup=ref(old),
            discovery=ref('pipeline/analyses/y56_broader_unique_6debf88fed634a13b772f27bdc244bbd.json'),
            sample='Fixed real-copy pair8 and metadata-first pair0/all16expressions/all7layers; conditioned on prior discovery.',
            protocol='Same trainedW2/seed1999/actual masks/GT. Original/repeat/inverse-count head context only. '
                     'No-copy rows use original exact graph. Native task loss and each logit-layer gradient, all head gradients. '
                     'No optimizer, saved weights, source/config mutation or admission.',
            prediction='Rare initial-layer correction may or may not reach native loss. Preserve loss sign, zero cases '
                       'and repeats; do not claim efficacy, final selector harm, or population prevalence.', result='pending')
        prereg = save(path.replace('_program.py','_prereg.json'),plan)
        work = tick.load(ROOT/'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id']=='AdaRES-Y56')
        bind(candidate, ref(path), prereg, ref(body))
        candidate['next_action']='Execute frozen actual-copy native task-loss/backward test now; preserve all controls/outcomes.'
        tick.save(ROOT/'pipeline/research_work.json',work)
        print('Frozen same-ID Y56 actual task-loss reachability test.')
    finally:
        lock.close()


if __name__ == '__main__':
    main()
