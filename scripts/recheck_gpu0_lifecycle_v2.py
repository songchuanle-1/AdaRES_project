"""Handle the collector's idle-slot stage transition without altering v1 evidence."""

import json
from pathlib import Path
import sys

import recheck_gpu0_lifecycle as v1


original_role = v1.role


def role(candidate, experiment):
    # The collector clears the resource blocker on a genuinely idle GPU0.
    # That is still an unrun smoke test, not an admission or completed experiment.
    if (candidate.get('id') == 'AdaRES-Y60' and candidate.get('stage') == 'validation'
            and not candidate.get('blocker_kind') and not candidate.get('scientific_status')):
        candidate = dict(candidate, blocker_kind='exclusive_gpu_slot')
    return original_role(candidate, experiment)


def main():
    v1.role = role
    try:
        v1.main()
    finally:
        v1.role = original_role
    args = sys.argv[1:]
    token = args[args.index('--token') + 1]
    v1.lease(token, 90)
    lock = v1.monitor.acquire_lock(str(v1.ROOT))
    assert lock is not None
    try:
        work = v1.tick.load(v1.ROOT / 'pipeline/research_work.json')
        result = v1.ROOT / ('pipeline/checks/gpu0_lifecycle_' + token + '.json')
        additions = [v1.ref(__file__), v1.ref(Path(__file__).with_name('test_gpu0_lifecycle_v2.py'))]
        ids = {row['id'] for row in v1.tick.load(result)['records']}
        for candidate in work['candidates']:
            if candidate['id'] in ids:
                candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
        v1.tick.save(v1.ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(runner_evidence=additions, admitted=False)))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
