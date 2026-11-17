"""Freeze one semantics-preserving resource revision before source changes."""

from datetime import datetime
import subprocess
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    assert not candidate.get('revision_history')
    for stem in ('cpu', 'native_gpu1', 'resources_gpu1'):
        bind(candidate, ref('pipeline/analyses/y56_' + stem + '_' + TOKEN + '.json'),
             ref('pipeline/analyses/y56_' + stem + '_' + TOKEN + '.log'))
    structural = subprocess.run(['python', '-B', str(ROOT / 'scripts/test_y56_prototype_dd8e.py')],
                                capture_output=True, text=True, check=True)
    structural_ref = save('pipeline/analyses/y56_structure_initial_' + TOKEN + '.json', dict(
        complete=True, tests=5, passed=True, output=structural.stdout + structural.stderr,
        tested_bundle=candidate['tested_bundle'], checked_at=datetime.now().astimezone().isoformat()))
    resources = tick.load(ROOT / ('pipeline/analyses/y56_resources_gpu1_' + TOKEN + '.json'))
    parent = {r['scene']: r['median_seconds'] for r in resources['paired_native_predict_profile'] if r['role'] == 'parent'}
    ratios = [r['median_seconds'] / parent[r['scene']] for r in resources['paired_native_predict_profile']
              if r['role'] == 'candidate']
    failure = save('pipeline/analyses/y56_resource_review_initial_' + TOKEN + '.json', dict(
        complete=True, failure_scope='implementation', resource_gate_passed=False,
        candidate_to_parent_latency_ratios=ratios,
        reason='Joint 128x(d+valid_support) exact unique runs even when raw queries are all distinct; '
               '1.73-2.94x native inference cost is not accepted as lightweight. No cutoff is retuned.',
        result=ref('pipeline/analyses/y56_resources_gpu1_' + TOKEN + '.json'),
        preserved_bundle=candidate['tested_bundle'], parameters_delta=0,
        scope='Runtime implementation, not hypothesis falsification; preserve all controls/native results.'))
    stamp = datetime.now().astimezone().isoformat()
    plan = save('pipeline/analyses/y56_revision1_prereg_' + TOKEN + '.json', dict(
        complete=True, id='AdaRES-Y56', revision=1, preregistered_at=stamp,
        reason='Avoid high-dimensional exact uniqueness on a provably injective raw-query projection.',
        change='First exact-unique raw query rows; if all distinct return original row graph without '
               'reading mask class keys; otherwise run unchanged full joint equivalence implementation.',
        prediction='Identical class membership/output and gradients for all former tests and exact '
                   'CPU original-implementation reference; paired native latency falls toward parent.',
        tests='Repeat frozen CPU/native/resources and structural suite, same actual10TRAIN scenes. '
              'Add bitwise CPU comparison with initial implementation, for both copy and no-copy inputs.',
        failed_bundle=candidate['tested_bundle'], failure=failure,
        no_new_hyperparameters=True, no_threshold_change=True, result='pending'))
    candidate['revision_history'] = [dict(revision=1, failure_scope='implementation',
        reason='High-dimensional exact grouping dominated runtime even for unique queries.',
        change='Provably exact raw-query injectivity prefilter, full joint grouping otherwise.',
        prediction='All outputs/gradients unchanged; reduce measured runtime overhead.',
        preregistered_at=stamp, failure_evidence=[failure, ref('pipeline/analyses/y56_resources_gpu1_' + TOKEN + '.json')],
        result='pending', evidence=[plan], result_evidence=[])]
    bind(candidate, structural_ref, failure, plan, ref(__file__))
    candidate.update(stage='implementation', failure_scope='implementation',
        next_action='Apply SAME Y56 preregistered exact-query prefilter then rerun all affected CPU/native/resource gates; '
                    'preserve initial overcost. No commit, admission or changed scientific threshold.')
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print('Registered SAME Y56 resource revision1 before any source edit.')
finally:
    lock.close()
