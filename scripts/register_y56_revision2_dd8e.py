"""Preregister the last default-budget exact grouping cost revision."""

from datetime import datetime
import statistics
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    first = candidate['revision_history'][-1]
    assert len(candidate['revision_history']) == 1 and first['result'] == 'pending'
    outputs = [ref('pipeline/analyses/' + name + '_' + TOKEN + '.json') for name in (
        'y56_cpu_rev1_v2', 'y56_native_rev1_gpu1', 'y56_resources_rev1_gpu1')]
    for row in outputs:
        bind(candidate, row)
        first['evidence'].append(row)
    first['result_evidence'] = outputs
    first['result'] = 'inconclusive'
    resources = tick.load(ROOT / outputs[-1]['path'])
    parent = {r['scene']: r['median_seconds'] for r in resources['paired_native_predict_profile'] if r['role'] == 'parent'}
    ratios = [r['median_seconds'] / parent[r['scene']] for r in resources['paired_native_predict_profile'] if r['role'] == 'candidate']
    review = save('pipeline/analyses/y56_resource_review_rev1_' + TOKEN + '.json', dict(
        complete=True, revision=1, CPU_equivalence_passed=True, native_backward_passed=True,
        native_predict_latency_ratios=ratios, median_ratio=statistics.median(ratios),
        resource_gate_passed=False, failure_scope='implementation',
        reason='Per-expression raw-query unique still sorts, and any duplicate raw query triggers '
               'full high-dimensional grouping of all128 rows. A semantics-equivalent sparse check is available.',
        source_bundle=candidate['tested_bundle'], evidence=outputs,
        interpretation='Revision1 improved runtime without resolving cost concern. Not a failed hypothesis or numerical gate relaxation.'))
    first['evidence'].append(review)
    first['result_evidence'].append(review)
    stamp = datetime.now().astimezone().isoformat()
    plan = save('pipeline/analyses/y56_revision2_prereg_' + TOKEN + '.json', dict(
        complete=True, id='AdaRES-Y56', revision=2, preregistered_at=stamp,
        failed_bundle=candidate['tested_bundle'], failure=review,
        change='Batch exact raw-query equality; transfer per-expression duplicate flags once. '
               'Only query-nonunique rows enter exact joint unique; singleton-query rows retain unique row IDs. '
               'Join counts and class IDs without changing graph/weights. No hash approximation or rounding.',
        prediction='Output and input/parameter gradients equal initial Y56, with lower native paired cost. '
                   'No-copy graph remains bitwise W2. Additional transient boolean query-pair memory is measured.',
        tests='All original CPU/native/resource/structural controls, unchanged10TRAIN scenes and seed. '
              'CPU equality to immutable initial source. No new scientific thresholds, parameter changes or validation selection.',
        result='pending', default_revision_budget_consumed=2))
    candidate['revision_history'].append(dict(revision=2, failure_scope='implementation',
        reason='Avoid per-expression sorting and grouping singleton-query rows in high-dimensional keys.',
        change='Batched exact-query equality and joint grouping only on query-nonunique rows.',
        prediction='Identical semantics/output/gradients, lower paired native cost with measured transient memory.',
        preregistered_at=stamp, failure_evidence=[review, outputs[-1]],
        result='pending', evidence=[plan], result_evidence=[]))
    bind(candidate, review, plan, ref(__file__))
    candidate['next_action'] = 'Execute last default-budget exact grouping revision2, retain prior runtime costs and driver failure; no new route or threshold tuning.'
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print('Frozen final default-budget Y56 revision2 before production edit.')
finally:
    lock.close()
