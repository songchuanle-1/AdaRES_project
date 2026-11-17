"""Review actual outputs and preserve an exact local-commit handoff."""

from datetime import datetime, timedelta
import statistics
import subprocess
import json
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor
import closure

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    assert len(candidate['revision_history']) == 2 and candidate['revision_history'][-1]['result'] == 'pending'
    bundle = tick.load(ROOT / candidate['tested_bundle']['path'])
    assert all(ref(r['path']) == r for r in bundle['files'])
    output_names = ['y56_cpu_rev2', 'y56_native_rev2_gpu1', 'y56_resources_rev2_gpu1',
                    'y56_gradient_properties', 'y56_margin_timing_v2_gpu1']
    evidence = [ref('pipeline/analyses/' + n + '_' + TOKEN + '.json') for n in output_names]
    cpu, native, resources, properties, audit = [tick.load(ROOT / r['path']) for r in evidence]
    assert all(r['complete'] for r in (cpu, native, resources, properties, audit))
    assert properties['cases'] == 18 and all(r['input_and_every_parameter_gradients_bitwise_equal'] for r in properties['rows'])
    assert all(r['no_copy_bitwise_equal'] and r['initial_revision_output_and_input_gradients_bitwise_equal'] for r in cpu['rows'])
    assert native['native_backward'] and len(native['rows']) == 15 and all(r['finite_backward'] for r in native['rows'])
    formal = next(r for r in audit['rows'] if r['regime'] == 'released_SPFormer_formal_init')
    assert formal['layers'][0]['maximum_confidence_delta'] > formal['layers'][0]['probability_arithmetic_bound']
    assert formal['layers'][0]['native_task_first_order_delta'] != 0
    assert resources['parameter_delta'] == 0 and all(r['peak_reserved_bytes'] < 24*2**30 for r in resources['stress'])
    measured = [r for r in audit['cases'] if not r['warmup']]
    profiles = []
    for scene in dict.fromkeys(r['scene'] for r in measured):
        values = {role: [r['seconds'] for r in measured if r['scene'] == scene and r['role'] == role]
                  for role in ('parent', 'candidate')}
        assert all(len(v) == 5 for v in values.values())
        medians = {role: statistics.median(v) for role, v in values.items()}
        profiles.append(dict(scene=scene, **medians, ratio=medians['candidate']/medians['parent']))
    stress_medians = {role: statistics.median(r['seconds'] for r in resources['stress']
        if r['role'] == role and not r['warmup']) for role in ('parent', 'candidate')}
    test = subprocess.run(['python', '-B', str(ROOT/'scripts/test_y56_prototype_rev2_dd8e.py')],
                          capture_output=True, text=True, check=True)
    structural = save('pipeline/analyses/y56_structure_rev2_' + TOKEN + '.json', dict(
        complete=True, tests=5, passed=True, output=test.stdout+test.stderr,
        tested_bundle=candidate['tested_bundle'], checked_at=datetime.now().astimezone().isoformat()))
    gates = dict(
        diagnosis=dict(passed=True, evidence=[evidence[4], ref('pipeline/analyses/y56_actual_transfer_6debf88fed634a13b772f27bdc244bbd.json')],
                       scope='Real exact-copy TRAIN mechanism; conditioned discovery, not population harm.'),
        controls=dict(passed=True, evidence=[evidence[0], evidence[1], evidence[3]],
                      scope='Focal-copy/unique-set invariance; raw equality negatives; no-copy W2 bitwise equality;18edge gradient properties.'),
        native_backward=dict(passed=True, evidence=[evidence[1], evidence[4]],
                             scope='Actual released-init loss/all7sites/every indicator tensor; natural initial copy affects confidence and task gradient.'),
        resources=dict(passed=True, evidence=[evidence[2], evidence[4]],
                       scope='Bounded pre-training feasibility/compatible cost, not proven post-training global Pareto. '
                             'One conditioned duplicate scene remains slower; no cost or efficacy threshold relaxed.'),
        independence=dict(passed=True, evidence=[structural, candidate['tested_bundle']],
                          scope='Physical six-file package, unique YAML, no cross-variant imports; all non-indicator AST unchanged.'),
        constraints=dict(passed=True, evidence=[structural, candidate['tested_bundle'],
                         ref('pipeline/designs/exp_design_Y56_dd8e.md')],
                         scope='Same SPFormer/30epochs/1999/Q128/loss/optimizer/singleargmax; server2GPU0/1 only. No commit/queue/launch.'))
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == '8943c8c460501b200a8dea958964579ffea65f76'
    auth = ROOT / 'pipeline/authorizations/AdaRES-Y56_commit.json'
    assert not auth.exists(), 'Review any supplied authorization; do not overwrite or ignore it'
    review = save('pipeline/analyses/y56_gate_review_' + TOKEN + '.json', dict(
        complete=True, id='AdaRES-Y56', revision=2, observed_at=datetime.now().astimezone().isoformat(),
        tested_bundle=candidate['tested_bundle'], files=bundle['files'], goal_sha256=ref('pipeline/goal.json')['sha256'],
        gates=gates, structural=structural, formal_margin_readout=formal,
        order_balanced_timing=profiles, median_paired_latency_ratio=statistics.median(r['ratio'] for r in profiles),
        maximum_paired_latency_ratio=max(r['ratio'] for r in profiles), stress_median_seconds=stress_medians,
        largest_stress_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resources['stress']),
        native_audit_peak_reserved_bytes=native['peak_reserved_bytes'],
        frozen_post_training_evaluation=ref('pipeline/designs/exp_design_Y56_dd8e.md'),
        local_HEAD=head, existing_commit_scopes=['AdaRES-Y50','AdaRES-Y52','AdaRES-Y54'],
        missing_operation='Only local git add/commit of the exact seven files in files; no push or unrelated changes.',
        scientific_prechecks_ready=True, admission_receipt_issued=False, local_commit_done=False,
        training_started=False, optimizer_steps=0, trained_efficacy='unknown', novelty='unverified',
        limitations=['Natural copies found only in one of40TRAIN scenes, initial layer; final selector changes not observed.',
                     'Formal native total loss delta below reported float32 loss resolution; nonzero directional readout is not benefit.',
                     'Single conditioned duplicate scene has higher latency; fixed10TRAIN timing is not a worst-case bound.',
                     'Default two implementation revisions consumed; original overcost and failed drivers preserved.']))
    revision = candidate['revision_history'][-1]
    revision['result'] = 'passed'
    revision['evidence'].extend([*evidence, structural, review])
    revision['result_evidence'] = [*evidence, structural, review]
    for name, row in zip(output_names, evidence):
        bind(candidate, row, ref('pipeline/analyses/'+name+'_'+TOKEN+'.log'))
    bind(candidate, structural, review, ref(__file__))
    now = datetime.now().astimezone()
    candidate.update(stage='blocked', blocker_kind='human_authorization',
        blocker='Scientific prechecks completed for exact Y56 revision2 bundle; no current user authorization covers '
                'local Git add/commit of these seven files. Existing exact scopes cover Y50/Y52/Y54, not Y56. '
                'This is NOT a GPU permission, mount, numerical-control or pipeline failure.',
        retry_after=(now+timedelta(hours=4)).isoformat(),
        unblock_condition='Owning user explicitly authorizes only the exact7tested Y56 files for local Git commit, '
                          'or supplies their exact commit. Revalidate hashes/current goal and scientific gates, '
                          'issue bound receipt and use existing authorized dispatcher with fresh idle/progress checks. '
                          'No push, other files/projects, trained-W2 initialization or schedule changes.',
        next_action='Ask once for exact seven-file Y56 local commit; record unambiguous owning-user confirmation '
                    'without magic wording. Then commit only tested bundle and revalidate admission/dispatch.',
        wake_paths=['pipeline/authorizations/AdaRES-Y56_commit.json'],
        scientific_status='pretraining_gates_reviewed_ready_pending_scoped_local_commit',
        gate_summary={name: gate['scope'] for name,gate in gates.items()})
    candidate['attempted_alternatives'] = candidate.get('attempted_alternatives', []) + [
        'Initial exact joint grouping: controls/native passed, costly1.7-2.9x native inference, retained.',
        'Revision1 exact query prefilter: same outputs/gradients; per-expression sorting cost remained, retained.',
        'Revision2 batch exact-query equality + subset joint grouping: all actual controls/native/resources '
        'and18edge gradient properties passed; no threshold/parameter/protocol change.',
        'Additive actual class-margin/task projection and order-balanced timing completed; no optimizer or source changes.']
    closure.validate_revisions(ROOT, candidate)
    work['design_cycle'] = dict(stop_reason='external_resource_blocked', attempted_ids=['AdaRES-Y56'],
        evidence=[review],
        resume_condition='Exact Y56 local-commit confirmation wakes its admission/deployment. Y57 remains a separate '
                         'unchanged evidenced numerical hold, not human/GPU permission. No claim of scientific route exhaustion.',
        detail='This bounded continuation completed the executable Y56 gates and two permitted local repairs. '
               'No new independent mechanism is claimed merely to consume an unused GPU slot.')
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(review=review, tested_bundle=candidate['tested_bundle'], stage=candidate['stage'],
        median_paired_latency_ratio=statistics.median(r['ratio'] for r in profiles),
        maximum_paired_latency_ratio=max(r['ratio'] for r in profiles), stress_medians=stress_medians,
        exact_commit_files=[r['path'] for r in bundle['files']], training_started=False), ensure_ascii=False, indent=2))
finally:
    lock.close()
