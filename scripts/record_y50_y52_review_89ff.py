"""Persist the unchanged frozen comparison and bounded native Y52 gate review."""
import ast
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
from prepare_y52_native_backward_89ff import ref, TOKEN
from analyze_frozen_full import comparison
import foreground_tick as tick
import pipeline_monitor as monitor


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 90
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidates = {c['id']: c for c in work['candidates']}
        y50, y52 = candidates['AdaRES-Y50'], candidates['AdaRES-Y52']
        for candidate in (y50, y52):
            for row in candidate['evidence']:
                assert ref(row['path']) == row
        capture_paths = {key: 'pipeline/analyses/y50_frozen_' + key + '_89ff_' + TOKEN + '.json' for key in ('y50', 'w2', 'b1')}
        wrappers = {key: tick.load(ROOT / path) for key, path in capture_paths.items()}
        captures = {key: value['rows'] for key, value in wrappers.items()}
        for capture in captures.values():
            assert capture['complete'] and capture['completed_passes'] == 2
            assert all(p['summary']['expressions'] == 9508 and p['summary']['scenes'] == 141 for p in capture['passes'])
        comparisons = {key: comparison(captures['y50'], captures[key], seed=1999, draws=10000) for key in ('w2', 'b1')}
        harm = []
        for key, groups in comparisons.items():
            for group, row in groups.items():
                row['frozen_harm_trigger_ci_upper_lt_zero'] = row.get('scene_clustered_ci95', [0, 0])[1] < 0
                if group != 'all' and row['frozen_harm_trigger_ci_upper_lt_zero']:
                    harm.append(dict(reference=key, group=group, result=row))
        analysis_path = 'pipeline/analyses/y50_frozen_comparison_' + TOKEN + '.json'
        assert not (ROOT / analysis_path).exists()
        analysis = dict(complete=True, id='AdaRES-Y50', token=TOKEN,
            analyzed_at=datetime.now().astimezone().isoformat(), sources={key: ref(path) for key, path in capture_paths.items()},
            analysis_implementation=ref('scripts/analyze_frozen_full.py'), seed=1999, bootstrap_draws=10000,
            protocol='Full9508expressions/141scenes twice EACH; FIRST primary; scene-clustered expression-weighted bootstrap; frozen TRAIN cuts1487/3325; subgroup CI upper<0 triggers harm review.',
            summary={key: value['passes'][0]['summary'] for key, value in captures.items()},
            comparisons=comparisons, material_harm_review=harm, target=0.515,
            target_reached=captures['y50']['passes'][0]['summary']['miou'] >= 0.515,
            cost={key: dict(parameters=value['parameters'], warm_first_chunk_median_seconds=statistics.median(value['warm_first_chunk_seconds']), peak_allocated_gib=value['peak_allocated_gib'], peak_reserved_gib=value['peak_reserved_gib'], active_queries=value['active_queries'], selected_queries=value['selected_queries']) for key, value in captures.items()},
            decision='Do not promote or requeue this fixed30 realization: medium-size harm versus W2 triggers the frozen review. Overall intervals cross zero; no robust baseline gain and no universal mechanism falsification.',
            limitations=['One training seed/checkpoint per method; bootstrap does not estimate seed uncertainty.', 'Repeats retain large individual-expression changes; group-mean repeat noise is not a universal bound.', 'Timing is five warm first-chunk forwards, not full-dataset latency.', 'Historical Y36 champion metrics belong to a separate evaluation and are not recomputed here.'])
        assert harm, 'No harm observation: review the disposition, do not force rejection'
        tick.save(ROOT / analysis_path, analysis)
        for path in list(capture_paths.values()) + [analysis_path, 'scripts/record_y50_y52_review_89ff.py']:
            if path not in {e['path'] for e in y50['evidence']}:
                y50['evidence'].append(ref(path))
        y50['stage'] = 'rejected'
        y50['failure_scope'] = 'implementation'
        y50['scientific_outcome'] = ref(analysis_path)
        y50['rejection_reason'] = analysis['decision']
        y50['next_action'] = 'Preserve fixed30 terminal/capture/harm evidence; no automatic rerun, epoch/threshold tuning or promotion. Canonical registry deathwatch1of3 remains pending; reconcile only in a later normal preflight, not by rewriting queue/history now.'

        resource_path = 'pipeline/analyses/y52_resources_batch_compat_89ff_' + TOKEN + '.json'
        readout_path = 'pipeline/analyses/y52_predict_readout_89ff_' + TOKEN + '.json'
        backward_path = 'pipeline/analyses/y52_backward_scalar_compat_89ff_' + TOKEN + '.json'
        resource, readout, backward = (tick.load(ROOT / p) for p in (resource_path, readout_path, backward_path))
        assert resource['complete'] and readout['complete'] and backward['complete']
        assert readout['diagnostic_state'] == 'fixed_trained_W2_not_trained_Y52'
        assert resource['parameter_delta'] == readout['parameter_delta'] == 0
        assert all(row['finite_backward'] and row['peak_reserved_bytes'] < 24 * 2**30 for value in (resource, readout) for row in value['stress'])
        assert all(backward['nonempty_optimizer_restore'].values()) and len(backward['optimizer_steps']) == 2
        trained = [row for row in backward['rows'] if row['state'] == 'trained_parent_diagnostic_only']
        assert len(trained) == 4 and all(row['native_backward_finite'] and row['full_query_gradient_signal']['max_abs'] > row['full_query_repeat_noise']['max_abs'] for row in trained)
        behavioral = []
        for row in readout['inference_parity']:
            repeats = row['parent_repeat'] + row['candidate_repeat']
            keys = ('binary_mask_points_changed', 'selected_query_changes', 'maximum_expression_iou_difference')
            behavioral.append(dict(batch=row['batch'], matched=row['matched'], repeat_max={key: max(v[key] for v in repeats) for key in keys}, within_measured_behavioral_repeats=all(row['matched'][key] <= max(v[key] for v in repeats) for key in keys)))
        parity_supported = all(row['within_measured_behavioral_repeats'] for row in behavioral)
        bundle_path = 'pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json'
        bundle = tick.load(ROOT / bundle_path)
        for entry in bundle['files']:
            assert ref(entry['path']) == entry
            if entry['path'].endswith('.py'):
                tree = ast.parse((ROOT / entry['path']).read_text(encoding='utf-8'))
                modules = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
                assert not any('adares_' in name and 'adares_y52_point_target' not in name for name in modules)
        paths = [entry['path'].removeprefix('AdaRES/') for entry in bundle['files']]
        tracked = subprocess.check_output(['git', '-C', str(ROOT / 'AdaRES'), 'ls-files', '--'] + paths, text=True).strip()
        assert not tracked, 'Candidate now tracked: re-evaluate human commit blocker'
        authorization = 'pipeline/authorizations/AdaRES-Y50_commit.json'
        auth = tick.load(ROOT / authorization)
        assert auth['id'] == 'AdaRES-Y50' and not (ROOT / 'pipeline/authorizations/AdaRES-Y52_commit.json').exists()
        review_path = 'pipeline/analyses/y52_native_gate_review_' + TOKEN + '.json'
        assert not (ROOT / review_path).exists()
        review = dict(complete=True, id='AdaRES-Y52', token=TOKEN, reviewed_at=datetime.now().astimezone().isoformat(),
            tested_bundle=ref(bundle_path), native_backward=ref(backward_path), native_resources=ref(resource_path), trained_parent_predict=ref(readout_path),
            parent_initial_zero_activity_preserved=True, trained_parent_gradient_batches=4,
            max_shared_query_gradient_delta=max(row['full_query_gradient_signal']['max_abs'] for row in trained),
            nonempty_state_restore=backward['nonempty_optimizer_restore'], parameter_delta=0,
            training_peak_reserved_bytes=max(backward['total_peak_reserved_bytes'], max(row['peak_reserved_bytes'] for row in resource['stress'])),
            trained_parent_behavior=behavioral, inference_within_measured_behavioral_repeats=parity_supported,
            uncommitted_seven_file_bundle=True, existing_authorization_Y50_only=ref(authorization),
            gate_scope='Native task-relevant target/gradient intervention, finite train graph and bounded resource/restore controls; no new trainable tensors. Released-init score inactivity retained, fixed trained-parent activation not represented as trained Y52 benefit.',
            limitations='No trained Y52 accuracy or global Pareto guarantee. Float probability counts alone are not binary-mask changes. Exact commit binding and admission receipt remain absent; no automatic Git commit.',
            disposition='Await explicit human scoped commit permission' if parity_supported else 'Inference behavior remains inconclusive; continue SAME candidate diagnostic before commit request')
        tick.save(ROOT / review_path, review)
        for path in (resource_path, readout_path, backward_path, review_path, 'scripts/record_y50_y52_review_89ff.py'):
            if path not in {e['path'] for e in y52['evidence']}:
                y52['evidence'].append(ref(path))
        for revision in y52.get('diagnostic_revision_history', []):
            if revision['scope'] == 'inference_batch_compatibility_not_model':
                revision['result'] = 'Native batchsize1 predict and unchanged batchsize2 finite resource stress completed; floating raw differences retained, supplemented by fixed-trained-parent behavioral repeat readout.'
                revision['result_evidence'] = [ref(resource_path), ref(readout_path)]
        y52['failure_scope'] = 'inconclusive'
        y52['gate_summary'] = dict(diagnosis='native_target_measure_difference_observed; direct_indicator_quality_premise_rejected', controls='same_native_matches_parent_corrected_repeat; CPU_negative_controls_retained', native_backward='finite_and_non_noop_in_fixed_trained_parent; released_init_zero_activity_retained', resources='parameter_delta0; finite_native_batch2_stress; bounded24GiB; exact_nonempty_restore', independence='seven_byte_bound_independent_files; actual_same_state_construction; no_cross_variant_import', constraints='fixed_protocol_design_frozen; no_validation_tuning; commit_and_receipt_pending', inference='within_measured_behavioral_repeats' if parity_supported else 'inconclusive')
        if parity_supported:
            y52['stage'] = 'blocked'
            y52['blocker_kind'] = 'human_authorization'
            y52['blocker'] = 'All currently planned native diagnostics completed within documented scope. Exact seven tested Y52 files remain uncommitted; canonical closure forbids automatic Git add/commit. Existing explicit authorization is Y50-only, not Y52; GPU availability is NOT the blocker.'
            y52['retry_after'] = (datetime.now().astimezone() + timedelta(hours=4)).isoformat()
            y52['unblock_condition'] = 'Human explicitly authorizes only the seven exact tested Y52 files, or supplies their exact commit; then revalidate hashes/current goal and six gates, issue bound receipt and use canonical dispatcher with fresh idle checks. Do not deploy diagnostic W2 weights.'
            y52['next_action'] = y52['unblock_condition']
            y52['wake_paths'] = ['pipeline/authorizations/AdaRES-Y52_commit.json', bundle_path]
            work['design_cycle'] = dict(stop_reason='external_resource_blocked', attempted_ids=['AdaRES-Y50', 'AdaRES-Y52'], evidence=[ref(analysis_path), ref(review_path)], resume_condition=y52['unblock_condition'], clarification='External mandatory human commit permission, not a GPU/resource shortage or scientific PASS; bounded SAME candidate preparation completed.')
        else:
            y52['stage'] = 'validation'
            y52['next_action'] = 'Preserve fixed-trained-parent parity discrepancies and repeats; inspect affected original inputs/selector without model or threshold tuning. No commit authorization request or admission until explained.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(y50_analysis=ref(analysis_path), y50_summary=analysis['summary'], y50_harm=harm, y52_review=ref(review_path), y52_behavior=behavioral, y52_stage=y52['stage'], y50_evidence=len(y50['evidence']), y52_evidence=len(y52['evidence']), queue_unchanged=True), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
