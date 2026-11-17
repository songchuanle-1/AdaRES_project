"""Close a falsified forward premise and register a distinct 3D-support diagnosis."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
from register_y53_padding_diagnosis_fde0 import ref, TOKEN
import foreground_tick as tick
import pipeline_monitor as monitor


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds() > 500
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        y53 = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y53')
        assert y53['stage'] == 'validation' and not any(c['id'] == 'AdaRES-Y54' for c in work['candidates'])
        for row in y53['evidence']:
            assert ref(row['path']) == row
        native_path = 'pipeline/analyses/y53_padding_native_eval_compat_fde0_' + TOKEN + '.json'
        component_path = 'pipeline/analyses/y53_padding_components_fde0_' + TOKEN + '.json'
        native, component = (tick.load(ROOT / path) for path in (native_path, component_path))
        assert native['complete'] and native['fixed_TRAIN_scenes'] == 8 and native['expressions'] == 32
        assert component['complete']
        for row in native['records']:
            assert row['actual_fps_and_decoder_inputs'] and row['native_sparse_forward']
            assert all(value == 0 for key, value in row['comparisons']['padding_only'].items() if key != 'layers')
            assert all(value == 0 for key, value in row['comparisons']['repeat'].items() if key != 'layers')
            assert row['language_gradient']['padding_max_abs'] == 0 and row['language_gradient']['valid_max_abs'] > 0
            assert row['comparisons']['valid_word_positive_control']['max_indicator_logit_abs'] > 0
        review_path = 'pipeline/analyses/y53_padding_falsification_' + TOKEN + '.json'
        assert not (ROOT / review_path).exists()
        review = dict(complete=True, id='AdaRES-Y53', token=TOKEN, reviewed_at=datetime.now().astimezone().isoformat(),
            component=ref(component_path), native=ref(native_path),
            result='Forward padding-leakage premise falsified in the byte-bound native decoder source and8actual TRAIN scenes/32expressions: all7layer padding/repeat output deltas0 and padding gradient0; valid-word controls/gradients active in every scene.',
            causal_source_trace=['SamplingModule normalizes over queries then removes padded-word contributions before summation.', 'Native language self attention, instance-to-language, query-to-language attention exclude language padding as keys; padding query rows are never consumed as unmasked keys by the indicator path.', 'Contrastive projected token outputs are a separate loss path, already historical Y23; not reopened here.'],
            failure_scope='hypothesis', model_files_changed=0,
            limitation='Not a claim that every possible padding bug is absent, nor a trained efficacy result. Eval invocation failure is preserved; the compatibility repair changed no model bytes.')
        tick.save(ROOT / review_path, review)
        for path in (native_path, component_path, review_path, 'scripts/close_y53_register_y54_fde0.py'):
            if path not in {e['path'] for e in y53['evidence']}:
                y53['evidence'].append(ref(path))
        y53['stage'] = 'rejected'
        y53['failure_scope'] = 'hypothesis'
        y53['rejection_reason'] = review['result']
        y53['next_action'] = 'Preserve actual negative/positive-control evidence; do not implement a language-padding forward variant or reopen historical Y23. Only genuinely new disconfirming source/data evidence would motivate a separate authorized review.'
        y53['gate_summary']['diagnosis'] = 'falsified_by_actual_source_and_native_matched_controls'
        y53['gate_summary']['controls'] = 'padding0/repeat0/valid_positive; native32expressions'
        y53['diagnostic_revision_history'][-1]['result'] = 'Split eval call succeeded; unchanged native diagnostic completed with zero padding effect and active valid-word controls.'
        y53['diagnostic_revision_history'][-1]['result_evidence'] = [ref(native_path)]

        source_path = 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
        source = (ROOT / source_path).read_text(encoding='utf-8')
        assert 'fine_query, _, _ = self.qia_layers_fine[i](inst_feats_fine, coarse_query_ffn, attn_mask=~scg_mask, att_weights = scg_weights)' in source
        short = [dict(scene=row['scene'], superpoints=row['superpoints_per_expression'][0]) for row in native['records'] if row['superpoints_per_expression'][0] < 512]
        assert len(short) == 2
        plan_path = 'pipeline/analyses/y54_fine_support_diagnosis_prereg_' + TOKEN + '.json'
        assert not (ROOT / plan_path).exists()
        hypothesis = 'In native two-scene TRAIN batches, fine 3D attention may include nonexistent padded superpoints when a scene has fewer than the fixed512fine supports. Fine-source gathering loses the scene validity mask and spatial nearest-neighbor selection can treat padded coordinates as real. This could contaminate query confidence with batch-composition nuisance; diagnose actual native attention/support incidence before proposing a correction.'
        control = 'Use the SAME eight frozen TRAIN scenes in their original four batchsize2 pairs, fixed trained W2 diagnostic weights and seed1999. Preserve actual sparse forward/FPS/coarse ranking. Derive and verify exact gathered support validity from actual coarse attention and native positions. Compare parent/repeat, padding-only source intervention and valid-source control; compare native fine block against correctly masked support plus nearest-neighbor masking. All-valid expressions are negative controls. No task-gain threshold or validation selection.'
        plan = dict(complete=True, id='AdaRES-Y54', token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(),
            hypothesis_id='fine_3d_support_validity_batch_nuisance', hypothesis=hypothesis, matched_control=control,
            source=ref(source_path), shape_intake=ref(native_path), native_short_scene_shapes=short,
            first_test='Instrument actual four native batchsize2 sparse/FPS forwards and replay SAME cached decoder inputs; verify gathered indices/positions, count padded fine supports and unmasked/positive attention mass, execute matched original/repeat/corrected support controls and report final/aux confidence changes.',
            expected_direction='A real missing-support mechanism requires actual padded supports to be eligible/attended and a matched control-separated fine/query effect, with corrected padding interventions inert. Zero incidence or no separated effect remains inconclusive/negative, never an admission PASS.',
            independence='Different variable/path from Y53 and Y23: nonexistent 3D fine supports under ragged scene batching, not language features or contrastive-token normalization. Historical Y26 discarded-postnorm and all other rejected directions remain untouched.',
            limits='Diagnostic only; no model implementation, novel claim, accuracy benefit, optimizer steps, commit, admission or launch yet. Preserve512fine/128query/256FPS, current weights/loss/epochs and all source hashes.', result='pending')
        tick.save(ROOT / plan_path, plan)
        work['candidates'].append(dict(id='AdaRES-Y54', stage='diagnosis', hypothesis=hypothesis,
            hypothesis_id=plan['hypothesis_id'], matched_control=control, next_action=plan['first_test'],
            evidence=[ref(plan_path), ref(native_path), ref('scripts/close_y53_register_y54_fde0.py')],
            failure_scope='inconclusive', created_at=plan['preregistered_at'],
            gate_summary=dict(diagnosis='native_missing_support_suspected_not_yet_measured', controls='pending_native_matched', native_backward='not_run', resources='not_run', independence='not_implemented', constraints='diagnosis_only_no_admission')))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(y53_disposition='rejected_hypothesis', y53_review=ref(review_path), y54_stage='diagnosis', y54_prereg=ref(plan_path), shape_evidence=short, no_model_changes=True), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
