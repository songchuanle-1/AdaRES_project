"""Register a decoder-path diagnosis before reviewing or changing model code."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'fde01739a9864bae9bb342a577235e66'
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor


def ref(relative):
    path = (ROOT / relative).resolve()
    assert ROOT in path.parents and path.is_file()
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 300
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
        view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
        assert view['design_bootstrap']['required'] and view['research']['due_count'] == 0
        assert view['research']['reserve_budget'] == 1
        work = tick.load(ROOT / 'pipeline/research_work.json')
        assert not any(c['id'] == 'AdaRES-Y53' for c in work['candidates'])
        sources = ['AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/' + name for name in ('dec.py', 'ipdn.py', 'loss.py')]
        plan_path = 'pipeline/analyses/y53_padding_diagnosis_prereg_' + TOKEN + '.json'
        assert not (ROOT / plan_path).exists()
        hypothesis = 'Masked language-padding feature values may leak through the native decoder into query confidence, although they represent no words. Test actual decoder attention/aggregation, not the semantic-alignment loss denominator addressed by historical Y23. If all native confidence paths mask these positions and matched interventions have no effect beyond repeat noise, reject THIS leakage premise without creating a model variant.'
        controls = 'Keep actual class, parameters, valid-language positions, 3D/query inputs and RNG fixed; compare padding-only feature replacement, an unchanged independent repeat, and a valid-token positive-control replacement. Audit every native path consuming lang_masks. Run exact native module controls first and a frozen TRAIN native-model/decoder-input capture if a task-relevant uncertainty remains. Do not tune mask thresholds, weights, temperature, sample or training epochs.'
        plan = dict(complete=True, id='AdaRES-Y53', token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(),
            hypothesis_id='decoder_padding_to_query_confidence_leak', hypothesis=hypothesis, matched_control=controls,
            source_bindings=[ref(path) for path in sources],
            historical_distinction='Y23 complete hypothesis semantic-alignment-padding-denominator-leak concerns contrastive loss support; this candidate concerns forward decoder/query-confidence invariance. Do not reopen Y23 or reuse its training result as new evidence.',
            first_test='Inspect byte-bound native decoder paths and execute actual attention/aggregation module controls with padding-only, unchanged-repeat and valid-token perturbations; record nonzero graph gradients or exact masked invariance, not a synthetic efficacy PASS.',
            falsification='Source trace plus controls showing padding positions excluded from all relevant native forward paths falsify the proposed missing-padding mechanism; positive control must demonstrate sensitivity to valid words. Otherwise continue diagnosis, never label absent tests as PASS.',
            limits='No model/config change, validation or accuracy claim, training/Git commit/admission, old-candidate reopen or new scheduler. A component screen does not satisfy full native/resource gates.', result='pending')
        tick.save(ROOT / plan_path, plan)
        candidate = dict(id='AdaRES-Y53', stage='diagnosis', hypothesis=hypothesis,
            hypothesis_id=plan['hypothesis_id'], matched_control=controls,
            next_action=plan['first_test'], evidence=[ref(plan_path), ref('scripts/register_y53_padding_diagnosis_fde0.py')],
            failure_scope='inconclusive', gate_summary=dict(diagnosis='pending_native_path_audit_and_controls', controls='pending', native_backward='not_run', resources='not_run', independence='not_implemented', constraints='diagnosis_only_no_admission'),
            created_at=plan['preregistered_at'])
        work['candidates'].append(candidate)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(candidate_id=candidate['id'], stage=candidate['stage'], preregistration=ref(plan_path), new_model_files=0, training_launches=0), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
