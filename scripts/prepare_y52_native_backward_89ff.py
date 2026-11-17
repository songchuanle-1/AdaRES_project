"""Freeze the paired native backward and complete nonempty-state restore test."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
TOKEN = '89ffd8db36cf43ed8cfd75a68c77b4b9'


def ref(relative):
    path = (ROOT / relative).resolve()
    assert ROOT in path.parents and path.is_file()
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None, 'Pipeline writer busy'
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 900
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
        assert candidate['stage'] == 'validation'
        for row in candidate['evidence']:
            assert ref(row['path']) == row, row['path']
        capture_path = 'pipeline/analyses/y52_native_capture_binding_v2_89ff_' + TOKEN + '.json'
        capture = tick.load(ROOT / capture_path)
        assert capture['complete'] and capture['actual_sparse_TRAIN_forward']
        assert capture['frozen_TRAIN_scenes'] == 8 and capture['expressions'] == 32
        bound_program = 'pipeline/analyses/y52_native_matched_capture_binding_v2_' + TOKEN + '.py'
        tree = ast.parse((ROOT / bound_program).read_text(encoding='utf-8'))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets))
        payload = ast.literal_eval(assignment.value)
        original_body = (ROOT / 'scripts/probe_y52_native_matched_capture.py').read_text(encoding='utf-8')
        setup = original_body.split('rows = []\n', 1)[0]
        assert setup.endswith('dataset = build_dataset(cfg.data.train.copy(), logger)\n')
        body_path = 'scripts/probe_y52_native_backward_89ff.py'
        body = (ROOT / body_path).read_text(encoding='utf-8')
        source = 'PAYLOAD=' + repr(payload) + '\n' + setup + '\n' + body
        program_path = 'pipeline/analyses/y52_native_backward_program_' + TOKEN + '.py'
        plan_path = 'pipeline/analyses/y52_native_backward_prereg_' + TOKEN + '.json'
        assert not (ROOT / program_path).exists() and not (ROOT / plan_path).exists()
        ast.parse(source, feature_version=(3, 8))
        compile(source, program_path, 'exec')
        plan = dict(complete=True, id='AdaRES-Y52', token=TOKEN,
            preregistered_at=datetime.now().astimezone().isoformat(),
            prior_capture=ref(capture_path), bound_setup=ref(bound_program), body=ref(body_path),
            fixed_sample='SAME eight frozen TRAIN scenes/32 expressions; actual batch_size2 and seed1999 per dataset access; no validation data or sample/threshold selection.',
            states=['released SPFormer with frozen initial candidate state',
                    'fixed trained W2 best e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52, diagnostic only'],
            mechanism_readout='On SAME native decoder forward, compare original parent Criterion and corrected Criterion on exactly equal real match indices, plus independent corrected repeat with restored RNG. Measure active >0.5 matches, native score losses, full/shared-query and score-only query-gradient deltas versus repeat noise; no fixed task gain threshold.',
            expected_direction='Where quality-target discrepancy is used by the existing score MSE, corrected-vs-parent gradients should differ beyond corrected-repeat noise. Zero activity or no separated signal is inconclusive, not PASS or mechanism rejection. Unexplained nonfinite/adverse results require investigation.',
            optimizer_restore='Reset exact released-init model state, then actual native-config AdamW/PolyLR two optimizer steps, clip0.1. Snapshot model/all optimizer/nonempty states/scheduler/RNG after first; restore after second and require exact equality. No tensors saved to disk.',
            limitations='No new parameters or model revision. Trained parent state is not a trained Y52 result. This is not inference cost/Pareto evidence, trained efficacy, commit authorization or admission.',
            result='pending')
        (ROOT / program_path).write_text(source, encoding='utf-8', newline='\n')
        plan['program'] = ref(program_path)
        tick.save(ROOT / plan_path, plan)
        for path in (capture_path, body_path, 'scripts/prepare_y52_native_backward_89ff.py', program_path, plan_path):
            if path not in {e['path'] for e in candidate['evidence']}:
                candidate['evidence'].append(ref(path))
        candidate['diagnostic_revision_history'][-1]['result'] = 'Strict source-binding repair accepted; same frozen native TRAIN capture completed. No model change.'
        candidate['diagnostic_revision_history'][-1].setdefault('result_evidence', []).append(ref(capture_path))
        candidate['next_action'] = 'Execute preregistered SAME-forward parent/corrected/repeat native gradients in released-init and fixed trained-parent diagnostic states, plus two native optimizer steps and complete nonempty-state restore. Then actual paired inference/resource gates. No admission or Y52 commit permission.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(program=ref(program_path), plan=ref(plan_path),
            native_backward_pending=True, candidate_model_revisions=0), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
