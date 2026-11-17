"""Freeze a task-native check after the exact decoder component screen."""
import ast
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
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y53')
        assert candidate['stage'] == 'diagnosis'
        for row in candidate['evidence']:
            assert ref(row['path']) == row
        component_path = 'pipeline/analyses/y53_padding_components_fde0_' + TOKEN + '.json'
        component = tick.load(ROOT / component_path)
        assert component['complete'] and component['indicator_gradient_wrt_language']['padding_max_abs'] == 0
        assert component['comparisons']['valid_token_positive_control']['max_indicator_logit_abs'] > 0
        prior = 'pipeline/analyses/y52_native_matched_capture_binding_v2_89ffd8db36cf43ed8cfd75a68c77b4b9.py'
        tree = ast.parse((ROOT / prior).read_text(encoding='utf-8'))
        old = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in node.targets)))
        payload = dict(parent_files=old['parent_files'], capture_sha256=old['capture_sha256'],
            config_sha256='d4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb',
            checkpoint_sha256='e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52')
        body = 'scripts/probe_y53_padding_native_fde0.py'
        source = 'PAYLOAD=' + repr(payload) + '\n' + (ROOT / body).read_text(encoding='utf-8')
        ast.parse(source, feature_version=(3, 8))
        program = 'pipeline/analyses/y53_padding_native_program_' + TOKEN + '.py'
        plan_path = 'pipeline/analyses/y53_padding_native_prereg_' + TOKEN + '.json'
        assert not (ROOT / program).exists() and not (ROOT / plan_path).exists()
        plan = dict(complete=True, id='AdaRES-Y53', token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(), component_result=ref(component_path), body=ref(body),
            fixed_protocol='SAME previously frozen8TRAIN scenes/32expressions, seed1999 per dataset access; actual sparse forward/FPS with fixed trained W2 checkpoint. Native predict batchsize1. Cache actual decoder inputs, freeze weights, compare unchanged repeat vs padding-only and valid-word feature sign flips at SAME input; all7layer indicators/masks/scores/sampler/argmax and gradient w.r.t.language.',
            prediction='If padding is excluded, actual query-confidence outputs and gradients show no padding effect, while valid-word positive control is active. Preserve nonzero repeats if observed; no invented numeric tolerance or validation tuning.',
            limits='No model/config changes, optimizer steps, weights saved, Git commit or admission. This does not reopen Y23 or retest blocked Y52; common fixed TRAIN data are reused for a distinct decoder-forward hypothesis.', result='pending')
        (ROOT / program).write_text(source, encoding='utf-8', newline='\n')
        plan['program'] = ref(program)
        tick.save(ROOT / plan_path, plan)
        for path in (component_path, body, 'scripts/prepare_y53_padding_native_fde0.py', program, plan_path):
            candidate['evidence'].append(ref(path))
        candidate['stage'] = 'validation'
        candidate['next_action'] = 'Execute the frozen native sparse/FPS TRAIN decoder capture and SAME-input padding/repeat/valid controls under a fresh exclusive authorized GPU lease; preserve independent Y23 and Y52 scopes.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(program=ref(program), plan=ref(plan_path)), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
