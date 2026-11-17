"""Freeze exact native sources and component readouts under the current lease."""
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
        assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 300
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y53')
        assert candidate['stage'] == 'diagnosis'
        for row in candidate['evidence']:
            assert ref(row['path']) == row
        prior = 'pipeline/analyses/y52_native_matched_capture_binding_v2_89ffd8db36cf43ed8cfd75a68c77b4b9.py'
        tree = ast.parse((ROOT / prior).read_text(encoding='utf-8'))
        payload = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in node.targets)))
        payload = dict(parent_files=payload['parent_files'], config_sha256='d4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb')
        body = 'scripts/probe_y53_padding_components_fde0.py'
        source = 'PAYLOAD=' + repr(payload) + '\n' + (ROOT / body).read_text(encoding='utf-8')
        ast.parse(source, feature_version=(3, 8))
        program = 'pipeline/analyses/y53_padding_components_program_' + TOKEN + '.py'
        plan_path = 'pipeline/analyses/y53_padding_components_prereg_' + TOKEN + '.json'
        assert not (ROOT / program).exists() and not (ROOT / plan_path).exists()
        plan = dict(complete=True, id='AdaRES-Y53', token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(),
            source_binding_intake=ref(prior), parent_files=payload['parent_files'], body=ref(body),
            test='Exact unchanged W2 DEC/native config on CPU fixtures. Freeze all parameters and all valid/3D/query inputs; change only padding features, compare independent unchanged repeat, and separately sign-flip valid words as positive control. Measure final/aux indicator/mask/score, sampler/topquery and autograd of all indicator logits w.r.t. valid vs padding language features.',
            prediction='Properly excluded padding gives no confidence/query gradient or output effect; valid-word control must have a nonzero effect. A forward leakage claim is not established by unused padding key arguments or by historical contrastive losses.',
            limits='Component fixtures are not actual native sparse TRAIN/FPS evidence or an admission gate. No model/config changes, tuning, Git commits, weight persistence or training.', result='pending')
        (ROOT / program).write_text(source, encoding='utf-8', newline='\n')
        plan['program'] = ref(program)
        tick.save(ROOT / plan_path, plan)
        for path in (body, 'scripts/prepare_y53_padding_components_fde0.py', program, plan_path):
            candidate['evidence'].append(ref(path))
        candidate['next_action'] = 'Execute frozen CPU exact-native-DEC padding/repeat/valid controls and language-input gradients; then capture actual frozen TRAIN decoder inputs if a task-relevant uncertainty remains. No model implementation or admission.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(program=ref(program), plan=ref(plan_path)), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
