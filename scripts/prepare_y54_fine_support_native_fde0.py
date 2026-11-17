"""Reuse exact accepted W2 native setup, freeze a distinct fine-support diagnostic."""
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
        assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds() > 450
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y54')
        assert candidate['stage'] == 'diagnosis'
        for row in candidate['evidence']:
            assert ref(row['path']) == row
        donor = 'pipeline/analyses/y53_padding_native_eval_compat_program_' + TOKEN + '.py'
        setup = (ROOT / donor).read_text(encoding='utf-8').split('records = []\n', 1)[0]
        assert setup.endswith('dataset = build_dataset(cfg.data.train.copy(), get_root_logger())\n')
        body = 'scripts/probe_y54_fine_support_native_fde0.py'
        source = setup + '\n' + (ROOT / body).read_text(encoding='utf-8')
        ast.parse(source, feature_version=(3, 8))
        program = 'pipeline/analyses/y54_fine_support_native_program_' + TOKEN + '.py'
        plan_path = 'pipeline/analyses/y54_fine_support_native_prereg_' + TOKEN + '.json'
        assert not (ROOT / program).exists() and not (ROOT / plan_path).exists()
        plan = dict(complete=True, id='AdaRES-Y54', token=TOKEN, preregistered_at=datetime.now().astimezone().isoformat(),
            reused_setup=ref(donor), body=ref(body),
            fixed_test='SAME8frozenTRAIN scenes/original4two-scene batches/32expressions/seed1999; fixed trained W2 diagnostic. Observe actual sparse forward/FPS/decoder input. Recompute exact coarse-derived gathered support positions and assert equality. Measure actual missing-mask fine supports, unmasked edges and attention mass. SAME fine source/query paired original/repeat/padding-only/corrected-padding/valid positive controls; original-vs-corrected full decoder all7layer confidence and all-valid negative controls.',
            intervention='Only diagnostic fine attention key validity and spatial neighbor k_mask. No model files, source samples, weights, native512fine/128queries, loss weights, thresholds or epochs changed.',
            prediction='True invalid support contamination produces positive native attention mass and a control-separated output effect, removed by validity masks; all-valid expressions should be unaffected within measured repeats. No trained gain or automatic PASS from a shape condition.',
            limits='Diagnostic forward only, hooks restored. No independent implementation, optimizer steps, commits, admission or launch. Layer/support counts are correlated.', result='pending')
        (ROOT / program).write_text(source, encoding='utf-8', newline='\n')
        plan['program'] = ref(program)
        tick.save(ROOT / plan_path, plan)
        for path in (body, 'scripts/prepare_y54_fine_support_native_fde0.py', program, plan_path):
            candidate['evidence'].append(ref(path))
        candidate['next_action'] = 'Run fixed actual two-scene native fine-support validity/attention and matched intervention controls under fresh exclusive GPU checks; only implement a correction if supported by observed task-relevant evidence.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(program=ref(program), plan=ref(plan_path)), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
