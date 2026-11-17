"""Bind the unchanged full evaluator to the prospectively frozen Y50 comparison."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
from prepare_y52_native_backward_89ff import ref, TOKEN
import foreground_tick as tick
import pipeline_monitor as monitor

lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
    assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
    assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 1000
    for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
        assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y50')
    assert candidate['stage'] == 'training'
    for row in candidate['evidence']:
        assert ref(row['path']) == row
    freeze_path = 'pipeline/analyses/y50_post_training_freeze_1cb1_1cb15474bfd346dcbc34837f3e9b79d3.json'
    freeze = tick.load(ROOT / freeze_path)
    assert 'twice EACH; FIRST primary' in freeze['protocol']['evaluation']
    binding_path = 'pipeline/analyses/y50_commit_binding_7548_75486f9263d34214b3bd2e1412c69639.json'
    binding = tick.load(ROOT / binding_path)
    for row in binding['files']:
        assert ref(row['path']) == row
    terminal_path = 'pipeline/analyses/y52_source_y50_exit_89ff_' + TOKEN + '.json'
    terminal = tick.load(ROOT / terminal_path)
    assert terminal['Y50']['fixed30_normal_exit_evidence']
    best = next(r for r in terminal['Y50']['checkpoints'] if r['path'].endswith('/best.pth'))
    old = ast.parse((ROOT / 'scripts/probe_y48_frozen_full_12be.py').read_text(encoding='utf-8'))
    adapter = ast.literal_eval(next(n.value for n in old.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'adapter' for t in n.targets)))
    evaluator_sha = '7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13'
    assert ref('AdaRES/tools/evaluate_adares_frozen_full.py')['sha256'] == evaluator_sha
    bindings = [
        ('y50', 'adares_y50_joint_incidence_res', best['path'],
         'd4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb', best['sha256']),
        ('w2', 'adares_w2_pointmass_competition_res',
         '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth',
         'd4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb',
         'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'),
        ('b1', 'adares_b1_standard_res',
         '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b1_standard_res/20260918_010854/best.pth',
         '04e3cc580751c72a6fc908b2ff93ad931ab4a3a7d07d1fc9b7ae30a9a993e1dc',
         'cfc89ec0dc64f240913968f307d19c623f5ff0dc51c8e61de64e35abea05dd8b')]
    body_path = 'scripts/probe_y50_frozen_eval_89ff.py'
    body = (ROOT / body_path).read_text(encoding='utf-8')
    programs = []
    for identity, config_name, checkpoint, config_sha, checkpoint_sha in bindings:
        payload = dict(identity=identity, config='configs_song/model_paper5/' + config_name + '.yaml',
            config_sha256=config_sha, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha,
            terminal_sha256=ref(terminal_path)['sha256'], code_commit=binding['code_commit'],
            source_files=[dict(path=r['path'].removeprefix('AdaRES/'), sha256=r['sha256']) for r in binding['files']],
            evaluator_sha256=evaluator_sha, adapter=adapter)
        source = 'PAYLOAD=' + repr(payload) + '\n' + body
        ast.parse(source, feature_version=(3, 8))
        path = 'pipeline/analyses/y50_frozen_' + identity + '_program_' + TOKEN + '.py'
        assert not (ROOT / path).exists()
        (ROOT / path).write_text(source, encoding='utf-8', newline='\n')
        programs.append(ref(path))
    plan_path = 'pipeline/analyses/y50_frozen_full_bindings_' + TOKEN + '.json'
    assert not (ROOT / plan_path).exists()
    plan = dict(complete=True, id='AdaRES-Y50', token=TOKEN,
        prepared_at=datetime.now().astimezone().isoformat(), prospective_freeze=ref(freeze_path),
        terminal_observation=ref(terminal_path), code_binding=ref(binding_path), body=ref(body_path),
        programs=programs, evaluator_sha256=evaluator_sha,
        protocol='Unchanged full9508/141, twice each Y50/W2/B1, FIRST primary; existing evaluator and redundant-lock adapter reused exactly. Scene-bootstrap10000/1999, TRAIN cuts1487/3325 and CI upper<0 harm review unchanged.',
        canonical_deathwatch='Preflight registry still running/deathwatch1of3; physical fixed30 exit is separately proven. No queue status forced, preflight repeated or history rewritten.',
        result='pending')
    tick.save(ROOT / plan_path, plan)
    for path in (terminal_path, body_path, 'scripts/prepare_y50_frozen_eval_89ff.py', plan_path):
        if path not in {e['path'] for e in candidate['evidence']}:
            candidate['evidence'].append(ref(path))
    candidate['evidence'].extend(programs)
    candidate['terminal_observation'] = ref(terminal_path)
    candidate['next_action'] = 'Physical fixed30 exit and epoch30 last metadata verified; canonical deathwatch1of3 still pending. Execute frozen Y50/W2/B1 full9508/141 twice EACH with FIRST primary, then paired scene-bootstrap/subgroup harm and cost review. Do not force queue reconciliation or promote from process metrics.'
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(plan=ref(plan_path), programs=programs, scientific_evaluator_unchanged=True), indent=2))
finally:
    lock.close()
