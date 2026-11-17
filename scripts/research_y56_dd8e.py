"""Lease-bound Y56 bundle and test registration; never commit or launch."""

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'dd8e4cf0cbdc46169e8b3eadac13593d'
SKILL = 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline'
sys.path.insert(0, SKILL)
import foreground_tick as tick
import pipeline_monitor as monitor


def lease(reserve=80):
    owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
    assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
    assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > reserve
    for name, key in (('.tick_context.json', 'context_sha256'),
                      ('.tick_context_state.next.json', 'pending_sha256')):
        assert tick.digest(ROOT / 'pipeline/headless' / name) == owner[key]


def ref(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file()
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, value):
    assert not (ROOT / path).exists(), 'Immutable evidence already exists'
    tick.save(ROOT / path, value)
    return ref(path)


def bind(candidate, *rows):
    for row in rows:
        if row not in candidate['evidence']:
            candidate['evidence'].append(row)


def prepare(candidate):
    package = 'adares_y56_unique_competitors'
    parent = ROOT / 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
    destination = parent.with_name(package)
    config = ROOT / 'AdaRES/configs_song/model_paper5' / (package + '_res.yaml')
    files = [ref(p) for p in sorted(destination.glob('*.py'))] + [ref(config)]
    assert len(files) == 7 and candidate['stage'] == 'design'
    for row in files:
        path = ROOT / row['path']
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    old = ROOT / 'pipeline/analyses/y57_native_task_ef2c7bd97d37400ea1b1a32da2909760_program.py'
    frozen = ast.literal_eval(next(n.value for n in ast.parse(old.read_text(encoding='utf-8')).body
                                  if isinstance(n, ast.Assign) and any(
                                      isinstance(t, ast.Name) and t.id == 'FROZEN' for t in n.targets)))
    broader = tick.load(ROOT / 'pipeline/analyses/y56_broader_unique_6debf88fed634a13b772f27bdc244bbd.json')
    scenes = frozen['scenes'] + [dict(dataset_index=r['dataset_index'], scan_id=r['scene'],
                                    ann_ids=r['ann_ids']) for r in broader['cohort'][16:18]]
    assert len(scenes) == 10 and len({s['scan_id'] for s in scenes}) == 10
    tree = ast.parse((parent / 'dec.py').read_text(encoding='utf-8'))
    head = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QueryCompetitionIndicator')
    bundle = save('pipeline/analyses/y56_bundle_' + TOKEN + '.json', dict(
        complete=True, id='AdaRES-Y56', created_at=datetime.now().astimezone().isoformat(),
        files=files, parent_files=[ref(p) for p in sorted(parent.glob('*.py'))],
        config_values=yaml.safe_load(config.read_text(encoding='utf-8')), parameter_delta_expected=0,
        formal_uncommitted=True, source_marker=frozen['commit'],
        semantics='Exact query+valid raw mask classes; exclude full focal class; inverse peer counts.',
        design=ref('pipeline/designs/exp_design_Y56_dd8e.md')))
    payload = dict(package=package, bundle=bundle, files={
        (ROOT / r['path']).name: dict(sha256=r['sha256'], base64=base64.b64encode(
            (ROOT / r['path']).read_bytes()).decode('ascii')) for r in files},
        parent_sources=frozen['sources'], checkpoint=frozen['checkpoint'],
        checkpoint_sha256=frozen['checkpoint_sha256'], scenes=scenes, commit=frozen['commit'],
        config_values=yaml.safe_load(config.read_text(encoding='utf-8')),
        parent_class=ast.unparse(head), config_name=config.name)
    setup = ROOT / 'scripts/setup_y56_prototype_dd8e.py'
    for name in ('cpu', 'native', 'resources'):
        body = ROOT / ('scripts/probe_y56_' + name + '_dd8e.py')
        program = 'PAYLOAD=' + repr(payload) + '\n' + setup.read_text(encoding='utf-8') + '\n' + body.read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3, 8))
        path = 'pipeline/analyses/y56_' + name + '_' + TOKEN + '_program.py'
        assert not (ROOT / path).exists()
        (ROOT / path).write_text(program, encoding='utf-8', newline='\n')
        plan = save(path.replace('_program.py', '_prereg.json'), dict(
            complete=True, id='AdaRES-Y56', preregistered_at=datetime.now().astimezone().isoformat(),
            program=ref(path), body=ref(body), setup=ref(setup), bundle=bundle,
            design=ref('pipeline/designs/exp_design_Y56_dd8e.md'), scenes=scenes,
            sampling='Fixed original8TRAIN plus prespecified discovery2; discovery-conditioned, not prevalence.',
            result='pending', optimizer_steps=0, no_commit_or_admission=True))
        bind(candidate, ref(path), plan, ref(body))
    bind(candidate, bundle, ref(setup), ref(__file__), ref('scripts/test_y56_prototype_dd8e.py'),
         ref('pipeline/reviews/motivation_review_Y56_dd8e.md'), ref('pipeline/designs/exp_design_Y56_dd8e.md'))
    candidate.update(stage='implementation', tested_bundle=bundle,
                     scientific_status='exact_class_realization_frozen_before_tests',
                     next_action='Execute frozen Y56 CPU/native backward/resources in THIS tick; no-copy exact, '
                                 'focal-copy equivalence now explicit. No gate PASS before actual outputs.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare'])
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    assert args.token == TOKEN
    lease()
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
        prepare(candidate)
        work.pop('design_cycle', None)
        lease()
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(id=candidate['id'], stage=candidate['stage'], tested_bundle=candidate['tested_bundle'])))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
