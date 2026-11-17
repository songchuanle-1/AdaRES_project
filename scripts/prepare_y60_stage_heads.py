"""Prepare the stage-specific confidence experiment under a current lease."""

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import experiment_first
import foreground_tick as tick
import pipeline_monitor as monitor

PACKAGE = 'adares_y60_stage_specific_confidence'
PARENT = 'adares_w2_pointmass_competition'
FILES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')


def ref(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file()
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def lease(token, reserve=90):
    base = ROOT / 'pipeline/headless'
    owner = tick.load(base / '.foreground.lockdir/owner.json')
    assert owner['token'] == token and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
    assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > reserve
    for name, key in (('.tick_context.json', 'context_sha256'),
                      ('.tick_context_state.next.json', 'pending_sha256')):
        assert tick.digest(base / name) == owner[key]
    assert experiment_first.effective(ROOT)['enabled']


def files():
    folder = ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE
    return [folder / name for name in FILES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def packet():
    paths = files()
    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])
    return dict(id='AdaRES-Y60', package=PACKAGE,
                config=paths[-1].relative_to(ROOT / 'AdaRES').as_posix(),
                python_files={p.name: encoded(p) for p in paths[:-1]},
                config_data=encoded(paths[-1]))


def prepare(token):
    row = packet()
    for path in files()[:-1]:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
        if path.name not in ('__init__.py', 'dec.py'):
            assert path.read_bytes() == (path.parent.parent / PARENT / path.name).read_bytes()
    assert list((ROOT / 'AdaRES/configs_song').rglob(Path(row['config']).name)) == [files()[-1]]
    body = ROOT / 'scripts/probe_y60_cpu.py'
    payload = dict(variant=row,
                   parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
                   train_source_lf_sha256=hashlib.sha256(
                       (ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    program = ROOT / ('pipeline/checks/y60_cpu_' + token + '_program.py')
    assert not program.exists()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'),
                       encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(c['id'] == row['id'] for c in work['candidates'])
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['research']['creation_budget'] >= 1
    candidate = dict(id=row['id'], stage='implementation', profile='experiment_first',
        hypothesis='One indicator shared across all seven decoder stages couples confidence fitting '
                   'for different refinement states. Independent stage-specific indicator heads may '
                   'let the final head fit its own mask/query distribution without intermediate-stage '
                   'head-gradient interference. This is a structural training comparison, not an efficacy claim.',
        matched_control='Unchanged AdaRES-W2 native YAML and all seven confidence losses. '
                        'Clone the initial head for every refinement stage without consuming RNG; '
                        'keep point-mass competition, masks, selector, targets, optimizer, seed1999 and fixed30.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=row['config'],
        next_action='Run frozen full CPU native-YAML construction, released-pretrain, actual TRAIN collate '
                    'and independent-head forward/backward controls; then real native two-step GPU smoke '
                    'on an exclusive authorized free slot before admission.',
        evidence=[ref(program), ref(body), ref(__file__)] + [ref(p) for p in files()])
    work['candidates'].append(candidate)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(program=ref(program), candidate=row['id'], native_executed=False)))


def save_version(token):
    result_path = ROOT / ('pipeline/analyses/y60_cpu_' + token + '.json')
    result = tick.load(result_path)
    assert result['complete'] and result['implementation_test_passed'] and not result['CUDA_initialized']
    assert result['configured_TRAIN_data_collated'] and result['full_model_CPU_constructed']
    program_path = ROOT / ('pipeline/checks/y60_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(program_path.read_text(encoding='utf-8')).body[0].value)
    row = payload['variant']
    for path in files():
        expected = row['config_data'] if path.suffix == '.yaml' else row['python_files'][path.name]
        assert ref(path)['sha256'] == expected['sha256']
    def git(*args, binary=False):
        out = subprocess.check_output(['git', '-c', 'core.autocrlf=false', *args], cwd=ROOT)
        return out if binary else out.decode('utf-8').strip()
    assert not git('diff', '--cached', '--name-only'), 'Unrelated staged work must remain untouched'
    parent = git('rev-parse', 'HEAD')
    paths = [p.relative_to(ROOT).as_posix() for p in files()]
    git('add', '--', *paths)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == set(paths)
    for path in paths:
        assert hashlib.sha256(git('show', ':' + path, binary=True)).hexdigest() == ref(path)['sha256']
    git('commit', '--only', '-m', 'Add Y60 stage-specific confidence heads', '--', *paths)
    commit = git('rev-parse', 'HEAD')
    assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).splitlines()) == set(paths)
    for path in paths:
        assert hashlib.sha256(git('show', commit + ':' + path, binary=True)).hexdigest() == ref(path)['sha256']
    binding = ROOT / ('pipeline/checks/y60_version_' + token + '.json')
    assert not binding.exists()
    tick.save(binding, dict(complete=True, parent_commit=parent, code_commit=commit,
        exact_files=[ref(p) for p in files()], test=ref(result_path),
        authorization=tick.load(ROOT / 'pipeline/closure_policy.json')['experiment_first']['authorization'],
        native_smoke_passed=False, admitted=False, pushed=False, synchronized=False))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    c = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y60')
    c.update(stage='validation', code_commit=commit)
    c['evidence'].extend([ref(result_path), ref(result_path.with_suffix('.log')), ref(binding)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(code_commit=commit, files=len(paths), version=ref(binding))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('copy', 'prepare', 'save-version'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        if args.operation == 'copy':
            assert all(not p.exists() for p in files()) and not files()[0].parent.exists()
            files()[0].parent.mkdir()
            for path in files()[:-1]:
                shutil.copyfile(path.parent.parent / PARENT / path.name, path)
            shutil.copyfile(files()[-1].parent / (PARENT + '_res.yaml'), files()[-1])
            print('Copied six independent source files and one native YAML; manual edits pending.')
        elif args.operation == 'prepare':
            prepare(args.token)
        else:
            save_version(args.token)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
