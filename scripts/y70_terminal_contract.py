"""Bind original Y70 to the exact, additive Y71 production deployment."""
import hashlib
import subprocess
import yaml
from prepare_y60_stage_heads import ROOT
from observe_y70_y71_exact_deployment import OLD, NEW, NAMES, validate_marker
from admit_dispatch_y71_gpu0 import validate_additive_delta, validate_keeper
import closure


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT, timeout=30)


def build_contract(task, server):
    validate_keeper(task)
    assert task['pid'] == 52561 and task['server'] == 'server-2'
    assert server['host'] == '192.168.1.163' and server['expected_hostname'] == 'dl01-MS-7C60'
    assert server['allowed_gpus'] == [0, 1] and server['max_tasks_per_gpu'] == 1
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == NEW
    assert closure.git(ROOT, 'rev-parse', NEW + '^') == OLD
    rows = [row.split('\t') for row in subprocess.check_output(
        ['git', 'diff', '--no-renames', '--name-status', OLD, NEW], cwd=ROOT, timeout=30).decode().splitlines()]
    validate_additive_delta(rows)
    additions = dict(source_files={}, configs={})
    for status, path in rows:
        raw = blob(NEW, path)
        assert (ROOT / path).read_bytes() == raw
        relative = path[len('AdaRES/'):]
        if path.endswith('.yaml'):
            additions['configs'][relative] = yaml.safe_load(raw)
        else:
            additions['source_files'][relative] = hashlib.sha256(raw).hexdigest()
    for name in NAMES:
        path = 'AdaRES/ipdn/model_paper5/adares_y70_query_scale_fusion/' + name
        assert (ROOT / path).read_bytes() == blob(OLD, path) == blob(NEW, path)
    path = 'AdaRES/' + task['config']
    assert yaml.safe_load((ROOT / path).read_bytes()) == yaml.safe_load(blob(OLD, path)) == yaml.safe_load(blob(NEW, path))
    return dict(allowed_markers=[OLD, NEW], deployments={
        OLD: dict(source_files={}, configs={}), NEW: additions})


IDENTITY = '''import hashlib,yaml
root=pathlib.Path(P['root'])
for relative,digest in P['source_files'].items():
 assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
assert yaml.safe_load((root/P['config']).read_text())==P['expected_config']
marker=(root/'.codex_source_commit').read_text().strip()
assert marker in P['allowed_markers']
for relative,digest in P['deployments'][marker]['source_files'].items():
 assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
for relative,config in P['deployments'][marker]['configs'].items():
 assert yaml.safe_load((root/relative).read_text())==config
'''
