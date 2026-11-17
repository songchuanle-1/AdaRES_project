"""Deploy only the preregistered decoder while preserving all other sources."""
import fcntl, hashlib, json, os
from pathlib import Path
import subprocess
root = Path(P['root'])
assert P['physical_gpu'] is None
lease = open('/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock', 'a')
fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
used, util = map(int, subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits']).decode().split(','))
assert used < 1024 and util < 10
assert not subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).decode().strip()
marker = root / '.codex_source_commit'
assert marker.read_text().strip() == 'd18c228fd3796b5a87dbe313075b6ee748b2d51b'
target = root / 'ipdn/model_paper5/adares_y46_component_coherence/dec.py'
old_sha = 'bf57bebe7c80e30b0777c824f3969606a8ce65342a127a3e87b6060f0721c25d'
new_sha = '363506506bf0e289d67dfe72c1a0041e66a67fd1962247a726c34d47b838c225'
assert hashlib.sha256(target.read_bytes()).hexdigest() == old_sha
new = Path('/tmp/adares_y46_revision2_dec_f56b.py').read_bytes()
assert hashlib.sha256(new).hexdigest() == new_sha
compile(new.decode(), 'revision2-decoder', 'exec')
protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for n in ('adares_w2_pointmass_competition', 'adares_y45_metric_decomposition', 'adares_y48_mask_equivalence', 'adares_y46_component_coherence') for p in (root / 'ipdn/model_paper5' / n).glob('*.py') if p != target}
archive = Path('/tmp/adares_y46_revision2_api_failed_f56b.py')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == old_sha
temporary = target.with_suffix('.f56b_' + P['token'] + '.tmp')
with temporary.open('xb') as stream:
    stream.write(new)
os.replace(str(temporary), str(target))
assert hashlib.sha256(target.read_bytes()).hexdigest() == new_sha
assert all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected.items()))
assert marker.read_text().strip() == 'd18c228fd3796b5a87dbe313075b6ee748b2d51b'
Path(P['output']).write_text(json.dumps(dict(complete=True, id='AdaRES-Y46', implementation_revision=2, old_decoder_sha256=old_sha, new_decoder_sha256=new_sha, protected_files=len(protected), protected_sources_unchanged=True, source_marker_unchanged=True, no_training=True)), encoding='utf-8')
