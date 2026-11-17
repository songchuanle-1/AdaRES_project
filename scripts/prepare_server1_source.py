"""Verify copied source and safely add the already committed candidates."""
import argparse
from datetime import datetime
import json
import shlex
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--token',required=True)
    a=p.parse_args()
    lease(a.token,300)
    server=monitor.load_servers(str(ROOT))['server-1']
    closure.validate_adares_server(ROOT,server)
    program="""import pathlib,subprocess,json
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
root=pathlib.Path('/data1/songcl/Project/AdaRES_project/AdaRES')
p=root/'backbones'
assert p.is_symlink() and str(p.readlink())=='/bak_home/songcl/3D_visual_grounding/CFNet/backbones'
assert pathlib.Path('/data-controller/songcl/IPDN/data/roberta-base').is_dir()
p.unlink()
p.symlink_to('/data-controller/songcl/IPDN/data',target_is_directory=True)
assert (p/'roberta-base/config.json').is_file()
print(json.dumps({'backbones':str(p.resolve()),'source_marker':(root/'.codex_source_commit').read_text().strip()}))
"""
    observed=json.loads(monitor.ssh(server['host'],server['user'],'python3 -B -c '+shlex.quote(program),timeout=30))
    lock=monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        assert closure.safe_sync(ROOT,ROOT,server,monitor.ssh)
        out=ROOT/('pipeline/checks/server1_source_'+a.token+'.json')
        assert not out.exists()
        tick.save(out,dict(complete=True,observed_at=datetime.now().astimezone().isoformat(),
            before=observed,code_commit=closure.git(ROOT,'rev-parse','HEAD'),
            actual_source_integrity_verified=True,source_model_changed=False,
            runtime_path_mappings=server['runtime_path_mappings'],
            evidence=[ref(__file__),ref('pipeline/authorizations/server1_gpu0_20261025.json')]))
        print(json.dumps(dict(evidence=ref(out))))
    finally:lock.close()

if __name__=='__main__':main()
