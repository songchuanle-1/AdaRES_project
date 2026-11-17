"""Use checksum-verified shared files and exact original replacements privately."""
import argparse,json,subprocess
from prepare_y60_stage_heads import ROOT,lease,ref,tick

PROGRAM=r'''
import hashlib,json,pathlib,subprocess,os
root=pathlib.Path('/data1/songcl/Project/AdaRES_project')
base=pathlib.Path('/data-controller/songcl/IPDN/data')
target=root/'data'
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
assert not target.exists()
comparison=json.loads((root/'runtime/bootstrap_copy_result.json').read_text())['data_comparison']
assert comparison['command'][0:2]==['rsync','-rcni']
rows=[line.split(':',1) for line in comparison['stdout'].splitlines() if line]
differences={name.rstrip('/') for flag,name in rows if not name.endswith('/')}
expected={'download_sacnnet.py','ScanRefer/scene_graphs_new_train.json','ScanRefer/scene_graphs_new_val.json',
 'scannetv2/sum.py','scannetv2/train/scene0420_00_refer.pth','scannetv2/val/scene0307_00_refer.pth','scannetv2/val/scene0426_00_refer.pth'}
assert differences==expected,differences
target.mkdir()
linked=0
for current,dirs,files in os.walk(str(base)):
    rel=pathlib.Path(current).relative_to(base)
    (target/rel).mkdir(exist_ok=True)
    for name in files:
        relative=rel/name
        if relative.as_posix() in differences:continue
        (target/relative).symlink_to(base/relative)
        linked+=1
verified=[]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
for name in sorted(differences):
    dest=target/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists()
    source='/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/'+name
    subprocess.run(['rsync','-a','--checksum','songcl@192.168.1.163:'+source,str(dest)],check=True,timeout=120)
    result=subprocess.check_output(['ssh','-o','BatchMode=yes','songcl@192.168.1.163','sha256sum '+source]).decode().split()[0]
    actual=sha(dest)
    assert result==actual
    verified.append(dict(path=name,source_sha256=result,local_sha256=actual,bytes=dest.stat().st_size))
assert (target/'sp_unet_backbone.pth').is_file() and (target/'roberta-base').is_dir()
out=dict(complete=True,required_inputs_equal=True,data_root=str(target)+'/',
  full_original_checksum_comparison=comparison,matched_existing_files_linked=linked,
  original_replacements=verified,shared_dataset_modified=False,
  note='The complete original-to-existing rsync checksum comparison found exactly these seven differences. All seven are replaced privately with source-SHA256 verified bytes; other input files refer to the unchanged matched read-only originals. Extra unused shared files are not used for protocol selection.')
(root/'runtime/dataset_verified.json').write_text(json.dumps(out))
print(json.dumps(out))
'''
p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,600)
out=ROOT/('pipeline/checks/server1_data_verified_'+a.token+'.json');assert not out.exists()
r=subprocess.run(['ssh','-o','BatchMode=yes','server-1','python3 -B -'],input=PROGRAM.encode(),capture_output=True,timeout=450)
out.with_suffix('.log').write_bytes(r.stdout+r.stderr)
assert r.returncode==0,'Preserve dataset overlay failure log'
d=json.loads(r.stdout);d['evidence']=[ref(__file__),ref(out.with_suffix('.log')),ref(ROOT/('pipeline/checks/server1_bootstrap_'+a.token+'.json'))];tick.save(out,d)
print(json.dumps(dict(evidence=ref(out),matched_files=d['matched_existing_files_linked'],replacements=d['original_replacements'])))
