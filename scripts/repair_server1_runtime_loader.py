"""Clear obsolete executable-stack ELF flags in this project's private runtime."""
import argparse
import json
import subprocess
from prepare_y60_stage_heads import ROOT, lease, ref, tick

PROGRAM=r'''
import hashlib,json,pathlib,struct,subprocess
root=pathlib.Path('/data1/songcl/Project/AdaRES_project/runtime/adares_project').resolve()
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
assert str(root)=='/data1/songcl/Project/AdaRES_project/runtime/adares_project'
python=root/'bin/python'
def probe():
    r=subprocess.run([str(python),'-B','-c','import torch; print(torch.__version__)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    return dict(rc=r.returncode,out=r.stdout,error=r.stderr)
before=probe()
assert before['rc']!=0 and 'cannot enable executable stack' in before['error']
changes=[]
for path in (root/'lib/python3.8/site-packages/torch/lib').glob('*.so'):
    assert path.resolve().is_relative_to(root)
    with path.open('rb') as stream:header=stream.read(64)
    if header[:6]!=b'\x7fELF\x02\x01':continue
    phoff=struct.unpack_from('<Q',header,32)[0]
    phsize,phnum=struct.unpack_from('<HH',header,54)
    with path.open('rb') as stream:
        stream.seek(phoff)
        phdr=stream.read(phsize*phnum)
    for index in range(phnum):
        kind,flags=struct.unpack_from('<II',phdr,index*phsize)
        if kind!=0x6474e551 or not (flags&1):continue
        offset=phoff+index*phsize+4
        original=path.read_bytes()
        patched=bytearray(original)
        struct.pack_into('<I',patched,offset,flags&~1)
        assert original[:offset]==patched[:offset] and original[offset+4:]==patched[offset+4:]
        with path.open('r+b') as stream:
            stream.seek(offset)
            stream.write(struct.pack('<I',flags&~1))
        changes.append(dict(path=str(path),offset=offset,flags_before=flags,flags_after=flags&~1,
            before_sha256=hashlib.sha256(original).hexdigest(),after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            all_other_bytes_identical=True))
assert changes
after=probe()
out=dict(complete=True,before=before,after=after,changes=changes,original_runtime_untouched=True,GPU_used=False,
    limitation='Only PT_GNU_STACK loader metadata changed in the private copy; original source, tensors and CUDA kernels are not edited.')
(root.parent/'loader_compatibility.json').write_text(json.dumps(out))
print(json.dumps(out))
assert after['rc']==0 and '1.12.1+cu113' in after['out']
'''

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--token',required=True)
    a=p.parse_args()
    lease(a.token,180)
    out=ROOT/('pipeline/checks/server1_loader_'+a.token+'.json')
    assert not out.exists()
    r=subprocess.run(['ssh','-o','BatchMode=yes','server-1','python3 -B -'],input=PROGRAM.encode(),capture_output=True,timeout=150)
    out.with_suffix('.log').write_bytes(r.stdout+r.stderr)
    assert r.returncode==0,'Keep loader repair log; do not change original environment'
    result=json.loads(r.stdout)
    result['evidence']=[ref(__file__),ref(out.with_suffix('.log'))]
    tick.save(out,result)
    print(json.dumps(dict(evidence=ref(out),changes=result['changes'],import_result=result['after'])))

if __name__=='__main__':main()
