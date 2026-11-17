"""Apply explicit human scoped-commit authorization; never stage unrelated paths."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
TOKEN='75486f9263d34214b3bd2e1412c69639'
parser=argparse.ArgumentParser()
parser.add_argument('operation',choices=('authorize','commit'))
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds()>180
AUTH='pipeline/authorizations/AdaRES-Y50_commit.json'
BUNDLE='pipeline/analyses/y50_production_bundle_1cb1_1cb15474bfd346dcbc34837f3e9b79d3.json'
bundle=tick.load(ROOT/BUNDLE)
assert bundle['complete'] and bundle['id']=='AdaRES-Y50' and len(bundle['files'])==7
files=[row['path'] for row in bundle['files']]
assert all(p.startswith('AdaRES/ipdn/model_paper5/adares_y50_joint_incidence/') or p=='AdaRES/configs_song/model_paper5/adares_y50_joint_incidence_res.yaml' for p in files)
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),'-c','core.autocrlf=false',*args])
for row in bundle['files']:
    assert ref(row['path'])['sha256']==row['sha256'],'Tested source changed: '+row['path']
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y50')
    for evidence in candidate['evidence']:
        assert ref(evidence['path'])['sha256']==evidence['sha256'],'Bound evidence changed: '+evidence['path']
    if args.operation=='authorize':
        assert candidate['stage']=='blocked' and candidate['blocker_kind']=='human_authorization'
        assert not (ROOT/AUTH).exists()
        head=git('rev-parse','HEAD').decode().strip()
        assert head=='d18c228fd3796b5a87dbe313075b6ee748b2d51b'
        assert not git('ls-files','--',*files).strip(),'Candidate unexpectedly already tracked'
        (ROOT/AUTH).parent.mkdir(parents=True,exist_ok=True)
        tick.save(ROOT/AUTH,dict(complete=True,id='AdaRES-Y50',kind='explicit_human_scoped_commit_authorization',
            recorded_at=datetime.now().astimezone().isoformat(),thread_id=owner['thread_id'],
            human_message='n那你授权啊',
            context='Direct human reply after assistant defined authorization as only committing tested AdaRES candidate code and continuing existing gated admission/dispatch.',
            scope='Only the seven exact tested Y50 files listed here; existing six scientific gates, server2GPU0/1 single-GPU capacity and fixed30 protocol remain unchanged.',
            files=bundle['files'],tested_bundle=ref(BUNDLE),starting_HEAD=head,
            permits=['Scoped explicit Git add/commit for these tested Y50 files.','After revalidation, canonical gated admission and dispatch for SAME Y50.'],
            excludes=['Unrelated tracked/staged/untracked files.','Other projects, scheduler/background/new tasks.','Goal/gate/protocol changes, historical reopening or diagnostic weight deployment.']))
        prior={key:candidate[key] for key in ('blocker_kind','blocker','retry_after','unblock_condition') if key in candidate}
        candidate.setdefault('blocker_history',[]).append(dict(**prior,resolved_at=datetime.now().astimezone().isoformat(),resolved_by=ref(AUTH)))
        for key in ('blocker_kind','blocker','retry_after','unblock_condition','required_server'):
            candidate.pop(key,None)
        candidate.update(stage='validation',next_action='Human scoped Y50 commit authorization received. Verify exact tested files/evidence, create ONLY seven-file commit, then bind six gates and frozen post-training plan to real commit before canonical gated dispatch.')
        candidate['evidence'].append(ref(AUTH))
        work.pop('design_cycle',None)
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(authorization=AUTH,stage=candidate['stage'],tested_files=len(files),no_commit_yet=True),indent=2))
    else:
        authorization=tick.load(ROOT/AUTH)
        assert authorization['kind']=='explicit_human_scoped_commit_authorization'
        assert authorization['files']==bundle['files'] and candidate['stage']=='validation'
        assert git('rev-parse','HEAD').decode().strip()==authorization['starting_HEAD']
        before=git('ls-files','--stage','-z').split(b'\0')
        def unrelated(rows):return sorted(row for row in rows if row and row.split(b'\t',1)[1].decode('utf-8') not in files)
        before_unrelated=unrelated(before)
        subprocess.run(['git','-C',str(ROOT),'-c','core.autocrlf=false','add','--',*files],check=True)
        for row in bundle['files']:
            assert hashlib.sha256(git('show',':'+row['path'])).hexdigest()==row['sha256'],'Staged bytes differ from tested bytes'
        subprocess.run(['git','-C',str(ROOT),'-c','core.autocrlf=false','commit','--only','-m','Add tested AdaRES Y50 joint-incidence confidence','--',*files],check=True)
        commit=git('rev-parse','HEAD').decode().strip()
        changes=git('diff-tree','--no-commit-id','--name-status','-r',commit).decode().splitlines()
        assert sorted(changes)==sorted('A\t'+p for p in files)
        for row in bundle['files']:
            assert hashlib.sha256(git('show',commit+':'+row['path'])).hexdigest()==row['sha256']
        assert unrelated(git('ls-files','--stage','-z').split(b'\0'))==before_unrelated
        record='pipeline/analyses/y50_commit_binding_7548_'+TOKEN+'.json'
        assert not (ROOT/record).exists()
        tick.save(ROOT/record,dict(complete=True,id='AdaRES-Y50',observed_at_local=datetime.now().astimezone().isoformat(),
            authorization=ref(AUTH),code_commit=commit,parent_commit=authorization['starting_HEAD'],
            files=bundle['files'],exact_committed_tested_bytes=True,only_seven_additions=True,unrelated_index_unchanged=True,
            no_admission_or_launch_yet=True))
        candidate['evidence'].append(ref(record))
        candidate.update(code_commit=commit,next_action='Exact seven-file tested-byte commit created under explicit human authorization; bind unchanged six gates/goal/frozen evaluation to this real HEAD, then canonical gated server2 dispatch with fresh GPU checks and native progress verification.')
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(commit=commit,record=record,committed_files=len(files),unrelated_index_unchanged=True),indent=2))
finally:lock.close()
