"""Preregister bounded Y49 implementation revision before executing its tests."""
import argparse, ast, copy, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
parser=argparse.ArgumentParser()
parser.add_argument('--token',required=True)
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
source=ROOT/'pipeline/analyses/y49_instrumentation_program_c3e5_c3e5627500e14e3fa88075dc5544095e.py'
assert ref(source)['sha256']=='5d762fe9267264a4d8889f4787957c635fab3996ff897cd4fc3329b590db7d54'
cpu=ROOT/'scripts/probe_y49_forest_revision_f56b.py'
tree=ast.parse(source.read_text())
helpers=ast.parse(cpu.read_text())
forest=copy.deepcopy(next(n for n in helpers.body if isinstance(n,ast.FunctionDef) and n.name=='forest_inverse_diagonal'))
forest.name='inverse_diagonal'
body=[]
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name=='inverse_diagonal':
        reference=copy.deepcopy(node)
        reference.name='reference_inverse'
        body.extend([reference,*ast.parse("forest_counts=dict(forest_calls=0,cyclic_fallback_calls=0)").body,forest])
    else:body.append(node)
tree.body=body
text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
compile(text,'<y49-revision1>','exec')
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None,'Pipeline writer busy'
try:
    program=ROOT/('pipeline/analyses/y49_revision1_native_program_f56b_'+args.token+'.py')
    assert not program.exists()
    with program.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    work_path=ROOT/'pipeline/research_work.json'
    work=tick.load(work_path)
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y49')
    assert candidate['stage']=='validation' and not candidate.get('revision_history')
    original=ROOT/'pipeline/analyses/y49_native_12be_12be943f16874034808ff67e741653ad.json'
    revision=dict(revision=1,failure_scope='implementation',reason='Actual warmed TRAIN8 native loss/backward costs .396-11.854s candidate versus .213-.304s disabled parent; repeated sparse inverse-diagonal solves are a concrete implementation cost risk.',change='For undirected unweighted forest induced graphs compute the SAME inverse diagonal by exact leaf Schur elimination and reverse messages. Any cyclic graph retains original sparse-LU block64 solver. No graph/mask/mass/head/target/scale change.',prediction='All32 frozen TRAIN expressions preserve descriptor values within the EXISTING8float64eps*nodes bound and component partitions exactly; first4 joint permutation and ring4/8 fallback controls pass. All4 proxy and then native loss gradients stay finite/nonzero. Measure paired CPU/native costs without fitting a success threshold.',preregistered_at=datetime.now().astimezone().isoformat(),failure_evidence=[ref(original)],result='pending',evidence=[])
    freeze=ROOT/('pipeline/analyses/y49_revision1_prereg_f56b_'+args.token+'.json')
    tick.save(freeze,dict(complete=True,id='AdaRES-Y49',revision=revision,original_program=ref(source),CPU_program=ref(cpu),native_program=ref(program),unchanged_native_diagnosis_limit='Zero native equal-component-mass/different-global-value witnesses; revision does not repair or erase this limitation.',no_training_or_admission=True))
    candidate['revision_history']=[revision]
    candidate['failure_scope']='implementation'
    candidate['evidence'].extend([ref(freeze),ref(cpu),ref(program)])
    candidate['next_action']='Execute preregistered same-ID revision1 CPU32 equivalence/permutation/cyclic controls, then SAME revised native TRAIN8 loss/backward/cached controls/costs only on live idle exclusive GPU. Native task diagnosis and production/resources remain pending.'
    tick.save(work_path,work)
    print(json.dumps(dict(complete=True,preregistered_at=revision['preregistered_at'],freeze=ref(freeze),native_program=ref(program))))
finally:lock.close()
