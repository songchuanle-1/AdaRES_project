"""Preregister and run CPU controls on existing frozen actual TRAIN inputs."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import authorize_prepared_candidates as scoped
import closure
import foreground_tick as tick
import pipeline_monitor as monitor

CAPTURE = 'pipeline/analyses/y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json'
CAPTURE_SHA = '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
CHECKPOINT_SHA = 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
BODY = 'scripts/probe_y56_cached_train_cpu.py'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token',required=True)
    parser.add_argument('operation',choices=('prepare','run'))
    args=parser.parse_args()
    scoped.lease(args.token)
    writer=monitor.acquire_lock(str(ROOT))
    if writer is None: raise ValueError('Project writer busy')
    try:
        work=tick.load(ROOT/'pipeline/research_work.json')
        candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y56')
        base=f'pipeline/analyses/y56_cached_train_{args.token}'
        plan_path,program_path,result_path,execution_path=(base+s for s in ('_prereg.json','_program.py','_result.json','_execution.json'))
        if args.operation=='prepare':
            assert candidate['stage']=='diagnosis' and not (ROOT/plan_path).exists()
            assert scoped.ref(CAPTURE)['sha256']==CAPTURE_SHA
            cached=tick.load(ROOT/CAPTURE)
            old_body=(ROOT/'scripts/probe_y56_competitor_density_cpu.py').read_text(encoding='utf-8')
            nodes=ast.parse(old_body).body
            row_index=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in n.targets))
            preamble='\n'.join(ast.get_source_segment(old_body,n) for n in nodes[:row_index])
            fixture_loop=ast.get_source_segment(old_body,nodes[row_index+1])
            assert isinstance(nodes[row_index+1],ast.For)
            source_path='AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
            source=(ROOT/source_path).read_text(encoding='utf-8')
            cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='QueryCompetitionIndicator')
            scenes=[dict(dataset_index=row['dataset_index'],scan_id=row['scan_id'],expressions=[
                dict(ann_id=e['ann_id'],packed_native_inputs={key:e['packed_native_inputs'][key] for key in ('query','masks','masses')})
                for e in row['expressions']]) for row in cached['rows']]
            payload=dict(class_source=ast.get_source_segment(source,cls),fixture_loop=fixture_loop,scenes=scenes,
                checkpoint='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth',
                checkpoint_sha256=CHECKPOINT_SHA,capture_sha256=CAPTURE_SHA)
            program='PAYLOAD='+repr(payload)+'\n'+preamble+'\n'+(ROOT/BODY).read_text(encoding='utf-8')
            ast.parse(program,feature_version=(3,8))
            (ROOT/program_path).write_text(program,encoding='utf-8',newline='\n')
            now=datetime.now(timezone(timedelta(hours=8))).isoformat()
            tick.save(ROOT/plan_path,dict(complete=True,id='AdaRES-Y56',token=args.token,preregistered_at=now,
                capture=scoped.ref(CAPTURE),capturer=scoped.ref('scripts/probe_y46_native_capture_d179.py'),
                parent_source=scoped.ref(source_path),body=scoped.ref(BODY),program=scoped.ref(program_path),
                fixed_test='SAME8frozenTRAIN scenes/32expressions/Q128 final cached indicator inputs; exact query/softmask duplicates, separately native hardmask>0 equivalence; no new sampling/thresholds. CPU trained W2 indicator strict state loading, repeat/reversal controls. Conditional transfer one existing exact-copy competitor to another only when singleton focal and all unique prototypes survive; otherwise not_applicable, never synthesize a task positive.',
                trained_fixture='Reuse original fixed float32/64 seed1999 counts[1,63,64]/[1,64,63] loop with strict trained W2 head, repeat/permutation/negative/unique-reference controls.',
                prediction='Measure incidence and separated trained-head effects without requiring a task-gain threshold; zero exact duplicates does not prove absence on all TRAIN batches/layers.',
                capture_provenance_limit='Original capture code specified W2 checkpoint path but did not record its blob SHA at capture time; current replay SHA does not retroactively certify that capture.',
                cpu_only=True,actual_GPU_run=False,scientific_gate_passed=False))
            candidate['evidence'].extend([scoped.ref(plan_path),scoped.ref(program_path)])
            candidate['next_action']='Run hash-frozen32expression actual-input CPU incidence and strict trained-head controls now; do not use any GPU or infer native gate/efficacy.'
            tick.save(ROOT/'pipeline/research_work.json',work)
            print(json.dumps(dict(plan=scoped.ref(plan_path),program_bytes=(ROOT/program_path).stat().st_size)))
            return
        plan=tick.load(ROOT/plan_path)
        assert scoped.ref(program_path)==plan['program'] and scoped.ref(BODY)==plan['body']
        assert not (ROOT/execution_path).exists() and not (ROOT/result_path).exists()
        command=['ssh','-T','-o','BatchMode=yes','-o','PasswordAuthentication=no','-o','KbdInteractiveAuthentication=no',
            '-o','StrictHostKeyChecking=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8','songcl@192.168.1.163',
            "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages /home/songcl/.conda/envs/adares_project/bin/python -B -S -"]
        started=datetime.now(timezone(timedelta(hours=8))).isoformat()
        try:
            response=subprocess.run(command,input=(ROOT/program_path).read_text(encoding='utf-8'),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
        except (subprocess.TimeoutExpired,OSError) as error:
            tick.save(ROOT/execution_path,dict(complete=False,observed_from=started,error=repr(error),program=plan['program']))
            candidate['evidence'].append(scoped.ref(execution_path));tick.save(ROOT/'pipeline/research_work.json',work)
            raise
        ended=datetime.now(timezone(timedelta(hours=8))).isoformat()
        tick.save(ROOT/execution_path,dict(complete=True,observed_from=started,observed_until=ended,command=command,
            returncode=response.returncode,stdout=response.stdout,stderr=response.stderr,program=plan['program']))
        candidate['evidence'].append(scoped.ref(execution_path))
        tick.save(ROOT/'pipeline/research_work.json',work)
        if response.returncode: raise ValueError('CPU replay failed; preserve evidence, never force PASS')
        measured=json.loads(response.stdout)
        tick.save(ROOT/result_path,measured)
        candidate['evidence'].append(scoped.ref(result_path))
        candidate['gate_summary']['diagnosis']='Actual cached32expression single-scene final-input CPU incidence plus strict current trained W2 head replay; provenance limitation retained; native batch2/all-layer diagnosis pending'
        candidate['next_action']='Inspect actual CPU incidence, retain conditional-not-applicable/negative outcomes; then prepare source-bound native batch2 all-layer TRAIN capture with true idle-GPU guards, without stopping Y52/Y54.'
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(result=scoped.ref(result_path),observed_from=started,observed_until=ended,
            summary={k:measured[k] for k in ('trained_head_fixture_signal','exact_joint_duplicate_excess','hard_mask_duplicate_excess','applicable_counterfactuals','max_repeat_error','max_permutation_error')}),indent=2))
    finally: writer.close()


if __name__=='__main__': main()
