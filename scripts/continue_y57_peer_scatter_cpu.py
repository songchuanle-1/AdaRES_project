"""Lease-bound registration and bounded CPU diagnostic for stable Y57."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import authorize_prepared_candidates as scoped
import foreground_tick as tick
import pipeline_monitor as monitor

CAPTURE = 'pipeline/analyses/y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json'
CAPTURE_SHA = '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
CHECKPOINT_SHA = 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
BODY = 'scripts/probe_y57_peer_scatter_cpu.py'
ID = 'AdaRES-Y57'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', required=True)
    parser.add_argument('operation', choices=('prepare', 'run', 'record'))
    args = parser.parse_args()
    scoped.lease(args.token)
    base = 'pipeline/analyses/y57_peer_scatter_' + args.token
    plan_path, program_path, execution_path = (base + ending for ending in ('_prereg.json', '_program.py', '_execution.json'))
    result_path = 'pipeline/analyses/y57_peer_scatter_result_' + args.token + '.json'
    if args.operation == 'run':
        plan = tick.load(ROOT / plan_path)
        assert scoped.ref(program_path) == plan['program'] and scoped.ref(BODY) == plan['body']
        assert scoped.ref(CAPTURE)['sha256'] == CAPTURE_SHA
        assert not (ROOT / execution_path).exists() and not (ROOT / result_path).exists()
        ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'PasswordAuthentication=no', '-o',
               'KbdInteractiveAuthentication=no', '-o', 'StrictHostKeyChecking=yes', '-o',
               'ClearAllForwardings=yes', '-o', 'ConnectTimeout=8', 'songcl@192.168.1.163']
        identity = subprocess.check_output(ssh + ['hostname'], timeout=20).decode().strip()
        assert identity == 'dl01-MS-7C60'
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'PasswordAuthentication=no',
                        str(ROOT / CAPTURE), 'songcl@192.168.1.163:' + plan['frozen']['remote_cache']],
                       check=True, timeout=90)
        scoped.lease(args.token)
        command = [sys.executable, '-B', str(ROOT / 'scripts/run_adares_authorized_probe.py'),
                   'y57_peer_scatter_result', str(ROOT / program_path), '--token', args.token, '--seconds', '300']
        started = datetime.now().astimezone().isoformat()
        response = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=380)
        writer = monitor.acquire_lock(str(ROOT))
        assert writer is not None
        try:
            scoped.lease(args.token)
            tick.save(ROOT / execution_path, dict(complete=response.returncode == 0, observed_from=started,
                observed_until=datetime.now().astimezone().isoformat(), returncode=response.returncode,
                program=plan['program'], stdout=response.stdout, stderr=response.stderr))
            work = tick.load(ROOT / 'pipeline/research_work.json')
            candidate = next(c for c in work['candidates'] if c['id'] == ID)
            candidate['evidence'].append(scoped.ref(execution_path))
            if response.returncode == 0:
                candidate['evidence'].append(scoped.ref(result_path))
                candidate['next_action'] = 'Review frozen CPU matched-control errors and feature-scatter witness; never infer task utility or admit a model from algebra alone.'
            tick.save(ROOT / 'pipeline/research_work.json', work)
        finally:
            writer.close()
        print(response.stdout)
        if response.returncode:
            print(response.stderr)
            raise ValueError('CPU diagnostic failed; all original evidence retained')
        return
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        scoped.lease(args.token)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        if args.operation == 'record':
            candidate = next(c for c in work['candidates'] if c['id'] == ID)
            result = tick.load(ROOT / result_path)
            assert result['complete'] and result['id'] == ID
            assert result['negative_controls_passed'] and result['valid_matched_cases'] == 64
            assert result['representation_witness_cases'] == 64
            candidate['stage'] = 'design'
            candidate['failure_scope'] = 'inconclusive'
            candidate['scientific_status'] = 'Representation capacity only; no task harm, utility or native gate established'
            candidate['target_invariance_limitation'] = 'Fixed masks and GT imply fixed actual IoU; invariant confidence under this intervention is not itself a prediction defect.'
            candidate['next_action'] = ('First inspect source/CPU reconstruction of actual point-GT with geometry/mass binding for the SAME8TRAIN scenes. Then preregister fresh native TRAIN quality/error associations and a decoder/data-reachable perturbation. Do not implement/admit a correction from an algebraic feature-scatter witness. Later sparse capture needs an idle authorized exclusive GPU; this candidate is not declared GPU-blocked.')
            candidate['gate_summary'] = dict(diagnosis='Trained-head input-space witness64/64; task-relevant causal deficit not established',
                controls='Repeat/reversal, focal/mask/count/weight preservation, per-query mean/variance, identical-peer negative all tested',
                native_backward='Not executed full model; CPU forward is not native backward PASS',
                resources='CPU only one thread; native GPU resources not tested', independence='Diagnostic only; no model/config implemented',
                constraints='No new queue/admission/training; no running source, labels or protocol edits')
            references = [
                'pipeline/reviews/motivation_review_Y57_' + args.token + '.md',
                'pipeline/designs/exp_design_Y57_' + args.token + '.md',
                'pipeline/analyses/adares_live_progress_' + args.token + '.json',
            ]
            for relative in references:
                if (ROOT / relative).is_file():
                    reference = scoped.ref(relative)
                    if reference not in candidate['evidence']:
                        candidate['evidence'].append(reference)
            if 'design_cycle' in work and work['design_cycle'].get('attempted_ids') == ['AdaRES-Y56']:
                history_key = 'design_cycle_previous_' + args.token[:8]
                assert history_key not in work
                work[history_key] = work.pop('design_cycle')
            work['reserve_preparation'] = dict(token=args.token, created_ids=[ID],
                completed_bounded_preparation=True, scientific_status=candidate['scientific_status'],
                result=scoped.ref(result_path), next_action=candidate['next_action'],
                no_new_queue_entries=True, training_launches=0, no_exhaustion_or_scientific_cooldown_claim=True)
            tick.save(ROOT / 'pipeline/research_work.json', work)
            print(json.dumps(dict(id=ID, stage=candidate['stage'], scientific_status=result['scientific_status'])))
            return
        assert not any(c['id'] == ID for c in work['candidates'])
        context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
        assert context['mode'] == 'design' and context['research_work']['reserve_only']
        assert context['research_work']['reserve_budget'] >= 1 and context['queue']['candidate_budget'] == 0
        assert scoped.ref(CAPTURE)['sha256'] == CAPTURE_SHA
        source_relative = 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
        source = (ROOT / source_relative).read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'QueryCompetitionIndicator')
        frozen = dict(source_path=source_relative[len('AdaRES/'):], class_source=ast.get_source_segment(source, node),
            source_lf_sha256=hashlib.sha256((ROOT / source_relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
            capture_sha256=CAPTURE_SHA, remote_cache='/tmp/adares_y57_cache_' + args.token + '.json',
            checkpoint='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth',
            checkpoint_sha256=CHECKPOINT_SHA)
        program = 'FROZEN=' + repr(frozen) + '\n' + (ROOT / BODY).read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3, 8))
        compile(program, program_path, 'exec')
        assert not (ROOT / plan_path).exists() and not (ROOT / program_path).exists()
        (ROOT / program_path).write_text(program, encoding='utf-8', newline='\n')
        created = datetime.now().astimezone().isoformat()
        plan = dict(complete=True, id=ID, token=args.token, preregistered_at=created,
            hypothesis_id='competitor_feature_scatter_mean_aliasing',
            capture=scoped.ref(CAPTURE), source=scoped.ref(source_relative), body=scoped.ref(BODY),
            program=scoped.ref(program_path), frozen=frozen,
            fixed_test='All inherited8TRAIN scenes/32expressions/Q128, focal index0 fixed before predictions, both float32/64. Per-row centered Householder reflection, coordinate0 projected orthogonal to constant vector and weighted standardized peer mean; no direction search.',
            prediction='Fixed focal query/masks/masses/overlap/counts and peer normalized weighted mean can hide changed peer feature second moments. Test representational capacity, not model harm or validation gain.',
            matched_controls=['Native trained strict W2 head', 'Repeat and reverse permutation', 'Per-query mean/variance invariance', 'Identical-peer feature negative, seed1999', 'Source/cache/checkpoint hashes unchanged before and after'],
            arithmetic_rule='Per-dtype128eps*(1+observed output/context or moment scale); numerical tolerance only, not an efficacy threshold',
            distinction='Feature-value distribution at fixed masks/counts, not Y56 duplicate multiplicity, rejected Y50 joint mask incidence, or within-superpoint feature covariance.',
            limitations=['Algebraic input intervention, native/data reachability and label relevance unproved', 'Original capture did not bind checkpoint blob SHA', 'No formal model, GPU use, native backward/resources, mIoU evaluation or novelty claim'],
            cpu_only=True, training_launches=0, admission=False)
        tick.save(ROOT / plan_path, plan)
        work['candidates'].append(dict(id=ID, stage='diagnosis', created_at=created,
            hypothesis_id=plan['hypothesis_id'], hypothesis=plan['prediction'],
            matched_control=plan['fixed_test'] + ' Repeat/reversal and identical-peer controls; strict trained W2 checkpoint.',
            next_action='Execute this exact frozen32expression CPU representation test now, then inspect failures and limitations before choosing a model realization.',
            failure_scope='inconclusive', evidence=[scoped.ref(plan_path), scoped.ref(program_path), scoped.ref(source_relative)],
            gate_summary=dict(diagnosis='Pre-registered, not executed', controls='Pending', native_backward='Not executed', resources='CPU only planned', independence='No model implementation', constraints='Reserve only, no queue/admission/training')))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(id=ID, stage='diagnosis', plan=scoped.ref(plan_path), program=scoped.ref(program_path)), indent=2))
    finally:
        writer.close()


if __name__ == '__main__':
    main()
