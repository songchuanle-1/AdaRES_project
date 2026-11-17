"""Freeze and execute the next existing Y57 CPU diagnostic under its lease."""
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
BODY = 'scripts/probe_y57_gt_binding_cpu.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', required=True)
    parser.add_argument('operation', choices=('prepare', 'run'))
    args = parser.parse_args()
    scoped.lease(args.token)
    base = 'pipeline/analyses/y57_gt_binding_' + args.token
    plan_path, program_path, execution_path = (base + ending for ending in ('_prereg.json', '_program.py', '_execution.json'))
    result_path = 'pipeline/analyses/y57_gt_binding_result_' + args.token + '.json'
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y57')
        assert candidate['stage'] in ('design', 'validation')
        if args.operation == 'prepare':
            assert scoped.ref(CAPTURE)['sha256'] == CAPTURE_SHA
            relatives = ['ipdn/model_paper5/adares_w2_pointmass_competition/' + name for name in ('dec.py', 'ipdn.py', 'loss.py')]
            config = 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
            relatives.append(config)
            sources = [dict(path=relative, lf_sha256=hashlib.sha256((ROOT / 'AdaRES' / relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest()) for relative in relatives]
            dec_source = (ROOT / 'AdaRES' / relatives[0]).read_text(encoding='utf-8')
            cls = next(node for node in ast.parse(dec_source).body if isinstance(node, ast.ClassDef) and node.name == 'QueryCompetitionIndicator')
            loss_source = (ROOT / 'AdaRES' / relatives[2]).read_text(encoding='utf-8')
            iou = next(node for node in ast.parse(loss_source).body if isinstance(node, ast.FunctionDef) and node.name == 'get_iou')
            frozen = dict(sources=sources, config_path=config, class_source=ast.get_source_segment(dec_source, cls),
                iou_source=ast.get_source_segment(loss_source, iou), capture_sha256=CAPTURE_SHA,
                remote_cache='/tmp/adares_y57_gt_cache_' + args.token + '.json',
                checkpoint='/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth',
                checkpoint_sha256=CHECKPOINT_SHA)
            program = 'FROZEN=' + repr(frozen) + '\n' + (ROOT / BODY).read_text(encoding='utf-8')
            ast.parse(program, feature_version=(3, 8))
            compile(program, program_path, 'exec')
            assert not (ROOT / plan_path).exists() and not (ROOT / program_path).exists()
            (ROOT / program_path).write_text(program, encoding='utf-8', newline='\n')
            plan = dict(complete=True, id='AdaRES-Y57', token=args.token,
                preregistered_at=datetime.now().astimezone().isoformat(), capture=scoped.ref(CAPTURE),
                body=scoped.ref(BODY), program=scoped.ref(program_path), frozen=frozen,
                fixed_sample='SAME8TRAIN scenes indices0,44,60,101,115,126,137,171; all32annotations and all128queries; seed1999 per-scene. No new sample or focal selection.',
                binding='Exact scan/ann/object identity, counts and point-GT totals, deterministic dataset/collate repeat; cached centroid error<=32eps*max(1,coordinate scale); exact native expanded-point IoU for first4queries. All32must bind before scene association summary.',
                descriptor='All256channels diagonal variance under unchanged point-overlap competition; scalar L2 of the variance vector. Not trace alone, channel selection, an added model or fitted predictor.',
                analysis='Within-expression Spearman dispersion vs absolute native foreground-probability/point-IoU error, and vs point-IoU. Expression means within each of8scenes; descriptive mean and10000/1999 scene-bootstrap95interval. Undefined correlations are unknown with denominators, never zero.',
                controls='Joint reversed query/label permutation; fixed1-step rotated quality labels break correspondence without changing marginals. Record null result without forced direction or significance claim.',
                selector='Exact existing RES foreground softmax argmax; no score-head multiplication or GT/oracle inference selection. Point quality is TRAIN diagnostic only.',
                limitations='Old capture did not hash raw point-GT/checkpoint at capture time; reconstruction cannot retroactively create provenance. Correlation on8TRAIN scenes is not conditional incremental information, causal harm, reachable intervention, validation effect, native gate PASS or novelty.',
                cpu_only=True, training_launches=0, admission=False)
            tick.save(ROOT / plan_path, plan)
            candidate['evidence'].extend([scoped.ref(plan_path), scoped.ref(program_path)])
            candidate['next_action'] = 'Execute the frozen Y57 CPU point-GT/geometry/count reconstruction now; compute descriptive associations only if all32expressions bind. Preserve mismatches and undefined results.'
            tick.save(ROOT / 'pipeline/research_work.json', work)
            print(json.dumps(dict(plan=scoped.ref(plan_path), program=scoped.ref(program_path)), indent=2))
            return
        plan = tick.load(ROOT / plan_path)
        assert scoped.ref(program_path) == plan['program'] and scoped.ref(BODY) == plan['body']
        assert not (ROOT / execution_path).exists() and not (ROOT / result_path).exists()
    finally:
        writer.close()
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'PasswordAuthentication=no', '-o',
           'KbdInteractiveAuthentication=no', '-o', 'StrictHostKeyChecking=yes', '-o',
           'ClearAllForwardings=yes', '-o', 'ConnectTimeout=8', 'songcl@192.168.1.163']
    assert subprocess.check_output(ssh + ['hostname'], timeout=20).decode().strip() == 'dl01-MS-7C60'
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'PasswordAuthentication=no',
                    str(ROOT / CAPTURE), 'songcl@192.168.1.163:' + plan['frozen']['remote_cache']], check=True, timeout=90)
    scoped.lease(args.token)
    command = [sys.executable, '-B', str(ROOT / 'scripts/run_adares_authorized_probe.py'),
               'y57_gt_binding_result', str(ROOT / program_path), '--token', args.token, '--seconds', '600']
    started = datetime.now().astimezone().isoformat()
    response = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=680)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        scoped.lease(args.token)
        tick.save(ROOT / execution_path, dict(complete=response.returncode == 0,
            observed_from=started, observed_until=datetime.now().astimezone().isoformat(),
            returncode=response.returncode, program=plan['program'], stdout=response.stdout, stderr=response.stderr))
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y57')
        candidate['evidence'].append(scoped.ref(execution_path))
        if response.returncode == 0:
            candidate['evidence'].append(scoped.ref(result_path))
            candidate['next_action'] = 'Review actual GT binding and descriptive association/null controls; retain provenance limitations before planning a realizable native intervention.'
        tick.save(ROOT / 'pipeline/research_work.json', work)
    finally:
        writer.close()
    print(response.stdout)
    if response.returncode:
        print(response.stderr)
        raise ValueError('Y57 CPU test failed; preserve exact program, plan, execution and raw log')


if __name__ == '__main__':
    main()
