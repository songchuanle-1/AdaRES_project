"""Preserve the failed guard and prepare diagnostics with the same assertion."""
import argparse
import ast
import json
from datetime import datetime
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from advance_y86 import prepared, live, idle, unreserved
import closure


def record(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate, plan, program, frozen = prepared(work)
        assert candidate['stage'] == 'validation' and not candidate.get('admission_receipt')
        raw = ROOT / ('pipeline/analyses/y86_gpu0_smoke_' + token + '.log')
        assert raw.exists() and not raw.with_suffix('.json').exists()
        text = raw.read_text(encoding='utf-8')
        assert 'capture_shortcut' in text and 'AssertionError' in text
        assert 'Train[1/30]' in text and 'Load train scanrefer: 36665 samples' in text
        server = monitor.load_servers(str(ROOT))['server-2']
        closure.validate_adares_server(ROOT, server)
        assert server['allowed_gpus'] == [0]
        unreserved(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), 'server-2', 0)
        resource = live(server, 0)
        idle(resource, 280.)
        source = program.read_text(encoding='utf-8')
        guard = '    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)'
        assert source.count(guard) == 1
        diagnostic = guard + """, dict(step=len(step_rows), stage=state['shortcuts'],
        weight_nonzero=nonzero, weight=module.weight.detach().cpu().tolist(),
        overlap_min=float(overlap.detach().min()), overlap_max=float(overlap.detach().max()),
        logit_min=float(logits.detach().min()), logit_max=float(logits.detach().max()),
        confidence_delta_max=delta, original_assertion_unchanged=True)"""
        traced = source.replace(guard, diagnostic)
        old = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assert)
                   and ast.unparse(node.test) == 'nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)')
        new = next(node for node in ast.walk(ast.parse(traced)) if isinstance(node, ast.Assert)
                   and ast.dump(node.test) == ast.dump(old.test))
        assert ast.dump(old.test) == ast.dump(new.test)
        ast.parse(traced, feature_version=(3, 8))
        path = ROOT / ('pipeline/checks/y86_gpu0_diagnostic_' + token + '_program.py')
        output = ROOT / ('pipeline/checks/y86_native_failure_' + token + '.json')
        assert not path.exists() and not output.exists()
        path.write_text(traced, encoding='utf-8', newline='\n')
        tick.save(output, dict(complete=True, id='AdaRES-Y86', token=token,
            observed_at=datetime.now().astimezone().isoformat(), kind='native_smoke', result='failed',
            failure_scope='implementation', original_guard_preserved=True, native_smoke_passed=False,
            scientific_efficacy='Unknown; the failed response witness is not a trained efficacy result.',
            source_manifest=candidate['source_manifest'], original_plan=candidate['native_smoke_plan'],
            raw_log=ref(raw), diagnostic_program=ref(path), idle_resource=resource,
            command='python -B scripts/run_experiment_first_probe.py y86_gpu0_smoke '
                + ref(program)['path'] + ' --token ' + token + ' --seconds 600 --gpu --gpu-id 0',
            next_action='Capture actual values at the unchanged guard; preserve failure, do not admit, then advance independent prepared work.',
            helper=ref(__file__)))
        candidate.update(stage='blocked', blocked_from_stage='validation',
            blocker_kind='numerical_control_failure', failure_scope='implementation',
            blocker='Actual original GPU0 TRAIN response witness failed at capture_shortcut on the second batch. Preserve the unchanged frozen guard and raw failure; no native PASS or admission.',
            native_smoke_failure=ref(output),
            next_action='Diagnose the exact response witness with unchanged assertion and recorded values; do not infer hypothesis falsification or admit. Advance independent Y87 on an authorized exclusive idle GPU0.')
        candidate['evidence'].extend([ref(output), ref(raw), ref(path), ref(__file__)])
        work['tick_attempt'] = dict(
            tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id='AdaRES-Y86', kind='native_smoke', result='failed',
            command=tick.load(output)['command'],
            observation='Original native TRAIN loaded and first step advanced; second batch failed the frozen capture_shortcut nonzero-response guard. Not admitted; diagnostic keeps the same condition.',
            evidence=[ref(output), ref(raw)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(failure=ref(output), diagnostic_program=ref(path), native_PASS=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    record(parser.parse_args().token)
