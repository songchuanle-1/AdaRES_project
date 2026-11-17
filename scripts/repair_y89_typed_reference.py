"""Correct only observer arithmetic dtype; preserve failed programs and tolerance."""
import argparse
import ast
from datetime import datetime
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure

BEFORE = '''    probabilities = masks.double().sigmoid() * (~pad).unsqueeze(1).double()
    weighted = probabilities * mass.double().unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    overlap = intersection / (areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection).clamp_min(1e-6)
    diagonal = torch.eye(query.shape[1], device=query.device, dtype=torch.bool).unsqueeze(0)
    overlap = overlap.masked_fill(diagonal, 0.)
    competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    peer = torch.einsum('bij,bjd->bid', competition, normalized.double())
    reference_evidence = torch.cat((normalized.double(), peer, normalized.double() - peer,
                                   overlap.max(-1).values.unsqueeze(-1)), -1)
'''
AFTER = '''    probabilities = masks.sigmoid() * (~pad).unsqueeze(1).to(masks.dtype)
    weighted = probabilities * mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    overlap = intersection / (areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection).clamp_min(1e-6)
    diagonal = torch.eye(query.shape[1], device=query.device, dtype=torch.bool).unsqueeze(0)
    overlap = overlap.masked_fill(diagonal, 0.)
    competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    peer = torch.einsum('bij,bjd->bid', competition, normalized)
    reference_evidence = torch.cat((normalized, peer, normalized - peer,
                                   overlap.max(-1).values.unsqueeze(-1)), -1).double()
'''


def repair(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y89')
    assert candidate['stage'] == 'implementation' and not candidate.get('code_commit')
    source = ROOT / 'scripts/y89_observer_core.py'
    original = source.read_text(encoding='utf-8')
    assert original.count(BEFORE) == 1
    diagnosis = ROOT / ('pipeline/analyses/y89_reference_diagnosis_' + token + '.json')
    value = tick.load(diagnosis)
    assert value['original_guard_repeated_unchanged'] and not value['original_guard_passed']
    assert value['same_dtype_reference_exact'] and value['same_dtype_max_error'] == 0.
    assert value['model_revision'] is False and value['guard_relaxed'] is False
    cpu_old = ROOT / ('pipeline/checks/y89_cpu_' + token + '_program.py')
    native_old = closure.artifact(ROOT, candidate['prepared_native_body'])
    qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
    models = {str(path): ref(path) for path in __import__('prepare_y89_stage_evidence_norm').files()}
    core = ROOT / 'scripts/y89_observer_core_v2.py'
    cpu = ROOT / ('pipeline/checks/y89_cpu_reference2_' + token + '_program.py')
    native = ROOT / ('pipeline/checks/y89_native_reference2_' + token + '_body.py')
    record = ROOT / ('pipeline/checks/y89_reference_repair_' + token + '.json')
    assert not any(path.exists() for path in (core, cpu, native, record))
    for old, new in ((source, core), (cpu_old, cpu), (native_old, native)):
        text = old.read_text(encoding='utf-8')
        assert text.count(BEFORE) == 1
        text = text.replace(BEFORE, AFTER)
        assert text.replace(AFTER, BEFORE) == old.read_text(encoding='utf-8')
        ast.parse(text, feature_version=(3, 8))
        new.write_text(text, encoding='utf-8', newline='\n')
    tick.save(record, dict(complete=True,id='AdaRES-Y89',token=token,
        recorded_at=datetime.now().astimezone().isoformat(),diagnosis=ref(diagnosis),
        original_CPU_program=ref(cpu_old),original_CPU_failure=ref('pipeline/analyses/y89_cpu_' + token + '.log'),
        original_core=ref(source),original_native_body=ref(native_old),updated_CPU_program=ref(cpu),
        updated_core=ref(core),updated_native_body=ref(native),exact_reversible_reference_dtype_fix=True,
        tolerance_unchanged=dict(rtol=1e-5,atol=1e-6),
        float64_LayerNorm_reference_unchanged=True,model_and_YAML_revision=False,
        original_failure_preserved=True,native_executed=False,implementation_test_passed=False,
        reason='The original observer recomputed sigmoid/overlap in float64 before comparing float32 evidence. '
               'Actual same-dtype reference is bitwise equal. Reference now follows original float32 operator '
               'semantics before conversion for comparison; tolerance and every positive-response/gradient/consumer guard unchanged.',
        helper=ref(__file__)))
    candidate.update(CPU_program=ref(cpu),prepared_native_body=ref(native),
        observer_core=ref(core),observer_reference_repair=ref(record))
    candidate['evidence'].extend([ref(record),ref(diagnosis),ref(diagnosis.with_suffix('.log')),
        ref(cpu),ref(native),ref(core),ref(__file__)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert models == {str(path): ref(path) for path in __import__('prepare_y89_stage_evidence_norm').files()}
    assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
    print(json.dumps(dict(repair=ref(record),program=ref(cpu),model_revision=False,tolerance_unchanged=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        repair(args.token)
    finally:
        writer.close()
