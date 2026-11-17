"""Reproduce the original CPU guard unchanged and record same-dtype arithmetic."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

DIAG = '''try:
        torch.testing.assert_allclose(args[0].double(), reference_evidence, rtol=1e-5, atol=1e-6)
    except AssertionError:
        p32 = masks.sigmoid() * (~pad).unsqueeze(1).to(masks.dtype)
        w32 = p32 * mass.unsqueeze(1)
        a32 = w32.sum(-1)
        i32 = torch.einsum('bim,bjm->bij', w32, p32)
        o32 = i32 / (a32.unsqueeze(-1) + a32.unsqueeze(1) - i32).clamp_min(1e-6)
        o32 = o32.masked_fill(diagonal, 0.)
        c32 = o32 / o32.sum(-1, keepdim=True).clamp_min(1e-6)
        peer32 = torch.einsum('bij,bjd->bid', c32, normalized)
        typed = torch.cat((normalized, peer32, normalized - peer32,
                           o32.max(-1).values.unsqueeze(-1)), -1)
        delta = (args[0].double() - reference_evidence).abs()
        limit = 1e-6 + 1e-5 * reference_evidence.abs()
        bad = delta > limit
        where = torch.nonzero(bad)
        Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],
            diagnosis_only=True,implementation_test_passed=False,native_smoke_passed=False,
            original_guard_repeated_unchanged=True,original_guard_passed=False,
            step=len(step_rows),stage=index,original_tolerance=dict(rtol=1e-5,atol=1e-6),
            mismatched_elements=int(bad.sum()),elements=delta.numel(),
            max_absolute_error=float(delta.max()),same_dtype_reference_exact=torch.equal(args[0],typed),
            same_dtype_max_error=float((args[0]-typed).abs().max()),
            errors=[dict(index=position.tolist(),actual=float(args[0][tuple(position)]),
                         float64_reference=float(reference_evidence[tuple(position)]),
                         same_dtype_reference=float(typed[tuple(position)]),
                         limit=float(limit[tuple(position)])) for position in where[:8]],
            CUDA_initialized=torch.cuda.is_initialized(),model_revision=False,guard_relaxed=False)),
            encoding='utf8')
        raise'''


def create(token):
    source = ROOT / ('pipeline/checks/y89_cpu_' + token + '_program.py')
    original = source.read_text(encoding='utf-8')
    anchor = "torch.testing.assert_allclose(args[0].double(), reference_evidence, rtol=1e-5, atol=1e-6)"
    assert original.count(anchor) == 1
    program = original.replace(anchor, DIAG)
    # The failed assertion still runs and raises; only its exception is logged.
    wrapper = "try:\n    exec(" + repr(program) + ",globals())\nexcept AssertionError:\n    assert Path(P['output']).is_file()\n"
    ast.parse(wrapper, feature_version=(3, 8))
    output = ROOT / ('pipeline/checks/y89_reference_diagnosis_' + token + '_program.py')
    assert not output.exists()
    output.write_text(wrapper, encoding='utf-8', newline='\n')
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y89')
    candidate['evidence'].extend([ref(source), ref('pipeline/analyses/y89_cpu_' + token + '.log'),
                                 ref(output), ref(__file__)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(program=ref(output),original_program=ref(source),guard_relaxed=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 240)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        create(args.token)
    finally:
        writer.close()
