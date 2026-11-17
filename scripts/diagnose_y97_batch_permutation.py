"""Repeat the exact failed component guard and preserve its numerical evidence."""
import argparse
import ast
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick


BODY = '''
import ast, base64, hashlib, json
from pathlib import Path
import torch
from torch import nn
assert P['physical_gpu'] is None and not torch.cuda.is_initialized()
torch.set_num_threads(2)
raw_source=base64.b64decode(PAYLOAD['variant']['python_files']['dec.py']['base64'])
assert hashlib.sha256(raw_source).hexdigest()==PAYLOAD['variant']['python_files']['dec.py']['sha256']
tree=ast.parse(raw_source.decode('utf8'))
node=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='LanguageQuadraticInput')
scope=dict(torch=torch,nn=nn)
exec(compile(ast.Module(body=[node],type_ignores=[]),'original_y97_component','exec'),scope)
Layer=scope['LanguageQuadraticInput']
rows=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999); rng=torch.get_rng_state().clone()
    layer=Layer(11,7).to(dtype=dtype)
    assert torch.equal(rng,torch.get_rng_state())
    raw=torch.randn(2,5,11,dtype=dtype,requires_grad=True)
    projected=torch.randn(2,5,7,dtype=dtype,requires_grad=True)
    anchor=layer(projected,raw)
    assert torch.equal(anchor,projected)
    torch.autograd.grad(anchor.sum(),(projected,raw,layer.weight))
    with torch.no_grad():
        layer.weight.copy_((torch.arange(77,dtype=dtype).reshape(11,7)-38)/77)
        value=layer(projected,raw)
        observed=layer(projected.flip(0),raw.flip(0))
        expected=value.flip(0)
        refused=False
        try:
            assert torch.equal(observed,expected)
        except AssertionError:
            refused=True
        error=(observed-expected).abs()
        index=tuple(int(v) for v in torch.nonzero(error==error.max())[0])
        tolerance=128*torch.finfo(dtype).eps
        allowance=tolerance+tolerance*expected.abs()
        exact_element_count=int((error==0).sum())
        scalar=torch.empty_like(value,dtype=torch.float64)
        for b in range(2):
            for i in range(5):
                for d in range(7):
                    scalar[b,i,d]=projected[b,i,d].double()+sum(
                        raw[b,i,k].double()**2*layer.weight[k,d].double() for k in range(11))
        scalar_error=(value.double()-scalar).abs()
        rows.append(dict(dtype=str(dtype),original_bitwise_guard_executed=True,
            original_bitwise_guard_failed=refused,exact_elements=exact_element_count,elements=value.numel(),
            max_absolute_difference=float(error.max()),max_error_index=list(index),
            actual_at_max=float(observed[index]),expected_at_max=float(expected[index]),
            epsilon=float(torch.finfo(dtype).eps),existing_scalar_control_rtol=tolerance,
            existing_scalar_control_atol=tolerance,existing_scalar_limit_at_max=float(allowance[index]),
            within_existing_scalar_tolerance=bool((error<=allowance).all()),
            independent_scalar_reference_max_error=float(scalar_error.max()),
            raw_shape=list(raw.shape),projected_shape=list(projected.shape)))
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y97',torch_version=str(torch.__version__),
    original_CPU_program_sha256=ORIGINAL_PROGRAM_SHA,original_log_sha256=ORIGINAL_LOG_SHA,
    original_model_source_sha256=hashlib.sha256(raw_source).hexdigest(),rows=rows,
    CPU_component_diagnostic_only=True,original_guard_unchanged=True,
    original_float32_failure_reproduced=rows[0]['original_bitwise_guard_failed'],
    model_or_YAML_modified=False,full_model_or_data_replayed=False,
    CUDA_initialized=bool(torch.cuda.is_initialized()),native_smoke_passed=False,admission=False,
    interpretation='Exact original bitwise failure is preserved. Existing scalar tolerance is descriptive evidence, '
        'not a replacement PASS, relaxed guard, repaired observer, trained harm or mechanism rejection.')),encoding='utf8')
'''


def prepare(token):
    lease(token, 600)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        original = ROOT / ("pipeline/checks/y97_cpu_" + token + "_program.py")
        log = ROOT / ("pipeline/analyses/y97_cpu_" + token + ".log")
        assert original.exists() and log.exists()
        assert not log.with_suffix(".json").exists()
        payload = ast.literal_eval(ast.parse(original.read_text(encoding="utf-8")).body[0].value)
        assert payload["variant"]["id"] == "AdaRES-Y97"
        source = ROOT / ("pipeline/checks/y97_batch_permutation_diagnosis_" + token + "_program.py")
        assert not source.exists()
        text = "PAYLOAD=" + repr(payload) + "\nORIGINAL_PROGRAM_SHA=" + repr(ref(original)["sha256"])
        text += "\nORIGINAL_LOG_SHA=" + repr(ref(log)["sha256"]) + "\n" + BODY
        ast.parse(text, feature_version=(3, 8))
        source.write_text(text, encoding="utf-8", newline="\n")
        work = tick.load(ROOT / "pipeline/research_work.json")
        candidate = next(row for row in work["candidates"] if row["id"] == "AdaRES-Y97")
        candidate["evidence"].extend([ref(original), ref(log), ref(source), ref(__file__)])
        tick.save(ROOT / "pipeline/research_work.json", work)
        print(json.dumps(dict(program=ref(source), original_program=ref(original), failed_log=ref(log))))
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    prepare(parser.parse_args().token)
