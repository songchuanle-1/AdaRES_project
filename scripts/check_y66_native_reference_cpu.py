"""Generate a CPU-only check of the native probe's independent reference."""
import argparse
import ast
from pathlib import Path
from prepare_y60_stage_heads import ROOT, lease, monitor, ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 400)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        body = ROOT / 'scripts/probe_y66_native_smoke.py'
        tree = ast.parse(body.read_text(encoding='utf-8'))
        source = ast.get_source_segment(body.read_text(encoding='utf-8'),
            next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reference_peer_rows'))
        decoder = ROOT / 'AdaRES/ipdn/model_paper5/adares_y66_peer_state_feedback/dec.py'
        decoder_text = decoder.read_text(encoding='utf-8')
        component = ast.get_source_segment(decoder_text,
            next(n for n in ast.parse(decoder_text).body
                 if isinstance(n, ast.ClassDef) and n.name == 'PeerStateFeedback'))
        output = ROOT / ('pipeline/checks/y66_native_reference_cpu_' + args.token + '_program.py')
        assert not output.exists()
        program = """import json
from pathlib import Path
import torch
from torch import nn
torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
""" + source + '\n\n' + component + '\n\n' + '''
cases = []
for dtype in (torch.float32, torch.float64):
    generator = torch.Generator().manual_seed(1999)
    query = torch.randn(2, 128, 5, dtype=dtype, generator=generator)
    masks = torch.randn(2, 128, 7, dtype=dtype, generator=generator)
    padded = torch.tensor([[False] * 5 + [True] * 2, [False] * 7])
    mass = torch.tensor([[1., 3., 2., 4., 5., 777., 999.], [1., 2., 1., 3., 1., 2., 4.]], dtype=dtype)
    module = PeerStateFeedback(5).to(dtype)
    assert torch.equal(module.weight, torch.zeros_like(module.weight))
    with torch.no_grad():
        module.weight.copy_(torch.arange(25, dtype=dtype).reshape(5, 5) / 200.)
    for name in ('ordinary', 'zero_mass', 'all_padding', 'extreme_logits'):
        current_masks = masks * (100. if name == 'extreme_logits' else 1.)
        current_mass = torch.zeros_like(mass) if name == 'zero_mass' else mass
        current_padding = torch.ones_like(padded) if name == 'all_padding' else padded
        context = reference_peer_rows(query, current_masks, current_padding, current_mass)
        # Full scalar loops over every peer and point, avoiding vectorized overlap algebra.
        scalar = torch.zeros(2, 3, 5, dtype=torch.float64)
        probabilities = current_masks.double().sigmoid()
        for batch in range(2):
            for row, own in enumerate((0, 64, 127)):
                overlaps = []
                for peer in range(128):
                    intersection = own_area = peer_area = 0.
                    for point in range(7):
                        if bool(current_padding[batch, point]): continue
                        a = float(probabilities[batch, own, point])
                        b = float(probabilities[batch, peer, point])
                        weight = float(current_mass[batch, point])
                        intersection += a * b * weight
                        own_area += a * weight
                        peer_area += b * weight
                    overlap = 0. if peer == own else intersection / max(own_area + peer_area - intersection, 1e-6)
                    overlaps.append(overlap)
                denominator = max(sum(overlaps), 1e-6)
                for peer, overlap in enumerate(overlaps):
                    scalar[batch, row] += query[batch, peer].double() * (overlap / denominator)
        torch.testing.assert_close(context, scalar, rtol=1e-10, atol=1e-10)
        observed = module(query, current_masks, current_padding, current_mass)[:, [0, 64, 127]].double()
        expected = query[:, [0, 64, 127]].double() + context @ module.weight.detach().double()
        torch.testing.assert_close(observed, expected, rtol=3e-5, atol=2e-6)
        cases.append(dict(dtype=str(dtype), case=name, scalar_reference_passed=True,
                          actual_component_matches_native_reference=True))
assert len(cases) == 8 and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id='AdaRES-Y66',
    native_reference_CPU_checked=True, cases=cases, CUDA_initialized=False,
    native_smoke_passed=False, native_task_backward=False, admission=False,
    limitation='CPU reference/guard preparation only, no native sparse TRAIN steps or efficacy.')),
    encoding='utf-8')
'''
        ast.parse(program, feature_version=(3, 8))
        output.write_text(program, encoding='utf-8', newline='\n')
        print(ref(output))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
