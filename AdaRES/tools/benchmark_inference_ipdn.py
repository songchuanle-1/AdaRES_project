"""
IPDN multi-variant inference benchmark.
Each variant runs in a subprocess (clean Python process per variant).
Before each subprocess, ipdn/__init__.py is temporarily patched to import
the correct model variant, then restored afterwards.

Each variant is benchmarked multiple times (default 5). Results reported as mean ± std.

Usage:
    python tools/benchmark_inference_ipdn.py --variants all --gpu_ids 1
    python tools/benchmark_inference_ipdn.py --variants cf12_4_0,cf12_4_1_11_0 --gpu_ids 1 --num-runs 3
"""
import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
CHECKPOINT_BASE = '/bak_home/songcl/3D_visual_grounding/CFNet_checkpoints'
INIT_FILE = '/bak_home/songcl/3D_visual_grounding/CFNet/ipdn/__init__.py'

IPDN_VARIANTS = {

    'cf12_4_1_6_0_new1': {
        'model_import_line': 'from ipdn.model_paper5.cf12_4_1_6_0_new1 import IPDN      ##  ## cf12_4_1_6_0  base  最开始第一层有attmask',
        'config': 'configs_song/model_paper5/cf12_4_1_6_new1.yaml',
        'checkpoint': f'{CHECKPOINT_BASE}/cf12_4_1_6_new1/20251004_215512/best0.pth',
        'desc': 'base (attmask in 1st layer)',
    },
    'cf12_4_1_10_0': {
        'model_import_line': 'from ipdn.model_paper5.cf12_4_1_10_0 import IPDN      ## cf12_4_1  base+1  494',
        'config': 'configs_song/model_paper5/cf12_4_1_10_0.yaml',
        'checkpoint': f'{CHECKPOINT_BASE}/cf12_4_1_10/20250902_204121/best0.pth',
        'desc': 'base+1 (494 mIoU)',
    },
    'cf12_4_1_11_0': {
        'model_import_line': 'from ipdn.model_paper5.cf12_4_1_11_0 import IPDN      ## cf12_4_1  base+2   502',
        'config': 'configs_song/model_paper5/cf12_4_1_11_0.yaml',
        'checkpoint': f'{CHECKPOINT_BASE}/cf12_4_1_11/20250902_204714/best0.pth',
        'desc': 'base+2 (502 mIoU)',
    },
    'cf12_4_1_5_0': {
        'model_import_line': 'from ipdn.model_paper5.cf12_4_1_5_0 import IPDN      ## cf12_4_1  base+3  # 493',
        'config': 'configs_song/model_paper5/cf12_4_1_5_0.yaml',
        'checkpoint': f'{CHECKPOINT_BASE}/cf12_4_1_5/20250828_041836/best0.pth',
        'desc': 'base+3 (493 mIoU)',
    },
    'cf12_4_0': {
        'model_import_line': 'from ipdn.model_paper5.cf12_4_2 import IPDN      ## cf12_4_0 修改脚本  base_123  ## gres good    gres   没有冗余的参数  yaml _0',
        'config': 'configs_song/model_paper5/cf12_4_2.yaml',
        'checkpoint': f'{CHECKPOINT_BASE}/cf12_4_1/20250819_212224/best0.pth',
        'desc': 'cf12_4_2 base_123 (514 mIoU)',
    },
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser('IPDN multi-variant inference benchmark')
    parser.add_argument('--variants', type=str, default='all')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--gpu_ids', type=int, default=[0], nargs='+')
    parser.add_argument('--warmup-iters', type=int, default=5)
    parser.add_argument('--measure-iters', type=int, default=50)
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--save-json', type=str, default=None)
    args = parser.parse_args()
    return args


def resolve_variants(names_str):
    if names_str.strip().lower() == 'all':
        return list(IPDN_VARIANTS.keys())
    selected = []
    for n in names_str.split(','):
        n = n.strip()
        if n not in IPDN_VARIANTS:
            print(f'Warning: unknown variant "{n}", skipping. '
                  f'Available: {list(IPDN_VARIANTS.keys())}')
            continue
        selected.append(n)
    return selected


# ---------------------------------------------------------------------------
# Subprocess worker script
# ---------------------------------------------------------------------------
WORKER_SCRIPT = '''
import json, os, sys, time
import gorilla, torch
import numpy as np

cfg_path      = sys.argv[1]
checkpoint    = sys.argv[2]
warmup        = int(sys.argv[3])
measure       = int(sys.argv[4])
gpu_ids       = json.loads(sys.argv[5])
num_gpus      = int(sys.argv[6])

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def bytes_to_mb(b):
    return b / (1024.0 * 1024.0)

def summarize_times(times):
    times_ms = [v * 1000.0 for v in times]
    srt = sorted(times_ms)
    total_s = sum(times)
    n = len(times)
    return {
        "iterations": n,
        "mean_ms": sum(times_ms) / n,
        "std_ms": float(np.std(times_ms, ddof=1)) if n > 1 else 0.0,
        "median_ms": srt[n // 2],
        "p90_ms": srt[max(int(n * 0.9) - 1, 0)],
        "min_ms": srt[0],
        "max_ms": srt[-1],
        "total_s": total_s,
        "throughput_samples_per_s": n / total_s if total_s > 0 else 0.0,
    }

gorilla.set_cuda_visible_devices(gpu_ids=gpu_ids, num_gpu=num_gpus)
torch.backends.cudnn.benchmark = True

cfg = gorilla.Config.fromfile(cfg_path)
gorilla.set_random_seed(cfg.test.seed)

# ipdn/__init__.py has already been patched by the orchestrator to import
# the correct model variant
from ipdn import IPDN
from ipdn.dataset import build_dataloader, build_dataset
from ipdn.utils import get_root_logger_val

logger = get_root_logger_val(log_file=None)
dataset = build_dataset(cfg.data.val, logger)
dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.val)

runs = []
for run_idx in range(int(sys.argv[7])):
    sync_cuda(); t0 = time.perf_counter()
    model = IPDN(**cfg.model).cuda()
    sync_cuda(); bt = time.perf_counter() - t0

    sync_cuda(); t0 = time.perf_counter()
    gorilla.load_checkpoint(model, checkpoint, strict=False)
    sync_cuda(); lt = time.perf_counter() - t0
    model.eval()

    total_target = len(dataloader) if measure == 0 else warmup + measure
    timings = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if measure != 0 and idx >= total_target:
                break
            sync_cuda(); t0 = time.perf_counter()
            _ = model(batch, mode="predict")
            sync_cuda(); t1 = time.perf_counter()
            if idx >= warmup:
                timings.append(t1 - t0)

    if not timings:
        raise RuntimeError("No timed iterations")

    stats = summarize_times(timings)
    stats["build_s"] = bt
    stats["load_s"] = lt
    stats["cuda_peak_alloc_mb"] = bytes_to_mb(torch.cuda.max_memory_allocated())
    # CPU RSS from /proc/self/status
    import re
    try:
        with open('/proc/self/status', 'r') as f:
            status = f.read()
        r = re.search(r'VmRSS:\s+(\d+)\s+kB', status)
        h = re.search(r'VmHWM:\s+(\d+)\s+kB', status)
        if r: stats['rss_mb'] = int(r.group(1)) / 1024.0
        if h: stats['peak_rss_mb'] = int(h.group(1)) / 1024.0
    except:
        pass
    runs.append(stats)
    del model; torch.cuda.empty_cache()

print(json.dumps(runs))
'''

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
PYTHON_BIN = '/bak_home/songcl/3D_visual_grounding/BQNet_env/BQNet/bin/python'
PROJECT_ROOT = '/bak_home/songcl/3D_visual_grounding/CFNet'
LIB_PATH = '/bak_home/songcl/3D_visual_grounding/CFNet/ipdn/lib'

ENV = os.environ.copy()
ENV['PYTHONPATH'] = f'{PROJECT_ROOT}:{LIB_PATH}:{ENV.get("PYTHONPATH", "")}'


def patch_init_file(import_line):
    """Rewrite ipdn/__init__.py so only the target model import is active."""
    with open(INIT_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep lines that are NOT model imports, AND the target import line
        if stripped.startswith('# from ipdn.model_paper5.') or stripped.startswith('from ipdn.model_paper5.'):
            # This is a model import line. Keep only if it matches the target.
            if stripped == import_line.strip() or stripped.lstrip('# ') == import_line.strip():
                # Uncomment it if it was commented
                new_lines.append(import_line + '\n')
            else:
                # Ensure it's commented out
                if stripped.startswith('#'):
                    new_lines.append(line)  # already commented, keep as-is
                else:
                    new_lines.append('# ' + line)
        else:
            new_lines.append(line)

    with open(INIT_FILE, 'w') as f:
        f.writelines(new_lines)


def restore_init_file(original_lines):
    """Restore the original ipdn/__init__.py content."""
    with open(INIT_FILE, 'w') as f:
        f.writelines(original_lines)


def run_variant(name, info, args):
    """Run all passes for a single variant in a subprocess."""
    gpu_ids_json = json.dumps(args.gpu_ids)
    cmd = [
        PYTHON_BIN, '-c', WORKER_SCRIPT,
        info['config'], info['checkpoint'],
        str(args.warmup_iters), str(args.measure_iters),
        gpu_ids_json, str(args.num_gpus),
        str(args.num_runs),
    ]
    print(f'  Launching subprocess for {name} ({args.num_runs} runs)...')
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=PROJECT_ROOT, env=ENV, timeout=3600)
    if result.returncode != 0:
        print(f'  STDERR: {result.stderr[-2000:]}')
        raise RuntimeError(f'Subprocess failed with code {result.returncode}')

    # stdout may have tqdm mixed in — find the final JSON line
    lines = result.stdout.strip().split('\n')
    json_line = None
    for line in reversed(lines):
        if line.strip().startswith('['):
            json_line = line.strip()
            break
    if json_line is None:
        print(f'  STDOUT tail: {"".join(lines[-20:])}')
        raise RuntimeError('No JSON output found in subprocess stdout')
    return json.loads(json_line)


def run_all(variant_names, args):
    # Save original __init__.py
    with open(INIT_FILE, 'r') as f:
        original_lines = f.readlines()

    all_results = []
    try:
        for name in variant_names:
            info = IPDN_VARIANTS[name]
            print(f'\n{"=" * 70}')
            print(f'Variant: {name}  ({info["desc"]})')
            print(f'Config:   {info["config"]}')
            print(f'Ckpt:     {info["checkpoint"]}')
            print(f'{"=" * 70}')

            # ---- Patch __init__.py to import this variant ----
            patch_init_file(info['model_import_line'])

            try:
                run_data = run_variant(name, info, args)
            finally:
                restore_init_file(original_lines)

            build_times = [r['build_s'] for r in run_data]
            load_times = [r['load_s'] for r in run_data]
            mean_lats = [r['mean_ms'] for r in run_data]
            throughputs = [r['throughput_samples_per_s'] for r in run_data]
            cuda_peaks = [r.get('cuda_peak_alloc_mb', 0) for r in run_data]
            rss_peaks  = [r.get('peak_rss_mb', 0) for r in run_data]

            for i, r in enumerate(run_data):
                print(f'  Run {i + 1}/{args.num_runs}: '
                      f'mean={r["mean_ms"]:.2f} ms  '
                      f'throughput={r["throughput_samples_per_s"]:.2f} samp/s  '
                      f'CUDA pk={r.get("cuda_peak_alloc_mb", 0):.1f} MB  '
                      f'RSS pk={r.get("peak_rss_mb", 0):.1f} MB')

            def _agg(arr):
                a = np.array(arr, dtype=np.float64)
                return {
                    'mean': float(np.mean(a)),
                    'std': float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                    'min': float(np.min(a)),
                    'max': float(np.max(a)),
                    'values': [float(x) for x in a],
                }

            result = {
                'variant': name,
                'description': info['desc'],
                'config': info['config'],
                'checkpoint': info['checkpoint'],
                'num_runs': args.num_runs,
                'warmup_iters': args.warmup_iters,
                'measure_iters_per_run': run_data[0]['iterations'],
                'build_time_s': _agg(build_times),
                'checkpoint_load_time_s': _agg(load_times),
                'aggregated': {
                    'mean_ms': _agg(mean_lats),
                    'throughput_samples_per_s': _agg(throughputs),
                    'cuda_peak_memory_allocated_mb': _agg(cuda_peaks),
                    'peak_rss_mb': _agg(rss_peaks),
                },
                'per_run': run_data,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            all_results.append(result)
            print_summary(result)
    finally:
        restore_init_file(original_lines)

    return all_results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def fmt_mean_std(entry):
    if not entry:
        return 'N/A'
    return f'{entry["mean"]:.2f}±{entry["std"]:.2f}'


def print_summary(r):
    agg = r['aggregated']
    print(f'\n  --- {r["variant"]} ({r["description"]}) ---')
    print(f'  Build time:     {fmt_mean_std(r["build_time_s"])} s')
    print(f'  Ckpt load time: {fmt_mean_std(r["checkpoint_load_time_s"])} s')
    print(f'  Mean latency:   {fmt_mean_std(agg["mean_ms"])} ms')
    print(f'  Throughput:     {fmt_mean_std(agg["throughput_samples_per_s"])} samples/s')
    if 'cuda_peak_memory_allocated_mb' in agg:
        print(f'  CUDA peak alloc:{fmt_mean_std(agg["cuda_peak_memory_allocated_mb"])} MB')
    if 'peak_rss_mb' in agg:
        print(f'  RSS peak:       {fmt_mean_std(agg["peak_rss_mb"])} MB')
    print()


def print_comparison_table(results):
    print(f'\n{"=" * 140}')
    print(f'{"Variant":<22} {"Latency(ms)":>22} {"Throughput/s":>18} {"CUDA pk(MB)":>16} {"RSS pk(MB)":>14}')
    print(f'{"-" * 140}')
    for r in results:
        agg = r['aggregated']
        lat = fmt_mean_std(agg['mean_ms'])
        tp = fmt_mean_std(agg['throughput_samples_per_s'])
        cuda = fmt_mean_std(agg.get('cuda_peak_memory_allocated_mb', {}))
        rss = fmt_mean_std(agg.get('peak_rss_mb', {}))
        print(f'{r["variant"]:<22} {lat:>22} {tp:>18} {cuda:>16} {rss:>14}')
    print(f'{"=" * 140}')

    if len(results) > 1:
        fastest = min(results, key=lambda x: x['aggregated']['mean_ms']['mean'])
        baseline_mean = fastest['aggregated']['mean_ms']['mean']
        baseline_std = fastest['aggregated']['mean_ms']['std']

        print(f'\nBaseline: {fastest["variant"]} '
              f'({baseline_mean:.2f}±{baseline_std:.2f} ms)')
        print(f'{"Variant":<22} {"Latency(ms)":>22} {"Δ(ms)":>18} {"Ratio":>8} {"Slower":>10}')
        print(f'{"-" * 80}')

        for r in sorted(results, key=lambda x: x['aggregated']['mean_ms']['mean']):
            m = r['aggregated']['mean_ms']['mean']
            s = r['aggregated']['mean_ms']['std']
            delta = m - baseline_mean
            delta_std = math.sqrt(s ** 2 + baseline_std ** 2)
            ratio = m / baseline_mean if baseline_mean > 0 else float('inf')
            pct = (ratio - 1) * 100
            marker = '  <-- FASTEST' if r is fastest else ''
            print(f'{r["variant"]:<22} {m:.2f}±{s:.2f} ms{"":>5} '
                  f'{delta:+.2f}±{delta_std:.2f}{"":>4} '
                  f'{ratio:.3f}x  {pct:+.1f}%{marker}')
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    variant_names = resolve_variants(args.variants)
    if not variant_names:
        print('No valid variants selected.')
        return

    print(f'Benchmarking {len(variant_names)} IPDN variants '
          f'({args.num_runs} runs each): {variant_names}')

    results = run_all(variant_names, args)
    print_comparison_table(results)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or '.', exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'Results saved to: {args.save_json}')


if __name__ == '__main__':
    main()
