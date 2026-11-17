import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime


MODEL_IMPORTS = {
    'model_base': 'cfnet.model_base',
    'model': 'cfnet.model',
    'model_vis': 'cfnet.model_vis',
}


def ensure_cfnet_package():
    if 'cfnet' in sys.modules:
        return

    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cfnet'))
    init_file = os.path.join(package_dir, '__init__.py')
    spec = importlib.util.spec_from_loader('cfnet', loader=None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [package_dir]
    module.__file__ = init_file
    sys.modules['cfnet'] = module


def parse_args():
    parser = argparse.ArgumentParser('CFNet single-model inference benchmark')
    parser.add_argument('config', nargs='?', type=str, help='path to config file')
    parser.add_argument('--variant', type=str, choices=sorted(MODEL_IMPORTS.keys()))
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--warmup-iters', type=int, default=5, help='number of warmup iterations')
    parser.add_argument(
        '--measure-iters',
        type=int,
        default=50,
        help='number of timed iterations, 0 means full dataloader after warmup',
    )
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gpu_ids', type=int, default=[0, 1], nargs='+', help='ids of gpus to use')
    parser.add_argument('--save-json', type=str, default=None, help='path to save timing result json')
    parser.add_argument('--log-file', type=str, default=None, help='path to save timing log')
    parser.add_argument('--disable-tqdm', action='store_true', help='disable progress bar')
    parser.add_argument(
        '--compare',
        type=str,
        nargs=2,
        metavar=('RESULT_A', 'RESULT_B'),
        help='compare two saved benchmark json files',
    )
    args = parser.parse_args()

    if args.compare is None:
        if not args.config:
            parser.error('config is required unless --compare is used')
        if not args.variant:
            parser.error('--variant is required unless --compare is used')
        if not args.checkpoint:
            parser.error('--checkpoint is required unless --compare is used')

    return args


def sync_cuda():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bytes_to_mb(num_bytes):
    return num_bytes / (1024.0 * 1024.0)


def kb_to_mb(num_kb):
    return num_kb / 1024.0


def read_proc_status_kb(field_name):
    pattern = re.compile(rf'^{field_name}:\s+(\d+)\s+kB$')
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    return int(match.group(1))
    except FileNotFoundError:
        return None
    return None


def get_process_memory_stats():
    rss_kb = read_proc_status_kb('VmRSS')
    hwm_kb = read_proc_status_kb('VmHWM')
    stats = {}
    if rss_kb is not None:
        stats['process_rss_mb'] = kb_to_mb(rss_kb)
    if hwm_kb is not None:
        stats['process_peak_rss_mb'] = kb_to_mb(hwm_kb)
    return stats


def reset_cuda_peak_stats():
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_cuda_memory_stats():
    import torch

    if not torch.cuda.is_available():
        return {}

    return {
        'cuda_memory_allocated_mb': bytes_to_mb(torch.cuda.memory_allocated()),
        'cuda_memory_reserved_mb': bytes_to_mb(torch.cuda.memory_reserved()),
        'cuda_peak_memory_allocated_mb': bytes_to_mb(torch.cuda.max_memory_allocated()),
        'cuda_peak_memory_reserved_mb': bytes_to_mb(torch.cuda.max_memory_reserved()),
    }


def to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, 'item') and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_default_paths(variant):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('benchmark_logs', variant)
    return (
        os.path.join(base_dir, f'{timestamp}.log'),
        os.path.join(base_dir, f'{timestamp}.json'),
    )


def build_logger(log_file):
    ensure_cfnet_package()
    from cfnet.utils import get_root_logger_val

    if log_file:
        ensure_parent(log_file)
    return get_root_logger_val(log_file=log_file)


def log_line(message, logger=None):
    print(message)
    if logger is not None:
        logger.info(message)


def build_val_dataloader(cfg, logger):
    ensure_cfnet_package()
    from cfnet.dataset import build_dataloader, build_dataset

    dataset = build_dataset(cfg.data.val, logger)
    return build_dataloader(dataset, training=False, **cfg.dataloader.val)


def load_model(cfg, variant, checkpoint, logger):
    import gorilla

    ensure_cfnet_package()
    module = importlib.import_module(MODEL_IMPORTS[variant])
    model_cls = getattr(module, 'CFNet')

    sync_cuda()
    build_start = time.perf_counter()
    model = model_cls(**cfg.model).cuda()
    sync_cuda()
    build_time = time.perf_counter() - build_start

    sync_cuda()
    load_start = time.perf_counter()
    meta = gorilla.load_checkpoint(model, checkpoint, strict=False)
    sync_cuda()
    load_time = time.perf_counter() - load_start

    log_line(
        f'[{variant}] checkpoint={checkpoint}, build_time={build_time:.4f}s, load_time={load_time:.4f}s',
        logger,
    )
    model.eval()
    return model, meta, build_time, load_time


def summarize_times(times):
    times_ms = [value * 1000.0 for value in times]
    sorted_times = sorted(times_ms)
    total_s = sum(times)
    return {
        'iterations': len(times),
        'mean_ms': sum(times_ms) / len(times_ms),
        'median_ms': sorted_times[len(sorted_times) // 2],
        'p90_ms': sorted_times[max(int(len(sorted_times) * 0.9) - 1, 0)],
        'min_ms': sorted_times[0],
        'max_ms': sorted_times[-1],
        'total_s': total_s,
        'throughput_samples_per_s': len(times) / total_s if total_s > 0 else 0.0,
    }


def benchmark_variant(model, dataloader, warmup_iters, measure_iters, show_progress=True):
    import torch
    from tqdm import tqdm

    total_target = len(dataloader) if measure_iters == 0 else warmup_iters + measure_iters
    timings = []
    iterator = dataloader
    if show_progress:
        iterator = tqdm(
            dataloader,
            total=min(len(dataloader), total_target),
            desc='Benchmark',
            bar_format='{l_bar}{bar:20}{r_bar}',
        )

    reset_cuda_peak_stats()
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            if measure_iters != 0 and batch_idx >= total_target:
                break

            sync_cuda()
            start = time.perf_counter()
            _ = model(batch, mode='predict')
            sync_cuda()
            elapsed = time.perf_counter() - start

            if batch_idx >= warmup_iters:
                timings.append(elapsed)

    if show_progress and hasattr(iterator, 'close'):
        iterator.close()

    if not timings:
        raise RuntimeError('No timed iterations were collected, please reduce warmup-iters.')

    stats = summarize_times(timings)
    stats.update(get_process_memory_stats())
    stats.update(get_cuda_memory_stats())
    return stats


def print_benchmark_summary(result, logger=None):
    lines = [
        '=' * 100,
        f'Variant: {result["variant"]}',
        f'Checkpoint: {result["checkpoint"]}',
        f'Build time: {result["build_time_s"]:.4f} s',
        f'Checkpoint load time: {result["checkpoint_load_time_s"]:.4f} s',
        f'Timed iterations: {result["iterations"]}',
        f'Mean latency: {result["mean_ms"]:.3f} ms',
        f'Median latency: {result["median_ms"]:.3f} ms',
        f'P90 latency: {result["p90_ms"]:.3f} ms',
        f'Min latency: {result["min_ms"]:.3f} ms',
        f'Max latency: {result["max_ms"]:.3f} ms',
        f'Total timed forward time: {result["total_s"]:.4f} s',
        f'Throughput: {result["throughput_samples_per_s"]:.3f} samples/s',
    ]
    if 'process_rss_mb' in result:
        lines.append(f'Process RSS: {result["process_rss_mb"]:.2f} MB')
    if 'process_peak_rss_mb' in result:
        lines.append(f'Process peak RSS: {result["process_peak_rss_mb"]:.2f} MB')
    if 'cuda_memory_allocated_mb' in result:
        lines.append(f'CUDA allocated: {result["cuda_memory_allocated_mb"]:.2f} MB')
    if 'cuda_memory_reserved_mb' in result:
        lines.append(f'CUDA reserved: {result["cuda_memory_reserved_mb"]:.2f} MB')
    if 'cuda_peak_memory_allocated_mb' in result:
        lines.append(f'CUDA peak allocated: {result["cuda_peak_memory_allocated_mb"]:.2f} MB')
    if 'cuda_peak_memory_reserved_mb' in result:
        lines.append(f'CUDA peak reserved: {result["cuda_peak_memory_reserved_mb"]:.2f} MB')
    for line in lines:
        log_line(line, logger)


def save_json(save_path, result):
    ensure_parent(save_path)
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def compare_results(result_a, result_b):
    mean_delta = result_b['mean_ms'] - result_a['mean_ms']
    mean_ratio = result_b['mean_ms'] / result_a['mean_ms'] if result_a['mean_ms'] > 0 else float('inf')
    throughput_delta = result_b['throughput_samples_per_s'] - result_a['throughput_samples_per_s']
    throughput_ratio = (
        result_b['throughput_samples_per_s'] / result_a['throughput_samples_per_s']
        if result_a['throughput_samples_per_s'] > 0
        else float('inf')
    )
    return {
        'result_a': result_a['variant'],
        'result_b': result_b['variant'],
        'mean_ms_a': result_a['mean_ms'],
        'mean_ms_b': result_b['mean_ms'],
        'mean_delta_ms': mean_delta,
        'mean_ratio_b_over_a': mean_ratio,
        'throughput_a': result_a['throughput_samples_per_s'],
        'throughput_b': result_b['throughput_samples_per_s'],
        'throughput_delta': throughput_delta,
        'throughput_ratio_b_over_a': throughput_ratio,
        'process_peak_rss_mb_a': result_a.get('process_peak_rss_mb'),
        'process_peak_rss_mb_b': result_b.get('process_peak_rss_mb'),
        'cuda_peak_memory_allocated_mb_a': result_a.get('cuda_peak_memory_allocated_mb'),
        'cuda_peak_memory_allocated_mb_b': result_b.get('cuda_peak_memory_allocated_mb'),
    }


def print_comparison_summary(result_a, result_b):
    comparison = compare_results(result_a, result_b)
    print('=' * 100)
    print(f'A: {result_a["variant"]} | {result_a["checkpoint"]}')
    print(f'B: {result_b["variant"]} | {result_b["checkpoint"]}')
    print(f'A mean latency: {comparison["mean_ms_a"]:.3f} ms')
    print(f'B mean latency: {comparison["mean_ms_b"]:.3f} ms')
    print(f'Latency delta (B-A): {comparison["mean_delta_ms"]:+.3f} ms')
    print(f'Latency ratio (B/A): x{comparison["mean_ratio_b_over_a"]:.3f}')
    print(f'A throughput: {comparison["throughput_a"]:.3f} samples/s')
    print(f'B throughput: {comparison["throughput_b"]:.3f} samples/s')
    print(f'Throughput delta (B-A): {comparison["throughput_delta"]:+.3f} samples/s')
    print(f'Throughput ratio (B/A): x{comparison["throughput_ratio_b_over_a"]:.3f}')
    if comparison["process_peak_rss_mb_a"] is not None and comparison["process_peak_rss_mb_b"] is not None:
        print(f'A process peak RSS: {comparison["process_peak_rss_mb_a"]:.2f} MB')
        print(f'B process peak RSS: {comparison["process_peak_rss_mb_b"]:.2f} MB')
    if comparison["cuda_peak_memory_allocated_mb_a"] is not None and comparison["cuda_peak_memory_allocated_mb_b"] is not None:
        print(f'A CUDA peak allocated: {comparison["cuda_peak_memory_allocated_mb_a"]:.2f} MB')
        print(f'B CUDA peak allocated: {comparison["cuda_peak_memory_allocated_mb_b"]:.2f} MB')


def run_compare(compare_paths):
    result_a = load_json(compare_paths[0])
    result_b = load_json(compare_paths[1])
    print_comparison_summary(result_a, result_b)


def run_benchmark(args):
    import gorilla
    import torch

    log_file = args.log_file
    save_json_path = args.save_json
    if log_file is None or save_json_path is None:
        default_log, default_json = make_default_paths(args.variant)
        log_file = log_file or default_log
        save_json_path = save_json_path or default_json

    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_ids, num_gpu=args.num_gpus)
    torch.backends.cudnn.benchmark = True

    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = build_logger(log_file)

    log_line(f'Saving log to: {log_file}', logger)
    log_line(f'Saving result json to: {save_json_path}', logger)

    dataloader = build_val_dataloader(cfg, logger)
    model, meta, build_time, load_time = load_model(cfg, args.variant, args.checkpoint, logger)
    benchmark_result = benchmark_variant(
        model=model,
        dataloader=dataloader,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        show_progress=not args.disable_tqdm,
    )

    result = {
        'variant': args.variant,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'log_file': log_file,
        'save_json': save_json_path,
        'warmup_iters': args.warmup_iters,
        'measure_iters': benchmark_result['iterations'],
        'build_time_s': build_time,
        'checkpoint_load_time_s': load_time,
        'checkpoint_meta': to_jsonable(meta.get('meta', {})),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    result.update(benchmark_result)

    print_benchmark_summary(result, logger)
    save_json(save_json_path, result)
    log_line(f'Benchmark json saved to: {save_json_path}', logger)


def main():
    args = parse_args()
    if args.compare:
        run_compare(args.compare)
    else:
        run_benchmark(args)


if __name__ == '__main__':
    main()
