"""Shared collective measurements over the production mesh backend.

Useful output bytes describe the operation. All initialization and validation is untimed.

    python -m torch_mesh.benchmarks CONFIG.json   # every rank: sizes, least_calls, most_calls, bytes_per_size, latency_calls
"""
import json
from pathlib import Path
import statistics
import sys
import time


def _boundary():
    """Measurement coordination outside the measured operation: torch's stream drained and the ranks met."""
    import torch
    import torch.distributed as dist
    torch.mps.synchronize()
    dist.barrier()


def timed(call, calls):
    import torch
    _boundary()
    began = time.perf_counter()
    for _ in range(calls):
        call()
    torch.mps.synchronize()
    return time.perf_counter() - began


def latency(call, calls):
    import torch
    seconds = []
    for _ in range(calls):
        _boundary()
        began = time.perf_counter()
        call()
        torch.mps.synchronize()
        seconds.append(time.perf_counter() - began)
    return statistics.median(seconds)


def bandwidth(config):
    import torch
    import torch.distributed as dist
    import torch_mesh
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        for requested in config['sizes']:
            if requested <= 0 or requested % 2:
                raise ValueError('float16 bandwidth operand bytes must be positive and even')
            n = requested // 2
            per = max(1, n // size)
            a = torch_mesh.empty(n, dtype=torch.half)
            c, part = torch_mesh.empty(per * size, dtype=torch.half), torch_mesh.empty(per, dtype=torch.half)
            calls = min(config['most_calls'], max(config['least_calls'], config['bytes_per_size'] // requested))
            # Idempotent MAX prevents repeated in-place reductions overflowing half precision.
            figures = [('all_reduce_max', lambda: dist.all_reduce(a, op=dist.ReduceOp.MAX), a, n * 2),
                       ('broadcast', lambda: dist.broadcast(a, 0), a, n * 2 if rank else 0),
                       ('all_gather', lambda: dist.all_gather_into_tensor(c, part), c, 2 * per * (size - 1)),
                       ('reduce_scatter_max', lambda: dist.reduce_scatter_tensor(part, c, op=dist.ReduceOp.MAX), part, 2 * per)]
            for what, call, output, useful in figures:
                a.fill_(rank + 1)
                c.fill_(rank + 1)
                part.fill_(rank + 1)
                call()
                seconds = timed(call, calls)
                once = latency(call, config['latency_calls'])
                expected = (torch.arange(1, size + 1).repeat_interleave(per) if what == 'all_gather'
                            else torch.full((output.numel(),), size if what.endswith('_max') else 1))
                error = float((output.cpu().float() - expected.float()).abs().max())
                if error:
                    raise RuntimeError(f'{what}: max absolute error {error}')
                print(json.dumps({'rank': rank, 'size': size, 'what': what, 'calls': calls,
                    'requested_bytes': requested, 'microseconds_per_call': seconds * 1e6 / calls,
                    'useful_output_bytes_per_call_rank': useful,
                    'useful_output_gigabytes_per_second_rank': useful * calls / seconds / 1e9,
                    'latency_microseconds': once * 1e6, 'max_abs_error': error}), flush=True)
    finally:
        dist.destroy_process_group()


def bandwidth_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bandwidth(json.loads(Path(argv[0]).read_text()))


if __name__ == '__main__':
    bandwidth_main()
