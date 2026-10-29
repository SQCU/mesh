"""design/replay-templates-2026-10-29.md: a prepared function saved as a template, loaded in a fresh process on new data.

  python examples/templates.py DIRECTORY          every case, each recorded and saved, then loaded in a new process
  python examples/templates.py DIRECTORY CASE save|load [LAYER]

Cases: replay (a compute call and a recorded replay, out = 4x + 3y), host (a CPU program, y = x w^T), collective (an
all-reduce on a 1-rank mesh communicator, planned only by the recording process), view (the replay case recorded on
layer 0 of stacked operands, loaded on layer 2). Each phase prints its preparation seconds and its largest error
against the reference, on its own data."""
import os
import struct
import subprocess
import sys
import time

import torch

SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void axpy(device const half *x [[buffer(0)]], device const half *y [[buffer(1)]], device half *out [[buffer(2)]],
                 constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = x[i] * half(2) + y[i];
}"""
N, THREADS, WRITE_SHIFT, LAYERS = 1 << 16, 256, 32, 4
READS_01_WRITES_2 = 1 | 2 | (4 << WRITE_SHIFT)


def wait(schedule):
    from torch_mesh import _stream
    capsule = schedule.completion()
    while not _stream.completed(capsule):
        time.sleep(0.001)


def report(case, mode, seconds, error):
    print(f'{case} {mode}: preparation {seconds:.4f} s, largest error {error:.3g}', flush=True)


def replay(directory, mode, layer):
    """Two stages through one template: t = 2x + y (a compute call), out = 2t + y (a recorded replay)."""
    from torch_mesh import _stream
    g = torch.Generator().manual_seed(1 if mode == 'save' else 2)
    xs, ys = (torch.randn(LAYERS, N, generator=g).half().to('mps') for _ in range(2))
    ts, outs = (torch.zeros(LAYERS, N, dtype=torch.half, device='mps') for _ in range(2))
    x, y, t, out = xs[layer], ys[layer], ts[layer], outs[layer]
    started = time.perf_counter()
    if mode == 'save':
        kernel = _stream.pipeline(SOURCE, 'axpy')
        schedule = _stream.Schedule()
        schedule.compute(kernel, [x, y, t], [N], 3, N // THREADS, 1, THREADS, READS_01_WRITES_2)
        schedule.append(_stream.Commands([(kernel, [t, y, out], {3: struct.pack('<I', N)}, (N // THREADS, 1, 1),
                                           (THREADS, 1, 1), False, READS_01_WRITES_2)]))
        schedule.finish()
        seconds = time.perf_counter() - started
        schedule.save(directory, [x, y, out])
    else:
        schedule = _stream.Schedule.load(directory, [x, y, out])
        schedule.finish()
        seconds = time.perf_counter() - started
    schedule.enqueue()
    wait(schedule)
    torch.mps.synchronize()
    error = (outs.float() - (xs.float() * 4 + ys.float() * 3) * (torch.arange(LAYERS, device='mps') == layer)[:, None]).abs().max()
    return seconds, error.item()


def host(directory, mode, layer):
    """A CPU product as a declared host operation, rebuilt by engines.rebuild."""
    import torch_mesh
    from torch_mesh import _stream, engines
    m, k, n = 64, 128, 96
    g = torch.Generator().manual_seed(1 if mode == 'save' else 2)
    x = torch_mesh.empty(m, k, dtype=torch.float32, device='cpu').copy_(torch.randn(m, k, generator=g))
    w = torch_mesh.empty(n, k, dtype=torch.float32, device='cpu').copy_(torch.randn(n, k, generator=g))
    y = torch_mesh.empty(m, n, dtype=torch.float32, device='cpu').zero_()
    started = time.perf_counter()
    if mode == 'save':
        native = engines._coreml()
        records = [(engines._CPU_PRODUCT, engines.CpuCommand(None, m, n, k, x.stride(0), w.stride(0), y.stride(0), x.stride(1),
                    w.stride(1), y.stride(1), engines._host_address(x), engines._host_address(w), engines._host_address(y), None))]
        binding, address = engines._cpu_program(native, records, lambda size: 0)
        program = engines.CpuProgram(native, binding, engines._cpu_declaration(records, address), (x, w), (y,), [], [])
        schedule = _stream.Schedule()
        program.append(schedule)
        schedule.finish()
        seconds = time.perf_counter() - started
        schedule.save(directory, [x, w, y])
    else:
        schedule = _stream.Schedule.load(directory, [x, w, y], host=engines.rebuild)
        schedule.finish()
        seconds = time.perf_counter() - started
    schedule.enqueue()
    wait(schedule)
    return seconds, (y - x @ w.T).abs().max().item()


def collective(directory, mode, layer):
    """An all-reduce crossing on a 1-rank communicator; only the recording process plans it, the template plans it again."""
    os.environ.update(RANK='0', WORLD_SIZE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT=os.environ.get('MASTER_PORT', '29690'))
    import torch.distributed as dist
    import torch_mesh
    from torch_mesh import _stream, backend
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    try:
        group = backend._mesh(dist.group.WORLD.group_name, 'plan')
        x, y = torch_mesh.empty(N, dtype=torch.float32), torch_mesh.empty(N, dtype=torch.float32)
        x.copy_(torch.arange(N, dtype=torch.float32, device='mps') * (1 if mode == 'save' else 3))
        y.zero_()
        torch.mps.synchronize()
        started = time.perf_counter()
        if mode == 'save':
            torch_mesh.plan(None, [[backend.Call(backend.ALLREDUCE, N, torch.float32, bindings=(x, y))]])
            crossing = _stream.Collectives([(backend.ALLREDUCE, [x], [y], [N], 0, 0)], group.handle, [x, y], [], [])
            schedule = _stream.Schedule()
            crossing.append(schedule)
            schedule.finish()
            seconds = time.perf_counter() - started
            schedule.save(directory, [x, y])
        else:
            schedule = _stream.Schedule.load(directory, [x, y], communicators=[group.handle])
            schedule.finish()
            seconds = time.perf_counter() - started
        schedule.enqueue()
        wait(schedule)
        torch.mps.synchronize()
        return seconds, (y - x).abs().max().item()
    finally:
        dist.destroy_process_group()


CASES = {'replay': (replay, 0, 0), 'host': (host, 0, 0), 'collective': (collective, 0, 0), 'view': (replay, 0, 2)}


def main():
    directory = sys.argv[1]
    if len(sys.argv) == 2:
        for k, (case, (_, recorded, loaded)) in enumerate(CASES.items()):
            for mode, layer in (('save', recorded), ('load', loaded)):
                env = dict(os.environ, MASTER_PORT=str(29690 + 2 * k + (mode == 'load')))
                subprocess.run([sys.executable, __file__, os.path.join(directory, case), case, mode, str(layer)], env=env, check=True)
        return
    case, mode = sys.argv[2], sys.argv[3]
    seconds, error = CASES[case][0](sys.argv[1], mode, int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    report(case, mode, seconds, error)


if __name__ == '__main__':
    main()
