"""The column-separable FFN (the first Amdahl studies' MLP: docs/measurement.md, tp.py's DIM 2048, FFN 5632) through the
compiler: every rank holds x, a column block of W1 and the matching row block of W2 (partition's split of the hidden
units); y = relu(x W1) W2 is each rank's relu(x W1_r) W2_r summed over the ranks in place (the all-reduce). `layers`
FFNs a run; rank 0 checks y against the CPU; each run's time and GFLOP/s. A run is bounded at 8 s."""
import json
import sys
import time

import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch_mesh import partition
from torch_mesh.compiler import Program
from mesh_statistics import now, measure, emit


def main(config=None):
    if config is None:
        config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    B, D, F, layers, seed = (config[key] for key in ('tokens', 'dim', 'hidden', 'layers', 'seed'))
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=F)
    group, blocks = mesh.get_group('tp').group_name, partition.split(mesh, 'tp', F)
    first = sum(blocks[:rank])
    g = torch.Generator().manual_seed(seed)
    x_cpu = torch.randn(B, D, generator=g) * 0.5
    w1_cpu = torch.randn(D, F, generator=g) * D ** -0.5
    w2_cpu = torch.randn(F, D, generator=g) * F ** -0.5
    x = torch_mesh.empty(B, D, dtype=torch.half).copy_(x_cpu.half().to('mps'))
    y = torch_mesh.empty(B, D, dtype=torch.half)
    w1 = torch_mesh.empty(D, blocks[rank], dtype=torch.half).copy_(w1_cpu[:, first:first + blocks[rank]].half().to('mps'))
    w2 = torch_mesh.empty(blocks[rank], D, dtype=torch.half).copy_(w2_cpu[first:first + blocks[rank]].half().to('mps'))
    t = torch.empty(1, dtype=torch.int64, device='mps').fill_(0)

    summed = not config.get('partial_only', False)
    staged = config.get('stages', False)
    a1 = torch_mesh.empty(B, blocks[rank], dtype=torch.half)
    a2 = torch_mesh.empty(B, blocks[rank], dtype=torch.half)

    def ffn(t):
        if staged:
            # each stage into a caller tensor of its own, so rank 0 checks each against torch
            a1.copy_(x @ w1)
            a2.copy_(torch.relu(a1))
            y.copy_(a2 @ w2)
        else:
            y.copy_(torch.relu(x @ w1) @ w2)
        if summed:
            torch.ops.torch_mesh.allreduce_(y, group)
        t.add_(1)
    program = Program([(ffn, (t,))] * layers)
    repeats = config.get('repeats', 3)
    begun = now()
    program.prepare(repeats)
    print(json.dumps(dict(phase='prepared', seconds=round(now() - begun, 3))), flush=True)
    flops = layers * 2 * 2 * B * D * blocks[rank]
    for repeat in range(repeats):
        started = now()
        program.forward()
        while (ended := program.completion()) is None:
            if now() - started > 8.0:
                raise TimeoutError('the run did not complete within 8 s (a run is bounded: docs/facts.md)')
            time.sleep(0)
        emit({**measure(started, ended, flops, layers), 'workload': 'ffn_column_separable', 'rank': rank, 'size': size,
              'report': dict(tokens=B, dim=D, hidden=F, block=blocks[rank], layers=layers, repeat=repeat,
                             gflops=round(flops / (ended - started) / 1e9, 2), us_per_layer=round((ended - started) / layers * 1e6, 1))})
    if rank == 0:
        torch.mps.synchronize()
        expected = torch.relu(x_cpu.half().float() @ w1_cpu.half().float()) @ w2_cpu.half().float() if summed else \
            torch.relu(x_cpu.half().float() @ w1_cpu[:, :blocks[0]].half().float()) @ w2_cpu[:blocks[0]].half().float()
        print(json.dumps(dict(phase='checked', max_abs_error=float((y.float().cpu() - expected).abs().max()),
                              output_max_abs=float(expected.abs().max()))), flush=True)
        if staged:
            e1 = x_cpu.half().float() @ w1_cpu[:, :blocks[0]].half().float()
            print(json.dumps(dict(phase='stages', x_w1_error=float((a1.float().cpu() - e1).abs().max()), x_w1_max=float(e1.abs().max()),
                                  relu_error=float((a2.float().cpu() - torch.relu(a1.float().cpu())).abs().max()),
                                  y_from_a2_error=float((y.float().cpu() - a2.float().cpu() @ w2_cpu[:blocks[0]].half().float()).abs().max()),
                                  w2_error=float((w2.float().cpu() - w2_cpu[:blocks[0]].half().float()).abs().max()),
                                  y_max=float(y.float().abs().max()), y_zero=float((y == 0).float().mean()))), flush=True)
    if staged:
        # every rank's own operands and stages read by the CPU from the region's bytes (no GPU op between the run and
        # the reading), against torch on the CPU: the run's writes apart from torch's MPS reads of them
        import ctypes

        def host(tensor):
            n = tensor.numel() * tensor.element_size()
            return torch.frombuffer(bytearray((ctypes.c_char * n).from_address(torch_mesh.address(tensor))),
                                    dtype=tensor.dtype).reshape(tensor.shape).float()
        torch.mps.synchronize()
        hx, hw1, hw2, ha1, ha2, hy = (host(v) for v in (x, w1, w2, a1, a2, y))
        mine = slice(first, first + blocks[rank])
        print(json.dumps(dict(phase='host_stages', rank=rank,
                              x_error=float((hx - x_cpu.half().float()).abs().max()),
                              w1_error=float((hw1 - w1_cpu[:, mine].half().float()).abs().max()),
                              w2_error=float((hw2 - w2_cpu[mine].half().float()).abs().max()),
                              a1_error=float((ha1 - hx @ hw1).abs().max()), a1_max=float(ha1.abs().max()),
                              a2_error=float((ha2 - torch.relu(ha1)).abs().max()),
                              y_error=float((hy - ha2 @ hw2).abs().max()) if not summed else None,
                              y_max=float(hy.abs().max()), y_zero=float((hy == 0).float().mean()))), flush=True)
        # torch's own reads of the same bytes, each against the CPU's (host): which of torch's paths reads them wrong
        reads = {'cpu': lambda v: v.cpu().float(), 'float_cpu': lambda v: v.float().cpu(),
                 'clone_cpu': lambda v: v.clone().cpu().float(), 'float_cpu_again': lambda v: v.float().cpu(),
                 'synchronized_float_cpu': lambda v: (torch.mps.synchronize(), v.float().cpu())[1]}
        print(json.dumps(dict(phase='torch_reads', rank=rank, **{
            f'{name}_{label}': float((read(v) - host(v)).abs().max())
            for label, v in (('y', y), ('w2', w2), ('a1', a1)) for name, read in reads.items()})), flush=True)
        # a cast's output (a fresh MPS lease) read by the CPU from its bytes and by torch's copy, and the leases' spans
        for label, v in (('y', y), ('w2', w2)):
            A = v.float()
            torch.mps.synchronize()
            hA, hv = host(A), host(v)
            C = A.cpu()
            held = dict(A=A, C=C, x=x, w1=w1, w2=w2, a1=a1, a2=a2, y=y)
            spans = {k: (torch_mesh.address(u), torch_mesh.address(u) + u.untyped_storage().nbytes()) for k, u in held.items()}
            overlaps = [(i, j) for i in spans for j in spans if i < j and spans[i][0] < spans[j][1] and spans[j][0] < spans[i][1]]
            print(json.dumps(dict(phase='cast_probe', rank=rank, what=label, cast_error=float((hA - hv).abs().max()),
                                  copy_error=float((C - hA).abs().max()), read_error=float((C - hv).abs().max()),
                                  overlaps=overlaps, A=[hex(a) for a in spans['A']], C=[hex(a) for a in spans['C']],
                                  outside=[k for k, (a, _) in spans.items() if a == 0])), flush=True)
    if staged and not summed:
        # the second product alone, its operand written before the run: its kernel apart from the order it follows
        alone = torch_mesh.empty(B, D, dtype=torch.half)
        program2 = Program([(lambda t: (alone.copy_(a2 @ w2), t.add_(1)), (t,))])
        program2.prepare(1)
        started = now()
        program2.forward()
        while program2.completion() is None:
            if now() - started > 8.0:
                raise TimeoutError('the run did not complete within 8 s (a run is bounded: docs/facts.md)')
            time.sleep(0)
        torch.mps.synchronize()
        if rank == 0:
            print(json.dumps(dict(phase='alone', error=float((alone.float().cpu() - a2.float().cpu() @ w2.float().cpu()).abs().max()),
                                  max=float(alone.float().abs().max()))), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
