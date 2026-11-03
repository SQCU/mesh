"""The trainable LSTM through the compiler, data-parallel: each rank its own batch shard of sequences and targets (from
its rank's seed), the same W and b; a step runs the LSTM over T steps, takes loss = mean((h_T - y)^2), autograd's
gradients of W and b (traced with the step: torch_mesh/compiler.py lowers the backward's products and pointwise ops),
sums them over every rank in place (the all-reduce), and steps W and b. `updates` steps a run. Every rank prints its loss
and a checksum of W (the same on every rank); rank 0 checks W against the same descent on the CPU. A run is bounded at
8 s."""
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


def lstm_loss(W, b, X, Y, device):
    T, B, H = X.shape[0], X.shape[1], Y.shape[1]
    h = torch.zeros(B, H, device=device)
    c = torch.zeros(B, H, device=device)
    for s in range(T):
        z = torch.nn.functional.linear(torch.cat([X[s], h], 1), W, b)
        i, f, g, o = z.unflatten(-1, (4, -1)).unbind(-2)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
    return ((h - Y) ** 2).mean()


def shard(rank, config):
    T, B, K, H, seed = (config[key] for key in ('steps', 'rows', 'inputs', 'width', 'seed'))
    g = torch.Generator().manual_seed(seed * 1000 + rank)
    return torch.randn(T, B, K, generator=g), torch.randn(B, H, generator=g) * 0.5


def main(config=None):
    if config is None:
        config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    K, H, updates, lr, seed = (config[key] for key in ('inputs', 'width', 'updates', 'lr', 'seed'))
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=H)
    group = mesh.get_group('tp').group_name
    x_cpu, y_cpu = shard(rank, config)
    w0 = torch.randn(4 * H, K + H, generator=torch.Generator().manual_seed(seed)) * (K + H) ** -0.5
    X = torch_mesh.empty(*x_cpu.shape, dtype=torch.float32).copy_(x_cpu.to('mps'))
    Y = torch_mesh.empty(*y_cpu.shape, dtype=torch.float32).copy_(y_cpu.to('mps'))
    W = torch_mesh.empty(4 * H, K + H, dtype=torch.float32)
    b = torch_mesh.empty(4 * H, dtype=torch.float32)
    GW = torch_mesh.empty(4 * H, K + H, dtype=torch.float32)
    Gb = torch_mesh.empty(4 * H, dtype=torch.float32)
    W.requires_grad_(True)
    b.requires_grad_(True)
    scale = lr / size
    t = torch.empty(1, dtype=torch.int64, device='mps').fill_(0)

    def update(t):
        loss = lstm_loss(W, b, X, Y, 'mps')
        gW, gb = torch.autograd.grad(loss, [W, b])
        with torch.no_grad():
            GW.copy_(gW)
            Gb.copy_(gb)
            torch.ops.torch_mesh.allreduce_(GW, group)
            torch.ops.torch_mesh.allreduce_(Gb, group)
            W.copy_(W - GW * scale)
            b.copy_(b - Gb * scale)
            t.add_(1)
    program = Program([(update, (t,))] * updates)
    repeats = config.get('repeats', 3)
    begun = now()
    program.prepare(repeats)
    print(json.dumps(dict(phase='prepared', seconds=round(now() - begun, 3))), flush=True)
    for repeat in range(repeats):
        with torch.no_grad():
            W.copy_(w0.to('mps'))
            b.zero_()
        started = now()
        program.forward()
        while (ended := program.completion()) is None:
            if now() - started > 8.0:
                raise TimeoutError('the run did not complete within 8 s (a run is bounded: docs/facts.md)')
            time.sleep(0)
        torch.mps.synchronize()
        w, bb = W.detach().cpu(), b.detach().cpu()
        loss = float(lstm_loss(w, bb, x_cpu, y_cpu, 'cpu'))
        emit({**measure(started, ended, 0, updates), 'workload': 'lstm_train', 'rank': rank, 'size': size,
              'report': dict(updates=updates, repeat=repeat, loss=loss, w_checksum=float(w.double().sum()))})
    if rank == 0:
        shards = [shard(r, config) for r in range(size)]
        ref_w, ref_b = w0.clone().requires_grad_(True), torch.zeros(4 * H, requires_grad=True)
        for _ in range(updates):
            grads = [torch.autograd.grad(lstm_loss(ref_w, ref_b, x, y, 'cpu'), [ref_w, ref_b]) for x, y in shards]
            with torch.no_grad():
                ref_w -= scale * sum(g[0] for g in grads)
                ref_b -= scale * sum(g[1] for g in grads)
        print(json.dumps(dict(phase='checked', max_abs_error=float((w - ref_w.detach()).abs().max()),
                              w_max_abs=float(ref_w.detach().abs().max()),
                              start_loss=float(lstm_loss(w0, torch.zeros(4 * H), x_cpu, y_cpu, 'cpu')))), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
