"""The load balancer the goal names (docs/facts.md; the user, 2026-10-21): a two-layer LSTM, trained online on every
rank's observed compute (the median and spread of its parts' compute spans, GPU timestamps), chooses the partition of
the LSTM's units over the ranks. Each round attaches a partition (partition.attach with each rank's units), runs the
LSTM's ticks on it (lstm.Tick through the compiler), gathers every rank's (median, spread) through a compiled
all-gather, trains the predictor (the round's partition and the round before's observations -> this round's), and
takes the next partition by gradient through the predictor (share logits minimizing a smooth maximum over ranks of
median + spread): no rule written by hand. Every rank runs the same computation on the same gathered numbers, so every
rank takes the same partition. Each round's partition, observations and tick period are printed."""
import importlib.util
import json
import os
import pathlib
import statistics
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch_mesh import partition
from torch_mesh.compiler import Program
from mesh_statistics import now

import lstm

spec = importlib.util.spec_from_file_location('lstm_service', pathlib.Path(__file__).with_name('lstm-service.py'))
lstm_service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lstm_service)


class Balancer:
    """observations -> partition: a two-layer LSTM over rounds predicting each rank's (median, spread) of compute from
    the round's shares and the previous round's observations; the next shares by gradient through it"""
    def __init__(self, n, seed, hidden=128):
        torch.manual_seed(seed)
        self.n, self.history = n, []
        self.lstm = torch.nn.LSTM(3 * n, hidden, 2)
        self.head = torch.nn.Linear(hidden, 2 * n)
        self.optimizer = torch.optim.Adam(list(self.lstm.parameters()) + list(self.head.parameters()), lr=3e-3)
        self.scale = None

    def _inputs(self, shares):
        rows = []
        for k, s in enumerate(shares):
            before = self.history[k - 1][1] if k else torch.zeros(2 * self.n)
            rows.append(torch.cat([s, before]))
        return torch.stack(rows)[:, None, :]

    def observe(self, shares, observed):
        if self.scale is None:
            self.scale = float(observed.abs().max()) or 1.0
        self.history.append((shares, observed / self.scale))
        shares_seq = [s for s, _ in self.history]
        target = torch.stack([o for _, o in self.history])
        for _ in range(200):
            out, _ = self.lstm(self._inputs(shares_seq))
            loss = ((self.head(out[:, 0, :]) - target) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return float(loss)

    def propose(self, explore, region, least):
        """the next shares within a trust region of the last: last + a zero-sum step bounded by `region` a rank (tanh),
        at least `least` each, the step taken by gradient through the predictor of the smooth worst case (median +
        spread over ranks), plus `explore` (a perturbation inside the region)"""
        last = self.history[-1][0]
        u = torch.zeros(self.n, requires_grad=True)
        step = torch.optim.Adam([u], lr=0.05)

        def shares_of(u, extra=0):
            delta = region * torch.tanh(u + extra)
            s = torch.clamp(last + delta - delta.mean(), min=least)
            return s / s.sum()
        for _ in range(200):
            out, _ = self.lstm(self._inputs([s for s, _ in self.history] + [shares_of(u)]))
            predicted = self.head(out[-1, 0, :])
            worst = torch.logsumexp(20 * (predicted[:self.n] + predicted[self.n:]), 0) / 20
            step.zero_grad()
            worst.backward()
            step.step()
        return shares_of(u.detach(), explore).detach()


def units_of(shares, total):
    raw = shares.numpy() * total
    units = np.maximum(1, np.floor(raw)).astype(int)
    while units.sum() < total:
        units[np.argmax(raw - units)] += 1
    while units.sum() > total:
        units[np.argmax(units)] -= 1
    return [int(u) for u in units]


def main(config=None):
    if config is None:
        config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    torch.set_num_threads(1)
    steps, rows, H, KIN, L = config['steps'], config['rows'], config['width'], config['input_width'], config['layers']
    k = config['parameters']['steps_per_replay']['value']
    inputs, values = lstm.inputs(config, steps, rows)
    held = lstm_service.weights(config, values)
    del values
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    group = mesh.get_group('tp').group_name
    X = torch_mesh.empty(steps, rows, KIN, dtype=torch.half)
    X.copy_(torch.from_numpy(inputs[:]))
    O = torch_mesh.empty(steps, rows, H, dtype=torch.half) if rank == 0 else None
    # every rank's (median, spread) gathered through the compiler: a row a rank
    every = torch_mesh.empty(size, 2, dtype=torch.float32)
    mine = every.narrow(0, rank, 1)  # the gather is in place: a rank's row is its SEND's source
    partition.attach(mesh, tp=size)
    g = torch.empty(1, dtype=torch.int64, device='mps').fill_(0)

    def gather(g):
        torch.ops.torch_mesh.allgatherv_out(mine, every, 0, [1] * size, group)
        g.add_(1)
    # prepared each round after the round's LSTM: a program's tables chain on each channel in the order prepared, and
    # the bridge walks them in that order, so runs are prepared in the order they run
    sharing = Program([(gather, (g,))])
    balancer = Balancer(size, config.get('seed', 0))
    explore = torch.Generator().manual_seed(config.get('seed', 0))
    shares = torch.full((size,), 1.0 / size)
    for round_ in range(config['rounds']):
        units = units_of(shares, H)
        partition.attach(mesh, tp=units)
        split = lambda t: partition.distribute_tensor(t, mesh, [Shard(1)])
        tick = lstm.Tick(mesh, [split(w) for w, _ in held], [split(b) for _, b in held], rows, steps)
        t = torch.empty(1, dtype=torch.int64, device='mps').fill_(L - 1)
        starting = [lambda _, s=s: tick.starting(s, X[s]) for s in range(L - 1)]
        whole, rest = divmod(steps, k)
        stepper = {n: tick.stepper(X, O, n) for n in {k, rest} if n}
        program = Program([(fn, (t,)) for fn in starting] + [(stepper[k], (t,))] * whole +
                          ([(stepper[rest], (t,))] if rest else []))
        program.prepare(1)
        started = now()
        program.forward()
        while (ended := program.completion()) is None:
            if now() - started > 8.0:
                raise TimeoutError('the run did not complete within 8 s (a run is bounded: docs/facts.md)')
            time.sleep(0)
        run = program.done[-1]
        spans = [x for x in run.spans() if x > 0]
        parts = run.parts()
        starts = sorted(s for _, s, _ in parts if s)
        period = statistics.median(b - a for a, b in zip(starts, starts[1:])) * 1e6
        mine.copy_(torch.tensor([[statistics.median(spans), statistics.pstdev(spans)]], device='mps'))
        sharing.prepare(1)
        sharing.forward()
        while sharing.completion() is None:
            time.sleep(0)
        torch.mps.synchronize()
        observed = every.cpu()
        fit = balancer.observe(shares, torch.cat([observed[:, 0], observed[:, 1]]))
        print(json.dumps(dict(phase='round', round=round_, units=units, compute_median_us=[round(float(x), 2) for x in observed[:, 0]],
                              compute_spread_us=[round(float(x), 2) for x in observed[:, 1]], tick_period_us=round(period, 1),
                              predictor_loss=round(fit, 6))), flush=True)
        noise = torch.randn(size, generator=explore) * (0.5 if round_ < config.get('explore_rounds', 2) else 0.0)
        shares = balancer.propose(noise, config.get('region', 0.06), 1.0 / H)
        del program, tick, stepper, starting
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
