"""The training distribution: every window the trainer runs is a case drawn here (trainer.h struct route_case), as a
numpy record array the trainer reads in place. A case is the data a learning step sees, all of it:

  the fabric      a random fabric (fabric.random_fabric: six kinds), from a pool drawn at the start
  the stream      the collectives on it: up to three already under way at 0 (some of their pages done), then arrivals
                  over the window's duration at the offered load, each the floor of the last over the load times an
                  exponential; random kind, root and size (log-uniform 6 KB-12 MB); a random backlog on half the lanes
  the window      its refreshes' spacing (log-uniform 20-300 us), the lag before a table is active (uniform in it), the
                  refreshes (1..refreshes), the router before the first table (shortest-path tree or uniform), posting
                  (burst or paced)
  the state       the policy's recurrent state at the window's start: zero, a Gaussian of a random scale, or a state
                  an earlier window ended with (the pool), each a third (the pool's third zero until it has states)
  the keys        the run's draws before the window (one, the four rollouts' the same) and each rollout's own

`online` replaces it with the deployment's distribution (tests/switch.c's measurement): the measured cluster, one
all-gather or all-reduce at 0 of a log-uniform size from rank 0, every page posted at its release, refreshed every
100 us and active 100 us late, the state an earlier window's (the pool) or zero.

The evaluation's cases (`measured`) are the measured cluster's one collective at a time, every kind and size, a zero
state, four rollouts under fixed keys."""
import numpy as np

from . import fabric
from .native import const, dtype, ffi, records

CASE = dtype('struct route_case')
OP = dtype('struct route_op')
D2 = 2 * const('ROUTE_D')
KINDS = ('all-gather', 'all-reduce', 'scatter')
SIZES = (6144, 98304, 3145728, 12582912)


class StatePool:
    """the states windows ended with: kept until `capacity`, then each new one replacing a random kept one with
    probability `replace` (a reservoir that keeps refreshing)"""

    def __init__(self, capacity, replace, rng):
        self.states = np.zeros((capacity, D2), np.float32); self.count = 0; self.replace = replace; self.rng = rng

    def add(self, states):
        for s in states:
            if self.count < len(self.states):
                self.states[self.count] = s; self.count += 1
            elif self.rng.random() < self.replace:
                self.states[int(self.rng.random() * len(self.states))] = s

    def draw(self, n):
        return self.states[(self.rng.random(n) * self.count).astype(int)] if self.count else np.zeros((n, D2), np.float32)


class Distribution:
    def __init__(self, cfg, measured, lib, rng):
        self.cfg, self.measured, self.lib, self.rng = cfg, measured, lib, rng
        # filled in place at the C dtype (np.stack would promote the records to a packed dtype, without the struct's
        # trailing padding: 940 bytes a record, not 944)
        self.fabrics = np.empty(1 if cfg['online'] else cfg['fabrics'], fabric.FABRIC)
        for i in range(len(self.fabrics)):
            self.fabrics[i] = measured if cfg['online'] else fabric.random_fabric(rng, measured)
        self.pool = StatePool(cfg['pool'], cfg['pool_replace'], rng)

    def floors(self, idx, kind, size, root):
        """the collectives' floors (sim.c route_op_floors), the i-th on fabric idx[i]"""
        n = len(idx); out = np.empty(n)
        idx, kind, root = (np.ascontiguousarray(a, np.int32) for a in (idx, kind, root)); size = np.ascontiguousarray(size, np.float64)
        self.lib.route_op_floors(records('struct route_fabric', self.fabrics), ffi.from_buffer('int[]', idx), n, ffi.from_buffer('int[]', kind),
                                 ffi.from_buffer('double[]', size), ffi.from_buffer('int[]', root), ffi.from_buffer('double[]', out, require_writable=True))
        return out

    def sizes(self, n):
        return np.round(np.exp(np.log(SIZES[0]) + self.rng.random(n) * (np.log(SIZES[-1]) - np.log(SIZES[0]))))

    def roots(self, fab):
        nc = self.fabrics['ncompute'][fab]
        return self.fabrics['compute'][fab, (self.rng.random(len(fab)) * nc).astype(int) % nc]

    def states(self, n):
        """a third zero, a third N(0, sigma^2) with sigma uniform in [0, 1), a third the pool's"""
        pick = self.rng.random(n); h = np.zeros((n, D2), np.float32)
        g = pick < 1 / 3
        h[g] = (self.rng.random(g.sum())[:, None] * self.rng.standard_normal((g.sum(), D2))).astype(np.float32)
        p = (pick >= 1 / 3) & (pick < 2 / 3)
        h[p] = self.pool.draw(p.sum())
        return h, np.select([g, p], [1, 2], 0)   # the source: 0 zero, 1 Gaussian, 2 pool

    def draw(self, n):
        """n cases of the training distribution (or the online one); and each one's state source"""
        c = np.zeros(n, CASE)
        c['key0'] = self.rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)[:, None]   # the group's run before the window, shared
        c['key'] = self.rng.integers(0, 2**32, (n, 4), dtype=np.uint64).astype(np.uint32)
        if self.cfg['online']:
            c['f'] = self.measured
            c['ops'] = 1; c['op']['kind'][:, 0] = (self.rng.random(n) * 2).astype(int) % 2; c['op']['size'][:, 0] = self.sizes(n)
            c['op']['root'][:, 0] = self.measured['compute'][0]
            c['w']['dt'] = 100e3; c['w']['lag'] = 100e3; c['w']['points'] = self.cfg['refreshes']; c['w']['warm'] = 2; c['w']['burst'] = 1
            c['h'] = self.pool.draw(n)
            return c, np.full(n, 2)
        fab = (self.rng.random(n) * len(self.fabrics)).astype(int)
        F = self.fabrics[fab]; c['f'] = F
        duration = 1e6 * (1 + 4 * self.rng.random(n)); load = 0.3 + 1.2 * self.rng.random(n)
        lanes = np.arange(const('ROUTE_L'))[None, :] < F['lanes'][:, None]
        c['background'] = np.where(lanes & (self.rng.random((n, const('ROUTE_L'))) < 0.5), F['rate'] * 200e3 * self.rng.random((n, const('ROUTE_L'))), 0)
        # under way at 0: 0-3 collectives, done 0-95% of their pages
        under = (self.rng.random(n) * 4).astype(int)
        U = 3
        uk = (self.rng.random((n, U)) * len(KINDS)).astype(int) % len(KINDS)
        ur = np.stack([self.roots(fab) for _ in range(U)], 1); us = self.sizes(n * U).reshape(n, U); ud = 0.95 * self.rng.random((n, U))
        # arrivals: t from 0, each next one the last's floor over the load times an exponential, while t <= duration;
        # floors computed a block of candidates at a time for the windows whose stream is still within its duration
        room = const('ROUTE_OPS'); arrive = [[] for _ in range(n)]; t = np.zeros(n); left = room - under
        live = np.arange(n)
        while len(live):
            B = 16
            k = np.minimum((self.rng.random((len(live), B)) * len(KINDS)).astype(int), len(KINDS) - 1)
            r = np.stack([self.roots(fab[live]) for _ in range(B)], 1); s = self.sizes(len(live) * B).reshape(len(live), B)
            fl = self.floors(np.repeat(fab[live], B), k.ravel(), s.ravel(), r.ravel()).reshape(len(live), B)
            e = -np.log(1 - self.rng.random((len(live), B)) * 0.999999)
            still = []
            for j, i in enumerate(live):
                for b in range(B):
                    if t[i] > duration[i] or len(arrive[i]) >= left[i]:
                        break
                    arrive[i].append((k[j, b], r[j, b], s[j, b], t[i])); t[i] += fl[j, b] / load[i] * e[j, b]
                else:
                    still.append(i)
            live = np.array(still, int)
        for i in range(n):
            ops = [(uk[i, q], ur[i, q], us[i, q], 0.0, ud[i, q]) for q in range(under[i])] + [(a, b, s, tt, 0.0) for a, b, s, tt in arrive[i]]
            c['ops'][i] = len(ops)
            for q, (kk, rr, ss, tt, dd) in enumerate(ops):
                c['op'][i, q] = (kk, rr, ss, tt, 0.0, dd)
        dt = np.exp(np.log(20e3) + self.rng.random(n) * (np.log(300e3) - np.log(20e3)))
        c['w']['dt'] = dt; c['w']['lag'] = self.rng.random(n) * dt
        c['w']['points'] = 1 + (self.rng.random(n) * self.cfg['refreshes']).astype(int) % self.cfg['refreshes']
        c['w']['warm'] = np.where(self.rng.random(n) < 0.5, 2, 3); c['w']['burst'] = (self.rng.random(n) < 0.5).astype(int)
        c['h'], source = self.states(n)
        return c, source


def measured(f, modes):
    """the evaluation: every kind and size on the measured cluster from rank 0, a zero state, keys 77.. and 91..; one
    case (four rollouts) for each (mode, burst, kind, size); the policy refreshed every 100 us, active 100 us late"""
    rows = [(m, b, k, s) for m, b in modes for k in range(len(KINDS)) for s in SIZES]
    c = np.zeros(len(rows), CASE)
    c['f'] = f; c['ops'] = 1
    for i, (m, b, k, s) in enumerate(rows):
        c['mode'][i] = m; c['op'][i, 0]['kind'] = k; c['op'][i, 0]['size'] = s; c['op'][i, 0]['root'] = f['compute'][0]
        c['w'][i] = (0, 100e3, 100e3, 0, 2, b)
    c['key0'] = np.arange(77, 81, dtype=np.uint32); c['key'] = np.arange(91, 95, dtype=np.uint32)
    return c, rows
