"""The data flow (design/dataflow.png) as one loop. Every native call is a whole array (kernels.h); this file is the
schedule, and its only waits are the data's own: a half's forward before its runs simulate on (route_net_wait), and a
learning batch's buffers before they are filled again.

  synthesis   cases.Distribution.draw -> a group's case (4 rollouts) -> the simulators run it, refresh by refresh,
              each refresh a forward of the actor's (its slots' tokens packed, its w answered at the next tick)
  loading     a group whose 4 runs ended: its refreshes' lanes, drawn contexts' tokens, behaviour w and draws exported
              into a learning batch; advantages and scales here (numpy)
  training    one GPU step on the batch (credit, backward, the ranks' sum, UPGrad, Adam, the codes), then theta
              published into the actor, in the GPU's queue order"""
import time

import numpy as np

from . import cases as C
from .native import const, dtype, ffi, records

L, IN, FS, CX = const('ROUTE_L'), const('ROUTE_IN'), const('ROUTE_FS'), const('ROUTE_CONTEXTS')
D2 = 2 * const('ROUTE_D')
DRAW = dtype('struct route_draw')
COSTS = dtype('struct route_costs')


def ptr(a, ctype='int[]'):
    return ffi.from_buffer(ctype, a, require_writable=True)


def view(p, dt, n):
    dt = np.dtype(dt)
    return np.frombuffer(ffi.buffer(p, max(n, 1) * dt.itemsize), dt)[:n]


class Batch:
    """a network's batch buffers (shared memory) as numpy views"""

    def __init__(self, lib, net, runs, refreshes, rows):
        self.lib, self.net, self.runs, self.K, self.rows = lib, net, runs, refreshes, rows
        b = self.p = lib.route_net_batch(net, runs, refreshes, rows)
        n = runs * refreshes
        self.lanes = view(b.lanes, np.float32, n * L * IN).reshape(n, L, IN); self.mask = view(b.mask, np.float32, n * L).reshape(n, L)
        self.h0 = view(b.h0, np.float32, runs * D2).reshape(runs, D2); self.next = view(b.next, np.float32, runs * D2).reshape(runs, D2)
        self.tokens = view(b.tokens, np.float32, rows * IN).reshape(rows, IN); self.pos = view(b.pos, np.int32, rows)
        self.w = view(b.w, np.float32, rows * FS).reshape(rows, FS); self.wcur = view(b.wcur, np.float32, rows * FS).reshape(rows, FS)
        self.scale0 = view(b.scale0, np.float32, rows); self.first = view(b.first, np.uint32, rows + 1); self.seg = view(b.seg, np.uint32, n * 2)

    def draws(self, n):
        self.lib.route_net_reserve(self.net, self.p, n)
        return view(self.p.draws, DRAW, n)

    def wait(self):
        self.lib.route_net_wait(self.net, self.p)


PHASES = {}   # seconds spent in each part of the loop (wall clock, the driver's thread), read by the reporter
OPS = {}      # int8 operations submitted (2 per multiply-add), by kind, read by the reporter
E = const('ROUTE_E'); D = const('ROUTE_D')
# a row's int8 multiply-adds through the network (net.swift's products): a lane row, the input projection (IN padded to 64),
# layer 0's gates and feed-forward, layer 1's gates; a context row, all of them and the readout. A learning step's rows
# go forward, then dgrad and wgrad: three times the forward's
LANE_MACS = 64 * D + 2 * D * D + 2 * D * E + 2 * D * D
CTX_MACS = 64 * D + 2 * (2 * D * D + 2 * D * E) + FS * D


class phase:
    def __init__(self, name): self.name = name
    def __enter__(self): self.t = time.perf_counter()
    def __exit__(self, *a): PHASES[self.name] = PHASES.get(self.name, 0.0) + time.perf_counter() - self.t


class Rollouts:
    """`groups` groups of 4 runs (a case each). Each run advances on its own (route_sims_launch) to its next refresh;
    poll() collects the runs at a refresh and packs them into whichever of the two forward batches is free, and answers
    a finished forward's runs and launches them again. A forward's size is whatever is ready when a batch is free."""

    def __init__(self, lib, net, groups, refreshes):
        self.lib, self.net, self.n, self.K = lib, net, 4 * groups, refreshes
        self.sims = lib.route_sims_new(self.n, refreshes)
        self.F = [Batch(lib, net, self.n, 1, self.n * CX) for _ in range(2)]
        self.inset = [None, None]                      # each forward batch's runs, rows and contexts, while it runs
        self.state = np.zeros((self.n, D2), np.float32); self.h0 = np.zeros((self.n, D2), np.float32)
        self.ended = np.ones(self.n, bool)
        self.waiting, self.waiting_nc = np.zeros(0, np.int32), np.zeros(0, np.int32)
        self.cases = np.zeros(groups, C.CASE); self.version = np.zeros(groups, int)
        self.got = [np.zeros(self.n, np.int32) for _ in range(3)]

    def begin(self, groups, cases, version=0):
        """groups g (slots 4g..4g+3) begin cases (one each, its 4 rollouts), launched to their first refresh"""
        groups = np.asarray(groups, np.int32)
        self.cases[groups] = cases; self.version[groups] = version
        slots = (4 * groups[:, None] + np.arange(4)).ravel().astype(np.int32)
        which = np.repeat(np.arange(len(groups), dtype=np.int32), 4); rollout = np.tile(np.arange(4, dtype=np.int32), len(groups))
        c = np.ascontiguousarray(cases)
        self.lib.route_sims_begin(self.sims, ptr(slots), len(slots), records('struct route_case', c), ptr(which), ptr(rollout))
        self.state[slots] = np.repeat(c['h'], 4, axis=0); self.h0[slots] = self.state[slots]
        self.ended[slots] = False
        self.lib.route_sims_launch(self.sims, ptr(slots), len(slots), ffi.NULL, ffi.NULL)

    def poll(self, wait):
        """one turn of the rollouts' data flow; returns the slots that ended"""
        slot, status, nc = self.got
        with phase('wait' if wait else 'collect'):
            m = self.lib.route_sims_collect(self.sims, 1 if wait else 0, ptr(slot), ptr(status), ptr(nc))
        slot, status, nc = slot[:m].copy(), status[:m], nc[:m]
        done = slot[status == 0]; self.ended[done] = True
        self.waiting = np.concatenate([self.waiting, slot[status == 1]]); self.waiting_nc = np.concatenate([self.waiting_nc, nc[status == 1]])
        for k, F in enumerate(self.F):   # a finished forward: its runs' states and answers, and they run on
            if self.inset[k] is not None and self.lib.route_net_done(self.net, F.p):
                runs, off, _ = self.inset[k]; self.inset[k] = None
                self.state[runs] = F.next[:len(runs)]
                with phase('answer and launch'):
                    self.lib.route_sims_launch(self.sims, ptr(runs), len(runs), F.p.w, ptr(off))
        free = [k for k in range(2) if self.inset[k] is None]
        if len(self.waiting) and free:   # every run waiting, into a free batch, one forward
            F, runs, nc = self.F[free[0]], self.waiting.astype(np.int32), self.waiting_nc
            off = np.zeros(len(runs), np.int32); np.cumsum(nc[:-1], out=off[1:])
            with phase('pack and commit'):
                self.lib.route_sims_pack(self.sims, ptr(runs), len(runs), ptr(off), F.p.lanes, F.p.mask, F.p.tokens, F.p.pos)
                F.h0[:len(runs)] = self.state[runs]
                self.lib.route_net_forward(self.net, F.p, len(runs), int(nc.sum()))
            self.inset[free[0]] = (runs, off, nc)
            self.waiting, self.waiting_nc = np.zeros(0, np.int32), np.zeros(0, np.int32)
            OPS['forward'] = OPS.get('forward', 0) + 2 * (len(runs) * L * LANE_MACS + int(nc.sum()) * CTX_MACS)
            self.lib.route_tel_inc(self.lib.RT_FORWARDS, 1); self.lib.route_tel_inc(self.lib.RT_EVALUATIONS, len(runs)); self.lib.route_tel_inc(self.lib.RT_FORWARD_ROWS, int(nc.sum()))
        return done

    def costs(self, slots):
        slots = np.ascontiguousarray(slots, np.int32)
        c = np.zeros(len(slots), COSTS); n = np.zeros((len(slots), 4), np.uint64)
        self.lib.route_sims_costs(self.sims, ptr(slots), len(slots), records('struct route_costs', c, True), ptr(n, 'uint64_t[]'))
        return c, n

    def ready(self):
        """groups whose four runs have ended"""
        return np.nonzero(self.ended.reshape(-1, 4).all(1))[0]

    def run_out(self):
        """polled until every run has ended (evaluation, checks)"""
        while not self.ended.all():
            self.poll(True)


def load(lib, ro, groups, B):
    """loading: groups' runs into learning batch B (their simulators' arena exported in place), as many groups as its
    runs and rows hold; their advantages (throughput against the mean of the other three), the tasks' scales. Returns
    (groups taken, runs, rows, draws, scale)"""
    K = ro.K
    slots = (4 * np.asarray(groups, np.int32)[:, None] + np.arange(4)).ravel().astype(np.int32)
    rows, draws = np.zeros((len(slots), K), np.int32), np.zeros((len(slots), K), np.int32)
    lib.route_sims_sizes(ro.sims, ptr(slots), len(slots), ptr(rows), ptr(draws))
    per = rows.sum(1).reshape(-1, 4).sum(1)
    take = int(min(np.searchsorted(np.cumsum(per), B.rows, side='right'), B.runs // 4))
    groups, slots, rows, draws = groups[:take], slots[:4 * take], rows[:4 * take], draws[:4 * take]
    m = len(slots)
    row0, draw0 = np.zeros(m * K, np.int32), np.zeros(m * K, np.int32)
    np.cumsum(rows.ravel()[:-1], out=row0[1:]); np.cumsum(draws.ravel()[:-1], out=draw0[1:])
    R, ND = int(rows.sum()), int(draws.sum())
    B.draws(ND)
    B.lanes[:m * K] = 0; B.mask[:m * K] = 0
    lib.route_sims_export(ro.sims, ptr(slots), m, 0, K, ptr(row0), ptr(draw0), B.p.lanes, B.p.mask, B.p.tokens, B.p.pos, B.p.w, B.p.draws, B.p.first)
    B.first[R] = ND
    B.h0[:m] = ro.h0[slots]
    B.seg[:2 * m * K] = np.stack([row0, row0 + rows.ravel()], 1).ravel()
    costs, counts = ro.costs(slots)
    T = costs['throughput'].reshape(-1, 4)
    adv = (T - (T.sum(1, keepdims=True) - T) / 3).ravel()
    scale = 1.0 / m
    B.scale0[:R] = np.repeat(adv * scale, rows.sum(1)).astype(np.float32)
    for k, name in enumerate(('RT_EVENTS', 'RT_DRAWS', 'RT_ENTRIES', 'RT_SENDS')):
        lib.route_tel_inc(getattr(lib, name), int(counts[:, k].sum()))
    lib.route_tel_inc(lib.RT_TRANSACTIONS, m); lib.route_tel_inc(lib.RT_GROUPS, take)
    return groups, m, R, ND, scale


def train(cfg, lib, dist, learner, actor, steps, world):
    """the loop: the rollouts polled (waiting only when nothing is ready); whenever groups have ended and a learning
    batch is free (two alternate), they are loaded and stepped, theta published to the actor, and their slots begun
    again on fresh cases"""
    K = cfg['capacity']['refreshes']
    groups = cfg['capacity']['forward_rows'] // CX // 4
    ro = Rollouts(lib, actor, groups, K)
    LB = [Batch(lib, learner, cfg['capacity']['learn_runs'], K, cfg['capacity']['forward_rows']) for _ in range(2)]
    ro.begin(np.arange(groups), dist.draw(groups)[0])
    lib.route_tel_put(lib.RT_WORKERS, ro.n)
    learned, j = 0, 0
    while learned < steps:
        ro.poll(True)
        ready = ro.ready()
        if not len(ready) or not lib.route_net_done(learner, LB[j].p):
            continue
        with phase('load'):
            taken, m, R, ND, scale = load(lib, ro, ready, LB[j])
        lag = lib.route_net_steps(learner) - ro.version[taken]
        lib.route_tel_inc(lib.RT_LAG_SUM, int(lag.sum())); lib.route_tel_inc(lib.RT_LAG_GROUPS, len(taken)); lib.route_tel_top(lib.RT_LAG_MAX, int(lag.max()))
        OPS['learn'] = OPS.get('learn', 0) + 3 * 2 * (m * K * L * LANE_MACS + R * CTX_MACS)
        with phase('commit step'):
            lib.route_net_step(learner, LB[j].p, m, K, R, ND, scale, cfg['run']['rate'], world)
            lib.route_net_publish(learner, actor)
        slots = (4 * taken[:, None] + np.arange(4)).ravel()
        dist.pool.add(ro.state[slots])
        with phase('draw and begin'):
            ro.begin(taken, dist.draw(len(taken))[0], version=lib.route_net_steps(learner))
        learned += 1; j ^= 1
        lib.route_tel_inc(lib.RT_STEPS, 1); lib.route_tel_reach(lib.RT_PHASE_FIRST_STEP)
        lib.route_tel_put(lib.RT_RUNS_IN_FLIGHT, int((~ro.ended).sum()))
    for B in LB:
        B.wait()
    while not ro.ended.all() or any(x is not None for x in ro.inset):   # the runs in flight finished before the arena goes
        ro.poll(True)
    lib.route_sims_free(ro.sims)


def evaluate(lib, learner, evaluator, cases):
    """each case's four rollouts under the learner's theta (published into the evaluator): completion over the floor"""
    lib.route_net_publish(learner, evaluator)
    ro = Rollouts(lib, evaluator, len(cases), 1)
    ro.begin(np.arange(len(cases)), cases)
    ro.run_out()
    costs, _ = ro.costs(np.arange(ro.n))
    lib.route_sims_free(ro.sims)
    return costs['throughput'].reshape(-1, 4)
