"""The data flow (design/dataflow.png) as one loop. Every native call is a whole array (kernels.h); this file is the
schedule, and its only waits are the data's own: a half's forward before its runs simulate on (route_net_wait), and a
learning batch's buffers before they are filled again.

  synthesis   cases.Distribution.draw -> a group's case (4 rollouts) -> the simulators run it, refresh by refresh,
              each refresh a forward of the actor's (its slots' tokens packed, its w answered at the next tick)
  loading     a group whose 4 runs ended: its refreshes' lanes, drawn contexts' tokens, behaviour w and draws exported
              into a learning batch; advantages and scales here (numpy)
  training    one GPU step on the batch (credit, backward, the ranks' sum, UPGrad, Adam, the codes), then theta
              published into the actor, in the GPU's queue order"""
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


class Rollouts:
    """`groups` groups of 4 slots (a case each), in `halves` halves that take turns: one simulates while the other's
    forward runs. tick(h) waits for half h's forward, steps its runs (answering it), packs and commits the next."""

    def __init__(self, lib, net, groups, refreshes, halves=2):
        self.lib, self.net, self.n, self.K = lib, net, 4 * groups, refreshes
        self.sims = lib.route_sims_new(self.n, refreshes)
        self.half = np.array_split(np.arange(self.n, dtype=np.int32), halves)
        self.F = [Batch(lib, net, len(s), 1, len(s) * CX) for s in self.half]
        self.state = np.zeros((self.n, D2), np.float32); self.h0 = np.zeros((self.n, D2), np.float32)
        self.wrow = np.full(self.n, -1, np.int32); self.ended = np.ones(self.n, bool)
        self.pend = [np.zeros(0, np.int32) for _ in self.half]
        self.cases = np.zeros(groups, C.CASE)
        self.version = np.zeros(groups, int)

    def begin(self, groups, cases, version=0):
        """groups g (slots 4g..4g+3) begin cases (one each, its 4 rollouts)"""
        groups = np.asarray(groups, np.int32)
        self.cases[groups] = cases; self.version[groups] = version
        slots = (4 * groups[:, None] + np.arange(4)).ravel().astype(np.int32)
        which = np.repeat(np.arange(len(groups), dtype=np.int32), 4); rollout = np.tile(np.arange(4, dtype=np.int32), len(groups))
        c = np.ascontiguousarray(cases)
        self.lib.route_sims_begin(self.sims, ptr(slots), len(slots), records('struct route_case', c), ptr(which), ptr(rollout))
        self.state[slots] = np.repeat(c['h'], 4, axis=0); self.h0[slots] = self.state[slots]
        self.wrow[slots] = -1; self.ended[slots] = False

    def tick(self, h):
        """returns the slots that ended"""
        F, sl = self.F[h], self.half[h]
        if len(self.pend[h]):
            F.wait(); self.state[self.pend[h]] = F.next[:len(self.pend[h])]
        active = sl[~self.ended[sl]]
        status, nc = np.zeros(len(active), np.int32), np.zeros(len(active), np.int32)
        wrow = np.ascontiguousarray(self.wrow[active])
        self.lib.route_sims_step(self.sims, ptr(active), len(active), F.p.w, ptr(wrow), ptr(status), ptr(nc))
        done, pend, nc = active[status == 0], active[status == 1], nc[status == 1]
        self.ended[done] = True; self.wrow[active] = -1
        off = np.zeros(len(pend), np.int32); np.cumsum(nc[:-1], out=off[1:])
        if len(pend):
            self.lib.route_sims_pack(self.sims, ptr(pend), len(pend), ptr(off), F.p.lanes, F.p.mask, F.p.tokens, F.p.pos)
            F.h0[:len(pend)] = self.state[pend]
            self.lib.route_net_forward(self.net, F.p, len(pend), int(nc.sum()))
        self.wrow[pend] = off; self.pend[h] = pend
        self.lib.route_tel_inc(self.lib.RT_FORWARDS, 1); self.lib.route_tel_inc(self.lib.RT_EVALUATIONS, len(pend)); self.lib.route_tel_inc(self.lib.RT_FORWARD_ROWS, int(nc.sum()))
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
        """every half ticked until every run has ended (evaluation, checks)"""
        while not self.ended.all() or any(len(p) for p in self.pend):
            for h in range(len(self.half)):
                self.tick(h)


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


def train(cfg, lib, dist, learner, actor, steps, world, emit_pool=None):
    """the loop: both halves ticked; every group whose runs have ended loaded into a learning batch (the two batches
    alternate) and stepped; theta published to the actor; the groups' slots begun again on fresh cases"""
    K = cfg['capacity']['refreshes']
    groups = cfg['capacity']['forward_rows'] // CX // 4
    ro = Rollouts(lib, actor, groups, K)
    LB = [Batch(lib, learner, cfg['capacity']['learn_runs'], K, cfg['capacity']['forward_rows']) for _ in range(2)]
    ro.begin(np.arange(groups), dist.draw(groups)[0])
    lib.route_tel_put(lib.RT_WORKERS, ro.n)
    learned, j = 0, 0
    while learned < steps:
        for h in range(len(ro.half)):
            ro.tick(h)
        ready = ro.ready()
        if not len(ready):
            continue
        LB[j].wait()                               # the batch's last step done with its buffers
        taken, m, R, ND, scale = load(lib, ro, ready, LB[j])
        lag = lib.route_net_steps(learner) - ro.version[taken]
        lib.route_tel_inc(lib.RT_LAG_SUM, int(lag.sum())); lib.route_tel_inc(lib.RT_LAG_GROUPS, len(taken)); lib.route_tel_top(lib.RT_LAG_MAX, int(lag.max()))
        lib.route_net_step(learner, LB[j].p, m, K, R, ND, scale, cfg['run']['rate'], world)
        lib.route_net_publish(learner, actor)
        slots = (4 * taken[:, None] + np.arange(4)).ravel()
        dist.pool.add(ro.state[slots])
        ro.begin(taken, dist.draw(len(taken))[0], version=lib.route_net_steps(learner))
        learned += 1; j ^= 1
        lib.route_tel_inc(lib.RT_STEPS, 1); lib.route_tel_reach(lib.RT_PHASE_FIRST_STEP)
        lib.route_tel_put(lib.RT_RUNS_IN_FLIGHT, int((~ro.ended).sum()))
    for B in LB:
        B.wait()
    lib.route_sims_free(ro.sims)


def evaluate(lib, learner, evaluator, cases):
    """each case's four rollouts under the learner's theta (published into the evaluator): completion over the floor"""
    lib.route_net_publish(learner, evaluator)
    ro = Rollouts(lib, evaluator, len(cases), 1, halves=1)
    ro.begin(np.arange(len(cases)), cases)
    ro.run_out()
    costs, _ = ro.costs(np.arange(ro.n))
    lib.route_sims_free(ro.sims)
    return costs['throughput'].reshape(-1, 4)
