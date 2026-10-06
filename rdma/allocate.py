"""Online share allocator: writes the configuration operand of a mesh function's next call.

metal-microbench docs/principles.md §4, "Adaptive configuration without synchronization or cold
start".  A share group of W units at grain g is split over n ranks from a given config_t0.  Rank
i's segment time is nearly linear in its share, t_i = a_i + b_i x_i with x_i = kappa c_i: c_i the
units it applied in the call, kappa the call's cost scale (1 for columns, MB per unit for KV
positions).  After each call every caller passes the same segment times and gets the same shares
back, so the next call's configuration is one more value in the dataflow: no root, no lock.

1. Recursive least squares with forgetting factor lambda on theta_i = (a_i, b_i), regressor
   u = (1, x_i) [Haykin 2014, Table 10.1]: pi = P u, k = pi / (lambda + u'pi),
   theta += k (t_i - theta'u), P = (P - k pi') / lambda.  trace(P) is capped at its prior's, the
   bound the constant-trace modification holds it at [Goodwin & Sin 1984], so P cannot wind up
   while a share stands still and still shrinks under excitation.  b_i is floored at the
   roofline time per unit.
2. g_i = t_i + b_i (kappa' s_i - kappa c_i): the observed time moved along the model to the
   continuous iterate s at the next call's scale kappa'.  It is the gradient of
   F(s) = sum_i integral t_i, whose minimum on sum s = W is equal finish [Beckmann 1956]; at
   kappa' = kappa and s = c it is the observed time itself.
3. A projected online gradient step preconditioned by b [Zinkevich 2003]:
   s'_i = s_i - eta (g_i - mu) / (b_i kappa'), mu keeping sum s' = W.  It is equal_finish below
   with a_i = eta g_i - b_i kappa' s_i and T = eta mu; eta = 1 is the equal-finish solve.
4. c_i = low_i + g floor((s'_i - low_i) / g), the leftover grains to the largest remainders,
   ties to the lower rank.  s' stays the iterate; c is the next call's operand and regressor.

Measured, not modelled (design/heterogeneity.md R3, R10, R11): `dfpa` is one iteration of DFPA [Lastovetsky & Reddy
2010, Distributed data partitioning for heterogeneous processors based on partial estimation of their functional
performance models]: each rank's model is the points it measured, (units, seconds) at the parts it ran (`time`: DFPA
interpolates the speed d / t and holds it constant past the points; here the time, nondecreasing, which keeps a
rank whose time is flat at small parts from reading as slow at large ones); the next parts are `min_max`'s on those
times, each grain to the rank finishing earliest with it [Ibaraki & Katoh 1988; Beaumont et al. 2001, Alg. 3.1].  The
parts stay where the models promise less than the relative accuracy epsilon under the measured largest finish
[Meta-Balancer 2012: rebalance only when the gain exceeds the
cost: here, the measured finishes' agreement within epsilon [DFPA] is the case of no gain], and once a part moves
back the way it came every part moves half as far [LB-BSP 2020, §3.3.2].  A part grows
at most to twice the largest it was measured at, a trust region [Conn, Gould & Toint 2000] over the model's
extrapolation past its points; the gain is the models' optimum's, unbounded, since a step inside the region can
leave the largest finish where it was (a rank whose time is flat in its part).  A rank's point at a
part it ran before replaces that point, and where the two differ by more than epsilon its older points go (its
performance changed [Clarke, Lastovetsky & Rychkov 2011]).

equal_finish(W, g, low, high, a, b) is steps 3-4 alone: rank i finishing at a_i + b_i s_i, the
shares s_i = (T - a_i) / b_i with sum s = W; ranks outside their bounds are held at them and T
re-solved, the side with the larger total violation first [Bitran & Hax 1981].  The mesh's
programs take config_t0 from it (metal-microbench tools/mesh/programs.py).

O(n) plain floats in a fixed order: identical inputs give identical shares on every rank.
"""
import math


class Allocator:
    """shares: config_t0, units per rank (multiples of grain); W = sum(shares).  low: each rank's
    minimum (a multiple of grain, at least grain).  prior: per rank (a, b, var_a, var_b) in the
    units of the times, x at scale 1.  floor: per rank, the least b (roofline time per unit).
    forget: lambda in (0, 1].  step: eta in (0, 1].  .shares is the next call's configuration
    operand; .a and .b are the estimates, for the run record."""

    def __init__(self, shares, grain, low, prior, floor, forget, step):
        self.shares, self.grain, self.low, self.floor = list(shares), grain, list(low), list(floor)
        self.total, self.forget, self.step = sum(shares), forget, step
        if (len({len(shares), len(low), len(prior), len(floor)}) != 1 or sum(low) >= self.total
                or any(c % grain or l % grain or l < grain or c < l for c, l in zip(shares, low))
                or min(floor) <= 0 or not 0 < forget <= 1 or not 0 < step <= 1):
            raise ValueError('allocator: shares and lower bounds on the grain with room above the bounds, '
                             'one prior and positive floor per rank, forget and step in (0, 1]')
        self.s = [float(c) for c in shares]
        self.a, self.b = [float(p[0]) for p in prior], [float(p[1]) for p in prior]
        self.P = [(float(p[2]), 0.0, float(p[3])) for p in prior]
        self.trace = [p00 + p11 for p00, _, p11 in self.P]

    def __call__(self, times, scale=1.0, next_scale=1.0):
        """Steps on each rank's segment time of the call that ran .shares at cost scale `scale`;
        returns the shares of the next call, whose cost scale is `next_scale`."""
        n, lam, eta = len(self.shares), self.forget, self.step
        g, w = [], []
        for i in range(n):
            x = scale * self.shares[i]
            p00, p01, p11 = self.P[i]
            pi0, pi1 = p00 + p01 * x, p01 + p11 * x
            den = lam + pi0 + pi1 * x
            k0, k1 = pi0 / den, pi1 / den
            e = times[i] - self.a[i] - self.b[i] * x
            self.a[i] += k0 * e
            self.b[i] += k1 * e
            p00, p01, p11 = (p00 - k0 * pi0) / lam, (p01 - k0 * pi1) / lam, (p11 - k1 * pi1) / lam
            if p00 + p11 > self.trace[i]:
                f = self.trace[i] / (p00 + p11)
                p00, p01, p11 = p00 * f, p01 * f, p11 * f
            self.P[i] = (p00, p01, p11)
            b = max(self.b[i], self.floor[i])
            g.append(times[i] + b * (next_scale * self.s[i] - x))
            w.append(1.0 / (b * next_scale))
        B = [1.0 / x for x in w]
        c, self.s, _ = equal_finish(self.total, self.grain, self.low, [math.inf] * n,
                                    [eta * g[i] - B[i] * self.s[i] for i in range(n)], B)
        self.shares = c
        return list(c)


def equal_finish(total, grain, low, high, a, b):
    """Shares of `total` units at `grain` over ranks finishing at a_i + b_i s_i, within
    [low_i, high_i] (multiples of grain): (c, s, T), c the integer shares, s the continuous ones
    and T their common finish (steps 3-4 above)."""
    n, held = len(a), {}
    while True:
        free = [i for i in range(n) if i not in held]
        T = ((total - sum(held.values()) + sum(a[i] / b[i] for i in free)) / sum(1.0 / b[i] for i in free)
             if free else math.inf)
        s = [held[i] if i in held else (T - a[i]) / b[i] for i in range(n)]
        below = [i for i in free if s[i] < low[i]]
        above = [i for i in free if s[i] > high[i]]
        if not below and not above:
            break
        if sum(low[i] - s[i] for i in below) >= sum(s[i] - high[i] for i in above):
            held.update((i, float(low[i])) for i in below)
        else:
            held.update((i, float(high[i])) for i in above)
    c = [low[i] + grain * math.floor((s[i] - low[i]) / grain) for i in range(n)]
    left = (total - sum(c)) // grain  # fewer than n: each floor drops less than one grain
    for i in sorted(range(n), key=lambda i: (c[i] - s[i], i)):
        if left and c[i] + grain <= high[i]:
            c[i] += grain
            left -= 1
    return c, s, T


def min_max(total, grain, low, high, cost):
    """Shares of `total` units at `grain` within [low_i, high_i] (multiples of grain) minimising the largest of
    cost[i](c_i), each nondecreasing (a cost is taken as the largest it has reached, so a dip is never a reason to
    give a rank more): each grain in turn to the rank finishing earliest with it, ties to the lower rank
    [Ibaraki & Katoh 1988; Beaumont et al. 2001, Alg. 3.1].  (c, T); grains no rank can hold are left unplaced."""
    import heapq
    n = len(low)
    c = list(low)
    reached = [cost[i](c[i]) for i in range(n)]
    heap = [(max(reached[i], cost[i](c[i] + grain)), i) for i in range(n) if c[i] + grain <= high[i]]
    heapq.heapify(heap)
    for _ in range((total - sum(low)) // grain):
        if not heap:
            break
        t, i = heapq.heappop(heap)
        c[i] += grain
        reached[i] = t
        if c[i] + grain <= high[i]:
            heapq.heappush(heap, (max(t, cost[i](c[i] + grain)), i))
    return c, max(reached)


def time(points, prior=None):
    """A rank's time for d units from its partial model: its measured points [(units, seconds)], made nondecreasing
    in units (pool adjacent violators [Ayer et al. 1955]: more units never take less time), the time interpolated
    linearly between them and extended along the line through the two nearest past either end (between none and
    the first's time below it); with one point, in proportion below it and its own time above (the least a
    nondecreasing time can be there: optimism where nothing is known [Auer et al. 2002], which the trust region
    bounds); with none, `prior` (a, b): a + b d, or d itself."""
    blocks = []
    for d, t in sorted((d, t) for d, t in points if d > 0 and t > 0):
        blocks.append([[d], t])
        while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
            ds1, t1 = blocks.pop()
            ds0, t0 = blocks.pop()
            blocks.append([ds0 + ds1, (t0 * len(ds0) + t1 * len(ds1)) / (len(ds0) + len(ds1))])
    known = [(d, t) for ds, t in blocks for d in ds]

    def at(d):
        if d <= 0:
            return 0.0
        if not known:
            return prior[0] + prior[1] * d if prior else float(d)
        if len(known) == 1:
            return known[0][1] * min(d / known[0][0], 1.0)
        if d <= known[0][0]:
            (d0, t0), (d1, t1) = known[0], known[1]
            return min(t0, max(0.0, t0 - (t1 - t0) / (d1 - d0) * (d0 - d)))
        if d >= known[-1][0]:
            (d0, t0), (d1, t1) = known[-2], known[-1]
            return t1 + (t1 - t0) / (d1 - d0) * (d - d1)
        for (d0, t0), (d1, t1) in zip(known, known[1:]):
            if d0 <= d <= d1:
                return t0 + (t1 - t0) * (d - d0) / (d1 - d0)
    return at


def _robust(samples):
    """(median, its standard error) of seconds an invocation: 1.2533 x 1.4826 MAD / sqrt(n), which a tail of slow
    invocations does not inflate."""
    import statistics
    med = statistics.median(samples)
    mad = statistics.median(abs(t - med) for t in samples)
    return med, 1.2533 * 1.4826 * mad / math.sqrt(len(samples))


def _median_points(points):
    import statistics
    by = {}
    for d, t in points:
        by.setdefault(d, []).append(t)
    return [(d, statistics.median(ts)) for d, ts in by.items()]


def least(total, grain, low, high, stretches):
    """Units of `total` at `grain` within [low_i, high_i] minimising sum_s c_s max_i f_si(u_i) over `stretches`
    [(c_s, [f_si, nondecreasing])]: one stretch by min_max (exact [Ibaraki & Katoh 1988]); two ranks by trying
    every split (exact); more, each grain to the rank that adds least, then a grain moved between two ranks while
    that lowers the sum.  (units, the sum.)"""
    n = len(low)
    if len(stretches) == 1:
        c, fs = stretches[0]
        units, top = min_max(total, grain, low, high, fs)
        return units, c * top
    value = lambda u: sum(c * max(fs[i](u[i]) for i in range(n)) for c, fs in stretches)
    if n == 2:
        best = None
        for a in range(max(low[0], total - high[1]), min(high[0], total - low[1]) + 1, grain):
            u = [a, total - a]
            v = value(u)
            if best is None or v < best[1]:
                best = (u, v)
        return best if best else (list(low), value(low))
    u = list(low)
    held = [[fs[i](u[i]) for i in range(n)] for _, fs in stretches]
    for _ in range((total - sum(low)) // grain):
        choice = None
        for i in range(n):
            if u[i] + grain > high[i]:
                continue
            v = sum(c * max(max(h[:i] + h[i + 1:], default=0.0), fs[i](u[i] + grain)) for (c, fs), h in zip(stretches, held))
            if choice is None or v < choice[1]:
                choice = (i, v)
        if choice is None:
            break
        i = choice[0]
        u[i] += grain
        for (c, fs), h in zip(stretches, held):
            h[i] = fs[i](u[i])
    v = value(u)
    for _ in range(4 * n * n):
        moved = False
        for i in range(n):
            for j in range(n):
                if i != j and u[i] - grain >= low[i] and u[j] + grain <= high[j]:
                    w = list(u)
                    w[i] -= grain
                    w[j] += grain
                    x = value(w)
                    if x < v:
                        u, v, moved = w, x, True
        if not moved:
            break
    return u, v


class Balancer:
    """A program's partitioned dimensions (`decisions`) balanced on its own evidence (design/heterogeneity.md R3,
    R5, R10, R11), whatever the program computes.

    A call runs stretches: a rank's own work between two consecutive collectives (never inside one: R2), named by
    the program (its sequence of collectives and the place in it), each run some count of invocations a call.  A
    stretch's time on a rank scales with the units the rank holds of the decisions the program names for it.  The
    call's time is the sum over its stretches of their count times the slowest rank's time, and each decision's
    parts minimise its stretches' part of that sum, a stretch's time on a rank divided among its decisions by the
    weights the program gives (any units: its own estimate of each one's work there), less the work it names no
    decision for (`fixed`, the same units).  A decision's parts are its groups' (`groups`, the ranks holding each part,
    default one rank a part): a part some coordinate's ranks hold together (a pipeline stage's layers), its time a
    stretch their slowest's; a rank in none of its groups is constant to it, as is a rank holding none of it, and a
    part none of whose ranks run a stretch is no part of that stretch.  A stretch a decision has seen and no rank now
    runs (its parts' ranks hold none: an emptied stage) stays in its models at the count, remainders and floor it
    last had, so a move back is priced on what it measured.

    The models are DFPA's [Lastovetsky & Reddy 2010, partial estimation of functional performance models]: each
    (decision, stretch, rank)'s measured points (units, seconds an invocation), up to `window` at a part, their median
    [LB-BSP 2020, §3.2.1]; the time interpolated between them, nondecreasing (`time`), within a doubling trust region
    past the largest part measured [Conn, Gould & Toint 2000]; a rank with none at a stretch read at the measured
    ranks' median time a unit.  A model's median at a part moving by more than the resolution drops its other points
    (its performance changed [Clarke, Lastovetsky & Rychkov 2011]).  A move the next call measures worse by more than
    the resolution is rejected: the parts go back, the trust region halves and the measurement stays in the models
    (a trust-region method's ratio test [Conn, Gould & Toint 2000, §6.1]); it doubles again, to twice, after a move
    that held.  Of the decisions sharing a stretch one moves a call, a rejection's return first, else the one whose
    models promise the most (block coordinate descent [Tseng 2001]): another's move would change the stretches its
    ratio test and its models read.  A decision's parts stand
    where the models'
    unbounded optimum promises less than its resolution under the measured sum [DFPA; Meta-Balancer 2012], and stay
    standing while none of its models changed; once a part moves back the way it came, every part of the decision
    moves half as far [LB-BSP 2020, §3.3.2].  The resolution is twice the measured sum's relative standard error (each
    stretch's slowest rank's median's, from its invocations, in quadrature), at least 1 %; a model changed where its
    window's median moved past that and past three of its own standard errors; the ratio test, which compares two
    calls, takes at least twice the spread of the measured sums of repeated calls at the same parts (what changes
    between calls, a peer program's load on a GPU, which no one call shows), so a peer's load does not reject a move.  Every rank given the same evidence gets the same
    parts (R4).

      Balancer({'tp': {'parts': [8, 8], 'grain': 1, 'low': [1, 1], 'high': [15, 15]}})
      .observe({'step:3': {'scales': {'tp': [1.0, 1.0]}, 'fixed': [0.2, 0.1], 'times': [[s, ...], [s, ...]]}, ...})
      -> {'tp': [5, 11]}

    `state()` and `Balancer.of(state)` carry it between processes."""

    def __init__(self, decisions, window=3):
        self.window, self.decisions, self.points = int(window), {}, {}
        for name, d in decisions.items():
            parts = [int(p) for p in d['parts']]
            n, grain = len(parts), int(d.get('grain', 1))
            low = [int(v) for v in d.get('low') or [0] * n]
            high = [int(v) for v in d.get('high') or [sum(parts)] * n]
            groups = [[int(i) for i in g] for g in d.get('groups') or [[i] for i in range(n)]]
            if len(groups) != n or any(p % grain or not lo <= p <= hi for p, lo, hi in zip(parts, low, high)):
                raise ValueError(f'balancer: {name}: parts {parts} of {groups} on the grain {grain} within {low} and {high}')
            self.decisions[name] = {'parts': parts, 'grain': grain, 'low': low, 'high': high, 'groups': groups,
                                    'before': None, 'stands': False}

    @property
    def parts(self):
        return {name: list(d['parts']) for name, d in self.decisions.items()}

    def stands(self, name):
        return self.decisions[name]['stands']

    def _groups(self, d):
        return d.get('groups') or [[i] for i in range(len(d['parts']))]

    def _holds(self, e, i):
        d = self.decisions[e]
        return next((p for p, g in zip(d['parts'], self._groups(d)) if i in g), 0)

    def observe(self, stretches):
        import statistics
        seen = {}
        for s, obs in stretches.items():
            scales = {d: w for d, w in (obs.get('scales') or {}).items() if d in self.decisions}
            ranks = {i: _robust([float(t) for t in ts]) for i, ts in enumerate(obs['times']) if ts}
            if scales and ranks:
                seen[s] = (max(len(ts) for ts in obs['times'] if ts), ranks, scales, list(obs.get('fixed') or []))
        proposals, runs = [], {}
        for name, d in self.decisions.items():
            groups = self._groups(d)
            mine = {}
            for s, (c, ranks, scales, fixed) in seen.items():
                if name not in scales:
                    continue
                view = {}
                for g, members in enumerate(groups):
                    present = [i for i in members if i in ranks]
                    if present:
                        i = max(present, key=lambda i: ranks[i][0])
                        view[g] = (ranks[i], i)
                outside = [ranks[i][0] for i in ranks if not any(i in g for g in groups)]
                if view:
                    mine[s] = (c, ranks, scales, fixed, view, max(outside, default=0.0))
            if not mine:
                continue
            runs[name] = set(mine)
            measured = sum(c * max(med for med, _ in r.values()) for c, r, *_ in mine.values())
            slowest = [(c, max(r.values())) for c, r, *_ in mine.values()]
            within = max(0.01, 2 * math.sqrt(sum((c * se) ** 2 for c, (_, se) in slowest)) / measured if measured else 0.0)
            history = d.setdefault('history', {})
            key = ','.join(map(str, d['parts']))
            history[key] = (history.get(key, []) + [measured])[-5:]
            spread = [abs(f / statistics.median(fs) - 1) for fs in history.values() if len(fs) > 1 for f in fs]
            epsilon, between = within, max(within, 2 * 1.4826 * statistics.median(spread) if len(spread) > 2 else 0.0)
            last, d['last'] = d.get('last'), [list(d['parts']), measured]
            changed = False
            for s, (c, ranks, scales, fixed, view, _) in mine.items():
                for g, ((med, se), i) in view.items():
                    units = d['parts'][g]
                    if not units:
                        continue
                    weights = {e: max(0.0, float(w[i] or 0.0)) * bool(self._holds(e, i)) for e, w in scales.items()}
                    total = sum(weights.values()) + max(0.0, float(fixed[i] if i < len(fixed) else 0.0))
                    share = med * (weights[name] / total if total else 1.0 / len(weights))
                    own = self.points.get((name, s, g), [])
                    earlier = [t for u, t in own if u == units][-self.window:]
                    now = (earlier + [share])[-self.window:]
                    old, new = (statistics.median(earlier) if earlier else None), statistics.median(now)
                    if old is not None and abs(new - old) > max(epsilon, 3 * se / med if med > 0 else 0.0) * max(new, old):
                        own, changed = [], True
                    self.points[(name, s, g)] = [(u, t) for u, t in own if u != units] + [(units, t) for t in now]
            if last and last[0] != d['parts'] and measured > last[1] * (1 + between):
                proposals.append((math.inf, name, last))
                continue
            if last and last[0] != d['parts']:
                d['reach'] = min(2.0, d.get('reach', 2.0) * 2)
            if d['before'] is not None and d['before'] == d['parts'] and not changed:
                d['stands'] = True
                continue
            n = len(d['parts'])
            built, remembered = [], d.setdefault('seen', {})
            for s, (c, rests, other) in remembered.items():
                if s in mine:
                    continue
                rests = dict((int(g), r) for g, r in rests)
                built.append((c, [(lambda u, o=other: o) if d['parts'][g] or g not in rests else
                                  (lambda u, m=self._model(name, s, g), r=rests[g], o=other: max(o, m(u) + r) if u else o)
                                  for g in range(n)]))
            for s, (c, ranks, scales, fixed, view, other) in mine.items():
                fs = []
                per_unit = [self._model(name, s, g)(d['parts'][g]) / d['parts'][g] for g in view if d['parts'][g]]
                per_unit = sorted(per_unit)[len(per_unit) // 2] if per_unit else 0.0
                for g in range(n):
                    if g not in view and d['parts'][g]:
                        fs.append(lambda u, o=other: o)
                        continue
                    med = view[g][0][0] if g in view else 0.0
                    rest = (med - self._model(name, s, g)(d['parts'][g]) if d['parts'][g] else med) if g in view else 0.0
                    if (name, s, g) in self.points:
                        fs.append(lambda u, m=self._model(name, s, g), r=rest, o=other: max(o, m(u) + r if u else r))
                    else:
                        fs.append(lambda u, r=rest, k=per_unit, o=other: max(o, k * u + r))
                remembered[s] = [c, [[g, view[g][0][0] - self._model(name, s, g)(d['parts'][g])] for g in view if d['parts'][g]], other]
                built.append((c, fs))
            total = sum(d['parts'])
            if least(total, d['grain'], d['low'], d['high'], built)[1] >= measured * (1 - epsilon):
                d['before'], d['stands'] = list(d['parts']), True
                continue
            grow = d.get('reach', 2.0)
            reach = [min(d['high'][g], max(d['grain'] * math.ceil(grow * max((u for s in mine for u, _ in self.points.get((name, s, g), [])), default=0) / d['grain']),
                                           d['grain']))
                     if any((name, s, g) in self.points for s in mine) else d['high'][g] for g in range(n)]
            nxt, promise = least(total, d['grain'], d['low'], reach, built)
            if sum(nxt) < total:
                nxt, promise = least(total, d['grain'], d['low'], d['high'], built)
            if d['before'] is not None and any((c - p) * (p - b) < 0 for c, p, b in zip(nxt, d['parts'], d['before'])):
                half = [(p + c) / 2 for c, p in zip(nxt, d['parts'])]
                nxt, promise = least(total, d['grain'], [math.floor(h / d['grain']) * d['grain'] for h in half],
                                     [math.ceil(h / d['grain']) * d['grain'] for h in half], built)
            proposals.append(((measured - promise) / measured if measured else 0.0, name, nxt))
        taken = set()
        for gain, name, value in sorted(proposals, key=lambda p: -p[0]):
            d, reverting = self.decisions[name], gain == math.inf
            if runs[name] & taken and not reverting:
                d['stands'] = False
                continue
            taken |= runs[name]
            if reverting:
                d['reach'] = max(1.0, d.get('reach', 2.0) / 2)
                d['before'], d['parts'], d['stands'], d['last'] = None, list(value[0]), False, value
            else:
                d['before'], d['parts'], d['stands'] = list(d['parts']), list(value), False
        return self.parts

    def _model(self, name, s, i):
        return time(_median_points(self.points.get((name, s, i), [])))

    def state(self):
        return {'window': self.window, 'decisions': self.decisions,
                'points': [[d, s, i, own] for (d, s, i), own in self.points.items()]}

    @classmethod
    def of(cls, state):
        b = cls({}, state.get('window', 3))
        b.decisions = {name: {**d, 'groups': d.get('groups') or [[i] for i in range(len(d['parts']))]} for name, d in state['decisions'].items()}
        b.points = {(d, s, int(i)): [tuple(p) for p in own] for d, s, i, own in state['points']}
        return b
