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


def least(total, grain, low, high, stretches, value=None):
    """Units of `total` at `grain` within [low_i, high_i] minimising sum_s c_s max_i f_si(u_i) over `stretches`
    [(c_s, [f_si, nondecreasing])], or `value(u)` where given (a call's latency path, _path): one stretch by min_max
    (exact [Ibaraki & Katoh 1988]); two ranks by trying every split (exact); more, each grain to the rank that adds
    least, then a grain moved between two ranks while that lowers the sum.  (units, the sum.)"""
    n = len(low)
    if len(stretches) == 1 and value is None:
        c, fs = stretches[0]
        units, top = min_max(total, grain, low, high, fs)
        return units, c * top
    value = value or (lambda u: sum(c * max(fs[i](u[i]) for i in range(n)) for c, fs in stretches))
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
            v = (sum(c * max(max(h[:i] + h[i + 1:], default=0.0), fs[i](u[i] + grain)) for (c, fs), h in zip(stretches, held))
                 if stretches else value(u[:i] + [u[i] + grain] + u[i + 1:]))
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


def _path(rows):
    """A call's time from its stretches `rows` [(c, order, latency, [a rank's seconds, None where it publishes none])]:
    each program's stretches (order: (program, place)) in place order as a longest path [the dependency graph's
    critical path; Baccelli, Cohen, Olsder & Quadrat 1992, max-plus], a stretch starting on a rank once its own
    previous one has ended and every other rank that published there has ended and crossed (the previous stretch's
    latency, seconds), the program's count times its end and its last crossing; a stretch without an order its count
    times its slowest rank.  With every latency 0 it is the sum of each stretch's slowest rank."""
    total, programs = 0.0, {}
    for c, order, latency, times in rows:
        if order is None:
            total += c * max((t for t in times if t is not None), default=0.0)
        else:
            programs.setdefault(order[0], []).append((order[1], c, latency, times))
    for seq in programs.values():
        seq.sort(key=lambda r: r[0])
        n = max(len(r[3]) for r in seq)
        end, crossed, sent = [0.0] * n, 0.0, [False] * n
        for _, _, latency, times in seq:
            done = sorted(((end[j] + crossed, j) for j in range(n) if sent[j]), reverse=True)[:2]
            end = [max([end[i]] + [t for t, j in done if j != i][:1]) + (times[i] or 0.0) for i in range(n)]
            sent, crossed = [t is not None for t in times], latency or 0.0
        total += max(c for _, c, _, _ in seq) * (max(end) + crossed)
    return total


class Balancer:
    """A program's partitioned dimensions (`decisions`) balanced on its own evidence (design/heterogeneity.md R3,
    R5, R10, R11), whatever the program computes.

    A call runs stretches: a rank's own work between two consecutive collectives (never inside one: R2), named by
    the program (its sequence of collectives and the place in it), each run some count of invocations a call.  A
    stretch's time on a rank scales with the units the rank holds of the decisions the program names for it.  The
    call's time is the sum over its stretches of their count times the slowest rank's time, or, where the stretches
    carry their place in their program (`order`: program, place) and their crossing's latency (`latency`, seconds),
    each program's longest path (`_path`): a stretch starts on a rank once its own previous stretch has ended and
    every other rank's has ended and crossed, so where the slower rank alternates between consecutive stretches the
    call is the cross cycle (one rank's stretch, a crossing, the other's, a crossing) that the sum misses.  Each
    decision's parts minimise its stretches' part of that time (its programs' whole paths where priced, every
    decision on one path then moving alone in a call, its gain weighed against its own stretches' sum), a stretch's
    time on a rank divided among its decisions by the weights the program gives (any units: its own estimate of each
    one's work there), less the work it names no decision for (`fixed`, the same units).  A decision's parts are its groups' (`groups`, the ranks holding each part,
    default one rank a part): a part some coordinate's ranks hold together (a pipeline stage's layers), its time a
    stretch their slowest's; a rank in none of its groups is constant to it, as is a rank holding none of it, and a
    part none of whose ranks run a stretch is no part of that stretch.  A stretch a decision has seen and no rank now
    runs (its parts' ranks hold none: an emptied stage) stays in its models at the count, remainders and floor it
    last had, so a move back is priced on what it measured.

    The models are DFPA's [Lastovetsky & Reddy 2010, partial estimation of functional performance models]: each
    (decision, stretch, rank)'s measured points (units, seconds an invocation), up to `window` at a part, their median
    [LB-BSP 2020, §3.2.1]; the time interpolated between them, nondecreasing (`time`), within a doubling trust region
    past the largest part measured, never less than a grain past it (a region that has halved to the measured parts
    cannot move them, and a decision whose models promise a gain then stood without a step) [Conn, Gould & Toint
    2000]; a rank with none at a stretch read at the measured
    ranks' median time a unit.  A model's median at a part moving by more than the resolution drops its other points
    (its performance changed [Clarke, Lastovetsky & Rychkov 2011]).  A move the next call measures worse by more than
    the resolution is rejected: the parts go back, the trust region halves and the measurement stays in the models
    (a trust-region method's ratio test [Conn, Gould & Toint 2000, §6.1]); it doubles again, to twice, after a move
    that held.  The test reads the whole call (every stretch's slowest rank): a move can carry work across stretches
    (a rank holding none of a decision publishes nothing where it did), so the moved decision's own stretches are no
    measure of it.  Of the decisions sharing a stretch one moves a call, a rejection's return first, else the one whose
    models promise the most (block coordinate descent [Tseng 2001]): another's move would change the stretches its
    ratio test and its models read; one that waits is weighed again at the next call, not read as standing.  A decision's parts stand
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
                order = tuple(obs['order']) if obs.get('order') is not None else None
                seen[s] = (max(len(ts) for ts in obs['times'] if ts), ranks, scales, list(obs.get('fixed') or []), order,
                           float(obs.get('latency') or 0.0), len(obs['times']))
        priced = any(v[4] is not None for v in seen.values())
        measure = 'path' if priced else 'sum'

        def rows(times, keep=None):
            return [(c, order, latency, times(s, ranks, n)) for s, (c, ranks, _, _, order, latency, n) in seen.items()
                    if keep is None or s in keep]

        medians = lambda s, ranks, n: [ranks[i][0] if i in ranks else None for i in range(n)]
        call = _path(rows(medians))
        whole = max(0.01, 2 * math.sqrt(sum((c * max(r.values())[1]) ** 2 for c, r, *_ in seen.values())) / call if call else 0.0)
        proposals, runs = [], {}
        for name, d in self.decisions.items():
            groups = self._groups(d)
            mine = {}
            for s, (c, ranks, scales, fixed, *_) in seen.items():
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
            path = {s for s, v in seen.items() if priced and v[4] is not None and any(seen[t][4] and seen[t][4][0] == v[4][0] for t in mine)}
            counted = runs[name] = set(mine) | path
            measured = sum(c * max(med for med, _ in r.values()) for c, r, *_ in mine.values())
            slowest = [(c, max(r.values())) for c, r, *_ in mine.values()]
            within = max(0.01, 2 * math.sqrt(sum((c * se) ** 2 for c, (_, se) in slowest)) / measured if measured else 0.0)
            base = _path(rows(medians, counted)) if path else measured
            history = d.setdefault('history', {})
            changed = d.get('measure', 'sum') != measure
            if changed:
                history.clear()
                d['last'] = None
            d['measure'] = measure
            key = ','.join(map(str, d['parts']))
            history[key] = (history.get(key, []) + [call])[-5:]
            spread = [abs(f / statistics.median(fs) - 1) for fs in history.values() if len(fs) > 1 for f in fs]
            epsilon, between = within, max(whole, 2 * 1.4826 * statistics.median(spread) if len(spread) > 2 else 0.0)
            last, d['last'] = d.get('last'), [list(d['parts']), call]
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
            if last and last[0] != d['parts'] and call > last[1] * (1 + between):
                proposals.append((math.inf, name, last))
                continue
            if last and last[0] != d['parts']:
                d['reach'] = min(2.0, d.get('reach', 2.0) * 2)
            if d['before'] is not None and d['before'] == d['parts'] and not changed:
                d['stands'] = True
                continue
            n = len(d['parts'])
            kept, remembered, ranked = [], d.setdefault('seen', {}), {}
            for s, (c, rests, other) in remembered.items():
                if s in mine:
                    continue
                rests = dict((int(g), r) for g, r in rests)
                kept.append((c, [(lambda u, o=other: o) if d['parts'][g] or g not in rests else
                                 (lambda u, m=self._model(name, s, g), r=rests[g], o=other: max(o, m(u) + r) if u else o)
                                 for g in range(n)]))
            built = list(kept)
            for s, (c, ranks, scales, fixed, view, other) in mine.items():
                fs, gs = [], {}
                per_unit = [self._model(name, s, g)(d['parts'][g]) / d['parts'][g] for g in view if d['parts'][g]]
                per_unit = sorted(per_unit)[len(per_unit) // 2] if per_unit else 0.0
                for g in range(n):
                    if g not in view and d['parts'][g]:
                        fs.append(lambda u, o=other: o)
                        continue
                    med = view[g][0][0] if g in view else 0.0
                    rest = (med - self._model(name, s, g)(d['parts'][g]) if d['parts'][g] else med) if g in view else 0.0
                    if (name, s, g) in self.points:
                        gs[g] = lambda u, m=self._model(name, s, g), r=rest: m(u) + r if u else r
                    else:
                        gs[g] = lambda u, r=rest, k=per_unit: k * u + r
                    fs.append(lambda u, f=gs[g], o=other: max(o, f(u)))
                remembered[s] = [c, [[g, view[g][0][0] - self._model(name, s, g)(d['parts'][g])] for g in view if d['parts'][g]], other]
                built.append((c, fs))
                ranked[s] = gs
            value = None
            if path:
                member = {i: g for g, members in enumerate(groups) for i in members}

                def value(u, ranked=ranked, member=member, kept=kept, counted=counted):
                    def at(s, ranks, n):
                        if s not in ranked:
                            return medians(s, ranks, n)
                        return [(ranked[s][member[i]](u[member[i]]) if i in ranks or u[member[i]] else None)
                                if member.get(i) in ranked[s] else (ranks[i][0] if i in ranks and i not in member else None)
                                for i in range(n)]
                    return _path(rows(at, counted)) + sum(c * max(f(u[g]) for g, f in enumerate(fs)) for c, fs in kept)
            total = sum(d['parts'])
            if least(total, d['grain'], d['low'], d['high'], [] if value else built, value)[1] >= base - epsilon * measured:
                d['before'], d['stands'] = list(d['parts']), True
                continue
            grow = d.get('reach', 2.0)
            failed = d.get('failed') or [None] * n
            reach = [min(d['high'][g], max(d['grain'] * math.ceil(grow * u / d['grain']), u + d['grain']))
                     if any((name, s, g) in self.points for s in mine) else d['high'][g] for g in range(n)
                     for u in [max((u for s in mine for u, _ in self.points.get((name, s, g), [])), default=0)]]
            reach = [r if f is None else max(min(r, f - d['grain']), d['parts'][g]) for g, (r, f) in enumerate(zip(reach, failed))]
            nxt, promise = least(total, d['grain'], d['low'], reach, [] if value else built, value)
            if sum(nxt) < total:
                nxt, promise = least(total, d['grain'], d['low'], [d['high'][g] if f is None else max(f - d['grain'], d['parts'][g])
                                                                  for g, f in enumerate(failed)], [] if value else built, value)
            if d['before'] is not None and any((c - p) * (p - b) < 0 for c, p, b in zip(nxt, d['parts'], d['before'])):
                half = [(p + c) / 2 for c, p in zip(nxt, d['parts'])]
                nxt, promise = least(total, d['grain'], [math.floor(h / d['grain']) * d['grain'] for h in half],
                                     [math.ceil(h / d['grain']) * d['grain'] for h in half], [] if value else built, value)
            proposals.append(((base - promise) / measured if measured else 0.0, name, nxt))
        taken = set()
        for gain, name, value in sorted(proposals, key=lambda p: -p[0]):
            d, reverting = self.decisions[name], gain == math.inf
            if runs[name] & taken and not reverting:
                d['before'], d['stands'] = None, False
                continue
            taken |= runs[name]
            if reverting:
                d['reach'] = max(1.0, d.get('reach', 2.0) / 2)
                d['before'], d['parts'], d['stands'], d['last'] = None, list(value[0]), False, value
            else:
                d['before'], d['parts'], d['stands'] = list(d['parts']), list(value), False
        return self.parts

    def reject(self):
        """The call at the current parts did not complete (a placement a program cannot run): every decision that moved
        goes back to its last measured parts and its trust region halves, as a move measured worse does, and each
        part that grew past its last measured one is held below where it failed (`failed`), which the region's grain
        past the measured parts would otherwise propose again."""
        for d in self.decisions.values():
            last = d.get('last')
            if last and last[0] != d['parts']:
                d['failed'] = [p if p > q else None for p, q in zip(d['parts'], last[0])]
                d['reach'] = max(1.0, d.get('reach', 2.0) / 2)
                d['before'], d['parts'], d['stands'] = None, list(last[0]), False
        return self.parts

    def _model(self, name, s, i):
        return time(_median_points(self.points.get((name, s, i), [])))

    def state(self):
        return {'window': self.window, 'decisions': self.decisions,
                'points': [[d, s, i, own] for (d, s, i), own in self.points.items()]}

    @classmethod
    def of(cls, state):
        b = cls({}, state.get('window', 3))
        b.decisions = {name: {'grain': 1, 'low': [0] * len(d['parts']), 'high': [sum(d['parts'])] * len(d['parts']), 'before': None,
                              'stands': False, **d, 'groups': d.get('groups') or [[i] for i in range(len(d['parts']))]}
                       for name, d in state['decisions'].items()}
        b.points = {(d, s, int(i)): [tuple(p) for p in own] for d, s, i, own in state.get('points', [])}
        return b


def _solve(matrix, vector):
    """x with matrix x = vector (a small symmetric positive definite system), by Gaussian elimination with partial
    pivoting; None where it is singular."""
    n = len(vector)
    m = [list(map(float, row)) + [float(v)] for row, v in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _lattice(total, grain, low, high):
    """Every allocation of `total` units in grains within [low_i, high_i]."""
    n = len(low)

    def place(i, left):
        if i == n - 1:
            if low[i] <= left <= high[i]:
                yield (left,)
            return
        for p in range(low[i], min(high[i], left) + 1, grain):
            for rest in place(i + 1, left - p):
                yield (p,) + rest
    return list(place(0, total))


class Coupled:
    """A decision among members that run at once and slow each other (a node's engines, sharing its power, cores and
    memory: torch_mesh/engines.py), solved jointly and globally rather than member by member.

    Model.  Member i, holding units (p_i > 0), takes t_i(p) = d_i + a_i p_i + sum_{j != i} c_ij p_i p_j: a fixed cost,
    its own work, and its work slowed in proportion to each co-runner's (power, cores and memory shared while both
    run: a multiplicative slowing, which an additive term in p_j alone cannot state and which, the units summing to the
    total, would be collinear with 1 and p_i); a member holding none takes nothing.  The call takes T(p) = max_{i: p_i > 0} t_i(p) + g(S), S = {i: p_i > 0},
    g(S) the call's measured time past its slowest member for that set (a prologue, the combine; a set not yet
    measured at none, optimism under uncertainty [Auer et al. 2002]: a set is tried where it promises better, and kept
    only where it measures better).  Each member's coefficients are the least-squares fit, with a small ridge on the interference terms
    toward none [Hoerl & Kennard 1970], to the median times of every allocation it ran, every allocation kept (one run
    again pools with its earlier calls); a_i is at least a tenth of its smallest measured time a unit.

    The parts.  The model's minimum over the whole lattice (every allocation of the units in grains within the bounds,
    enumerated: a node's few members give hundreds to thousands of points), so no member's move is judged by the
    others' stale models and none conflicts with another's.  Until each member's coefficients are determined, the
    next allocations are a design about the start: each member given a step from each other, every pair both ways
    [a two-level design about a centre point, Box, Hunter & Hunter 2005, ch. 5]; a member excluded or a pair forbidden
    lays the design out again about the allocation moved to.  An allocation runs `window` calls
    before it counts.  The model's minimum, unmeasured, runs next; measured, the parts stand there, unless an
    allocation measured better by more than the resolution (twice its call time's relative standard error, at least
    1 %), where they stand instead: a measurement outranks the model's prediction.  Members that occupy one physical
    unit are forbidden to hold units together (`forbid`), so no allocation time-shares a unit between two of them.  A standing allocation whose call
    time moves past the resolution from the median it stood at is solved again (its performance changed [Clarke,
    Lastovetsky & Rychkov 2011]).

    The total.  A decision whose total is another decision's part (a member's units of a decision among members,
    divided among the member's engines) follows it (`resize`): the model is in units, so every allocation measured at
    any total stays evidence, and the parts at the new total are the model's minimum over its lattice (the design about
    the parts scaled to it, where the model is undetermined).  Only allocations of the current total are candidates
    (the measured best, the standing).  The member's time at each part the other decision tries is then its best
    division's, which is all that decision's time depends on: the two solve the joint problem exactly, the division
    being the member's alone.

      c = Coupled(3456, 128, [0, 0, 0], [3456, 3456, 3456], [1152, 1152, 1152])
      c.observe([[t, ...], [t, ...], [t, ...]], [call, ...])   # each member's seconds, the calls' seconds
      c.parts -> the next allocation
      c.resize(4096) -> the next allocation of 4096 units
      Coupled.of(c.state()) -> the same solver (JSON between processes)

    Shape.  Where a decision states its rows and each member's tile (`rows`, `tiles`: an engine computes whole tiles), a
    member's work is its whole tiles' units, not its units, and the fit pools other decisions' measured allocations
    of the same members and tiles (`pool`, their `entries`: other totals, rows or configurations): one model of the
    node's operation, not one memo an identity, so a decision at a new shape starts at the model's minimum (`prior`)
    and its own measurements then outrank it."""

    def __init__(self, total, grain, low, high, parts, window=3, ridge=1e-3, rows=None, tiles=None):
        self.total, self.grain, self.window, self.ridge = int(total), int(grain), int(window), float(ridge)
        self.low, self.high = [int(v) for v in low], [int(v) for v in high]
        self.parts, self.start = tuple(int(p) for p in parts), tuple(int(p) for p in parts)
        self.able = [i for i, h in enumerate(self.high) if h > 0]
        self.seen, self.stands, self.stood, self.forbidden = {}, False, None, set()
        self.rows, self.tiles, self.pool = rows, [list(t) for t in tiles] if tiles else None, []
        self.design = self._design(self.start)

    def state(self):
        """The solver as JSON data (Coupled.of)."""
        return {'total': self.total, 'grain': self.grain, 'window': self.window, 'ridge': self.ridge, 'low': self.low,
                'high': self.high, 'parts': list(self.parts), 'start': list(self.start), 'able': self.able,
                'stands': self.stands, 'stood': self.stood, 'forbidden': sorted(self.forbidden),
                'seen': [[list(p), e] for p, e in self.seen.items()], 'design': [list(p) for p in self.design],
                'rows': self.rows, 'tiles': self.tiles}

    @classmethod
    def of(cls, state):
        c = cls.__new__(cls)
        c.total, c.grain, c.window, c.ridge = state['total'], state['grain'], state['window'], state['ridge']
        c.low, c.high, c.able = list(state['low']), list(state['high']), list(state['able'])
        c.parts, c.start = tuple(state['parts']), tuple(state['start'])
        c.stands, c.stood = state['stands'], state['stood']
        c.forbidden = {tuple(f) for f in state['forbidden']}
        c.seen = {tuple(p): e for p, e in state['seen']}
        c.design = [tuple(p) for p in state['design']]
        c.rows, c.tiles, c.pool = state.get('rows'), state.get('tiles'), []
        return c

    def work(self, i, units, rows=None):
        """Member i's work holding `units` at `rows` rows: its units, or, where the decision states each member's tile
        (`tiles`: [rows, units] a member), the units of its whole tiles, rows and units padded up (an engine computes
        whole tiles: what a share costs follows its tiles, not its units)."""
        if not self.tiles or units <= 0:
            return float(units)
        rows = self.rows if rows is None else rows
        along, across = self.tiles[i]
        return float(-(-rows // along) * along * -(-units // across) * across) / float(self.rows or rows)

    def entries(self):
        """This decision's measured allocations as evidence another decision of the same members and tiles pools in
        (`pool`): each its rows, parts, members' medians and the call's median."""
        return [{'rows': self.rows, 'parts': list(p), 'members': [self._median(m) if m else None for m in e['members']],
                 'call': self._median(e['calls'])} for p, e in self._measured().items()]

    def prior(self):
        """Where this decision has measured nothing and the evidence it pools determines a model: its parts the
        model's minimum over its lattice (a decision at a new total, row count or configuration starts where the
        node's evidence of the same operation puts it); the parts."""
        if self.seen:
            return list(self.parts)
        fitted = self.fit()
        lattice = [p for p in _lattice(self.total, self.grain, self.low, self.high) if self.allowed(p)]
        if fitted is not None and lattice:
            self.parts = self.start = min(lattice, key=lambda p: (self.predict(p, fitted), p))
            self.design = self._design(self.start)
        return list(self.parts)

    def resize(self, total):
        """The total moved to `total` units (a multiple of the grain): each bound that was the whole total becomes the
        new one; the next allocation is the model's minimum over the new lattice, or, undetermined, the parts scaled to
        it and the design about them."""
        total = int(total)
        if total == self.total:
            return list(self.parts)
        if total % self.grain:
            raise ValueError(f'coupled: total {total} is not in grains of {self.grain}')
        old, self.total = self.total, total
        self.high = [total if h >= old else min(h, total) for h in self.high]
        self.low = [min(lo, total) for lo in self.low]
        scaled = [self.grain * round(p * total / old / self.grain) if old else 0 for p in self.parts]
        biggest = max(self.able, key=lambda i: (scaled[i], -i))
        scaled[biggest] += total - sum(scaled)
        self.start, self.stands, self.stood = tuple(scaled), False, None
        self.design = self._design(self.start)
        fitted = self.fit()
        lattice = [p for p in _lattice(self.total, self.grain, self.low, self.high) if self.allowed(p)]
        if fitted is not None and lattice:
            self.parts = min(lattice, key=lambda p: (self.predict(p, fitted), p))
        else:
            self.parts = self.start if self.start in lattice else (self.design[0] if self.design else lattice[0])
        return list(self.parts)

    def _partners(self, i):
        """The able members allowed to hold units beside member i."""
        return [j for j in self.able if j != i and (min(i, j), max(i, j)) not in self.forbidden]

    def _design(self, start):
        """The design about `start`, each able member identifiable among the allocations allowed: for each member a
        base (`start` with its forbidden partners' units given to it, and a step from the largest partner where it held
        none) and the base with a step of units moved each way between it and each allowed partner, every allocation
        within the bounds."""
        step = max(self.grain, self.grain * round(self.total / self.grain / 8))
        inside = lambda p: all(lo <= v <= hi for v, lo, hi in zip(p, self.low, self.high)) and self.allowed(p)
        design = [start] if inside(start) else []
        for i in self.able:
            base = list(start)
            for j in self.able:
                if j != i and j not in self._partners(i):
                    base[i], base[j] = base[i] + base[j], 0
            if base[i] == 0:
                donor = max(self._partners(i) or [j for j in self.able if j != i], key=lambda j: (base[j], -j), default=None)
                if donor is None or base[donor] < step:
                    continue
                base[i], base[donor] = step, base[donor] - step
            for p in [base] + [[v + (step if k == a else -step if k == b else 0) for k, v in enumerate(base)]
                               for j in self._partners(i) for a, b in ((i, j), (j, i))]:
                if inside(p) and tuple(p) not in design:
                    design.append(tuple(p))
        return design

    def allowed(self, p):
        """Whether allocation p gives units to no two members forbidden to run together."""
        return not any(p[i] > 0 and p[j] > 0 for i, j in self.forbidden)

    def forbid(self, i, j):
        """Members i and j never hold units together from now on (they occupy one physical unit, which running at once
        would time-share: two engines on the CPU's cores, two programs on one GPU): every allocation giving both units
        leaves the lattice, the design and the evidence's candidates; the current one, where it gives both, moves j's
        units to i, and the decision is solved again."""
        self.forbidden.add((min(i, j), max(i, j)))
        if not self.allowed(self.parts):
            parts = list(self.parts)
            parts[i], parts[j] = parts[i] + parts[j], 0
            self.parts, self.start, self.stands = tuple(parts), tuple(parts), False
        self.design = self._design(self.parts)
        return list(self.parts)

    def bound(self, i, high):
        """Member i holds at most `high` units from now on (an allocation giving it more failed a requirement the times
        do not show: an operation's accuracy, a share that overran its wait, either growing with its units): the
        allocations past it leave the evidence and the lattice, the current parts give their excess to the member that
        measured fastest a unit beside it, and the decision is solved again; a bound of none excludes it."""
        high = int(high) // self.grain * self.grain
        if high <= 0:
            return self.exclude(i)
        self.high[i] = min(self.high[i], high)
        self.seen = {p: e for p, e in self.seen.items() if p[i] <= self.high[i]}
        parts = list(self.parts)
        if parts[i] > self.high[i]:
            others = [j for j in self.able if j != i]
            rate = {j: min((self._median(e['members'][j]) / p[j] for p, e in self._measured().items() if p[j] > 0 and e['members'][j]),
                           default=math.inf) for j in others}
            target = min(others, key=lambda j: (rate[j], j))
            parts[target] += parts[i] - self.high[i]
            parts[i] = self.high[i]
        self.parts, self.start, self.stands = tuple(parts), tuple(parts), False
        self.design = self._design(self.start)
        return list(self.parts)

    def exclude(self, i):
        """Member i holds no units from now on (it failed a requirement the times do not show: an operation's
        accuracy): its bound goes to 0, its units to the member that measured fastest a unit beside it, the
        allocations it held leave the evidence, and the decision is solved again."""
        self.high[i], self.able = 0, [j for j in self.able if j != i]
        parts = list(self.parts)
        rate = {j: min((self._median(e['members'][j]) / p[j] for p, e in self._measured().items() if p[j] > 0 and e['members'][j]),
                       default=math.inf) for j in self.able}
        target = min(self.able, key=lambda j: (rate[j], j))
        parts[target], parts[i] = parts[target] + parts[i], 0
        self.parts, self.start, self.stands = tuple(parts), tuple(parts), False
        self.seen = {p: e for p, e in self.seen.items() if p[i] == 0}
        self.design = self._design(self.start)
        return list(self.parts)

    def observe(self, times, calls):
        """The calls at the current parts: `times` each member's seconds a call (empty where it holds nothing),
        `calls` the calls' seconds."""
        entry = self.seen.setdefault(self.parts, {'members': [[] for _ in self.low], 'calls': []})
        for i, ts in enumerate(times):
            entry['members'][i].extend(float(t) for t in ts or [])
        entry['calls'].extend(float(t) for t in calls or [])
        self.parts = self._next()
        return list(self.parts)

    def _median(self, values):
        values = sorted(values)
        return values[len(values) // 2] if values else None

    def _error(self, values):
        if len(values) < 2:
            return 1.0
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        return sd / math.sqrt(len(values)) / mean if mean else 1.0

    def _measured(self):
        return {p: e for p, e in self.seen.items() if len(e['calls']) >= self.window}

    def _evidence(self):
        """Every measured allocation this decision fits: its own (at its rows) and those it pools (`pool`, other
        decisions' entries: the same members and tiles at other totals, rows or configurations), each (rows, parts,
        members' medians, call's median)."""
        own = [(self.rows, p, [self._median(m) if m else None for m in e['members']], self._median(e['calls']))
               for p, e in self._measured().items()]
        return own + [(x['rows'], tuple(x['parts']), x['members'], x['call']) for x in self.pool]

    def fit(self):
        """Each able member's coefficients (d, a, [c_ij]) and g(S), or None where a member's are undetermined: least
        squares over the evidence (its own and pooled), each member's time a line in its work and its partners'."""
        n, evidence = len(self.low), self._evidence()
        models = {}
        for i in self.able:
            rows = [(r, p, m[i]) for r, p, m, _ in evidence if p[i] > 0 and m[i] is not None]
            others = self._partners(i)
            if len({(r, p) for r, p, _ in rows}) < 2 + len(others):
                return None
            features = [self._features(p, i, others, r) for r, p, _ in rows]
            k = len(features[0])
            normal = [[sum(f[a] * f[b] for f in features) + (self.ridge if a == b and a >= 2 else 0.0) for b in range(k)] for a in range(k)]
            target = [sum(f[a] * t for f, (_, _, t) in zip(features, rows)) for a in range(k)]
            x = _solve(normal, target)
            if x is None:
                return None
            floor = min(t / f[1] for f, (_, _, t) in zip(features, rows) if f[1] > 0) / 10
            x[1] = max(x[1], floor)
            models[i] = (x, others)
        overhead = {}
        for r, p, members, call in evidence:
            held = [members[i] for i in range(n) if p[i] > 0 and members[i] is not None]
            if held and call is not None:
                overhead.setdefault(frozenset(i for i in range(n) if p[i] > 0), []).append(call - max(held))
        return models, {s: self._median(v) for s, v in overhead.items()}

    def _features(self, p, i, others, rows=None):
        scale = float(self.total)
        w = [self.work(j, p[j], rows) / scale for j in range(len(p))]
        return [1.0, w[i]] + [w[i] * w[j] for j in others]

    def predict(self, p, fitted):
        models, overhead = fitted
        active = frozenset(i for i in range(len(p)) if p[i] > 0)
        times = []
        for i in active:
            x, others = models[i]
            times.append(sum(c * f for c, f in zip(x, self._features(p, i, others))))
        g = overhead.get(active, 0.0)
        return max(times) + g

    def _next(self):
        measured = self._measured()
        if self.parts not in measured:
            return self.parts
        if self.stands:
            e = measured[self.parts]
            now = self._median(e['calls'][-self.window:])
            if abs(now - self.stood) <= max(0.01, 2 * self._error(e['calls'])) * self.stood:
                return self.parts
            self.stands, e['calls'], e['members'] = False, e['calls'][-self.window:], [m[-self.window:] for m in e['members']]
        pending = [p for p in self.design if p not in measured and self.allowed(p)]
        fitted = self.fit()
        if fitted is None:
            return pending[0] if pending else self.parts
        best = min((p for p in measured if self.allowed(p) and sum(p) == self.total), key=lambda p: self._median(measured[p]['calls']))
        proposal = min((p for p in _lattice(self.total, self.grain, self.low, self.high) if self.allowed(p)),
                       key=lambda p: (self.predict(p, fitted), p))
        if proposal not in measured:
            return proposal
        chosen = proposal
        best_time, proposal_time = self._median(measured[best]['calls']), self._median(measured[proposal]['calls'])
        resolution = max(0.01, 2 * self._error(measured[proposal]['calls']))
        if proposal_time - best_time > resolution * best_time:
            chosen = best
        self.stands, self.stood = True, self._median(measured[chosen]['calls'])
        return chosen
