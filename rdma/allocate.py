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


def dfpa(total, grain, low, high, parts, times, points, epsilon, before=None, prior=None, window=3):
    """One DFPA iteration after a call that ran `parts` (each rank's units, multiples of grain) and measured `times`
    (each rank's seconds at its part, None where it ran none): `points` each rank's earlier [(units, seconds)], up to
    `window` at a part, its time their median [LB-BSP 2020, §3.2.1: prediction robust to non-deterministic
    perturbation]; `before` the parts of the call before it (equal to `parts` where they stood); `prior` each rank's
    (a, b) for a rank with no point.  A rank's performance changed where its median at its part moved by more than
    epsilon: its points at other parts go [Clarke, Lastovetsky & Rychkov 2011].  Parts that stood stand while no
    rank's performance changed.  Returns (the next parts, each rank's points, whether the parts stand)."""
    import statistics
    n = len(parts)
    kept, changed, medians = [], False, []
    for i in range(n):
        own = list(points[i])
        if times[i] is not None and parts[i] > 0:
            earlier = [t for d, t in own if d == parts[i]][-window:]
            now = (earlier + [times[i]])[-window:]
            old, new = (statistics.median(earlier) if earlier else None), statistics.median(now)
            if old is not None and abs(new - old) > epsilon * max(new, old):
                own, changed = [], True
            own = [(d, t) for d, t in own if d != parts[i]] + [(parts[i], t) for t in now]
            medians.append(new)
        kept.append(own)
    if not medians or (before is not None and list(before) == list(parts) and not changed):
        return list(parts), kept, True
    model = [time(_median_points(kept[i]), prior[i] if prior else None) for i in range(n)]
    if min_max(total, grain, low, high, model)[1] >= max(medians) * (1 - epsilon):
        return list(parts), kept, True
    reach = [min(high[i], max(2 * max((d for d, _ in kept[i]), default=0), grain)) if kept[i] else high[i] for i in range(n)]
    nxt, _ = min_max(total, grain, low, reach, model)
    if sum(nxt) < total:
        nxt, _ = min_max(total, grain, low, high, model)
    if before is not None and any((c - p) * (p - b) < 0 for c, p, b in zip(nxt, parts, before)):
        half = [(p + c) / 2 for c, p in zip(nxt, parts)]
        nxt, _ = min_max(sum(nxt), grain, [math.floor(h / grain) * grain for h in half],
                         [math.ceil(h / grain) * grain for h in half], model)
    return nxt, kept, False


def _median_points(points):
    import statistics
    by = {}
    for d, t in points:
        by.setdefault(d, []).append(t)
    return [(d, statistics.median(ts)) for d, ts in by.items()]


class Balancer:
    """One share decision balanced on the calls' evidence (design/heterogeneity.md R3, R5, R10, R11): `parts` the
    units each rank holds now (multiples of `grain`; their sum the decision's units), within [low_i, high_i] (default
    0 and the units: a partition operand's are 1 and its capacities).  After a call, `observe` takes each rank's
    times of the call's steps at those parts (seconds, one a step; None or [] where it held none) and returns the
    parts to hold next, one DFPA iteration (`dfpa`) on the points it has kept: the same parts until the evidence
    moves them, and the same again once they stand.  Its resolution is twice the largest relative standard error
    of the ranks' medians (1.2533 x 1.4826 MAD / sqrt(steps): a tail of slow steps does not inflate it), at least
    1 %.  Every rank given the same times gets the same parts (R4).  `state()` and `Balancer.of(state)` carry it
    between processes."""

    def __init__(self, parts, grain=1, low=None, high=None, prior=None, window=3):
        n = len(parts)
        self.parts, self.grain, self.window, self.prior = [int(p) for p in parts], int(grain), int(window), prior
        self.total = sum(self.parts)
        self.low = [int(v) for v in (low if low is not None else [0] * n)]
        self.high = [int(v) for v in (high if high is not None else [self.total] * n)]
        if any(p % self.grain or not lo <= p <= hi for p, lo, hi in zip(self.parts, self.low, self.high)):
            raise ValueError(f'balancer: parts {self.parts} on the grain {self.grain} within {self.low} and {self.high}')
        self.points, self.before, self.stands = [[] for _ in range(n)], None, False

    def observe(self, samples):
        import statistics
        medians, errors = [], []
        for own, part in zip(samples, self.parts):
            own = [float(t) for t in (own or ()) if t is not None]
            if not part or not own:
                medians.append(None)
                continue
            med = statistics.median(own)
            mad = statistics.median(abs(t - med) for t in own)
            medians.append(med)
            errors.append(1.2533 * 1.4826 * mad / math.sqrt(len(own)) / med if med > 0 else 0.0)
        epsilon = max(0.01, 2 * max(errors, default=0.0))
        nxt, self.points, self.stands = dfpa(self.total, self.grain, self.low, self.high, self.parts, medians, self.points,
                                             epsilon, self.before, self.prior, self.window)
        self.before = nxt if self.stands else self.parts
        self.parts = list(nxt)
        return list(self.parts)

    def state(self):
        return {'parts': self.parts, 'grain': self.grain, 'low': self.low, 'high': self.high, 'prior': self.prior,
                'window': self.window, 'points': self.points, 'before': self.before, 'stands': self.stands}

    @classmethod
    def of(cls, state):
        b = cls(state['parts'], state['grain'], state['low'], state['high'], state.get('prior'), state.get('window', 3))
        b.points = [[tuple(p) for p in own] for own in state['points']]
        b.before, b.stands = state.get('before'), state.get('stands', False)
        return b
