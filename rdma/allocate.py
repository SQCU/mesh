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
4. c is min_max's on the costs a_i + b_i c_i: each grain to the rank finishing earliest with it, ties to the lower
   rank (design/heterogeneity.md R5; never the largest remainders, which round the slow rank up by a whole grain).
   s' stays the iterate; c is the next call's operand and regressor.

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

`least` is exact for any number of ranks: a branch and bound over boxes of the lattice [Land & Doig 1960] whose bound
is the value at the box's least corner (every model nondecreasing), started from the pairwise descent's parts.

O(n) plain floats in a fixed order: identical inputs give identical shares on every rank.
"""
import heapq
import math

RESOLUTION = 0.01  # the least relative accuracy a decision resolves (R11): a gain below it is no gain [DFPA 2011]


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
    c, _ = min_max(total, grain, low, high, [lambda u, i=i: a[i] + b[i] * u for i in range(n)])
    return c, s, T


def min_max(total, grain, low, high, cost):
    """Shares of `total` units at `grain` within [low_i, high_i] (multiples of grain, a high may be math.inf) minimising
    the largest of cost[i](c_i), each nondecreasing: each grain to the rank finishing earliest with it, ties to the
    lower rank [Ibaraki & Katoh 1988; Beaumont et al. 2001, Alg. 3.1].  That greedy takes the (total - sum low) / grain
    least of the ranks' grain costs (rank i's k-th grain costing cost[i](low_i + k grain), a sorted list a rank), so it
    is found as that order statistic of the sorted lists [Frederickson & Johnson 1982]: a pivot the weighted median of
    the lists' middle costs, every list cut at it by bisection, until the pivot is the statistic; O(n log^2(total /
    grain)) evaluations where a grain at a time took total / grain.  A cost that dips is the caller's to pass as the
    largest it has reached (its least nondecreasing majorant, which the grain-at-a-time greedy took; metal-microbench
    tools/mesh/balance_sim.py envelope): every cost here is read as sorted.  (c, T), T the largest cost, each rank's at
    its low counted; grains no rank can hold are left unplaced."""
    n = len(low)
    want = max(0, (total - sum(low)) // grain)
    top = [max(0, want if h == math.inf else min(want, (h - lo) // grain)) for lo, h in zip(low, high)]
    seen = {}

    def at(i, k):
        if (i, k) not in seen:
            seen[i, k] = cost[i](low[i] + k * grain)
        return seen[i, k]

    def cut(i, v, a, b, strict):
        while a < b:
            m = (a + b + 1) // 2
            if at(i, m) < v or not strict and at(i, m) == v:
                a = m
            else:
                b = m - 1
        return a

    floor = max((cost[i](low[i]) for i in range(n)), default=0.0)
    if sum(top) <= want:
        return [low[i] + top[i] * grain for i in range(n)], max([floor] + [at(i, top[i]) for i in range(n) if top[i]])
    if not want:
        return list(low), floor
    a, b, need = [0] * n, list(top), want
    while True:
        middles = sorted((at(i, (a[i] + b[i] + 1) // 2), i) for i in range(n) if b[i] > a[i])
        half, reach = sum(b[i] - a[i] for i in range(n)) / 2, 0
        for pivot, i in middles:
            reach += b[i] - a[i]
            if reach >= half:
                break
        below = [cut(i, pivot, a[i], b[i], True) for i in range(n)]
        through = [cut(i, pivot, a[i], b[i], False) for i in range(n)]
        if need <= sum(x - y for x, y in zip(below, a)):
            b = below
        elif need > sum(x - y for x, y in zip(through, a)):
            need -= sum(x - y for x, y in zip(through, a))
            a = through
        else:
            need -= sum(x - y for x, y in zip(below, a))
            c = []
            for i in range(n):
                tied = min(need, through[i] - below[i])
                need -= tied
                c.append(low[i] + (below[i] + tied) * grain)
            return c, max(floor, pivot)


def time(points, prior=None):
    """A rank's time for d units from its partial model: its measured points [(units, seconds)], made nondecreasing
    in units (pool adjacent violators [Ayer et al. 1955]: more units never take less time), the time interpolated
    linearly between them and extended along the line through the two nearest past either end (between none and
    the first's time below it); with one point, in proportion below it and its own time above (the least a
    nondecreasing time can be there: optimism where nothing is known [Auer et al. 2002], which the trust region
    bounds); with none, `prior` (a, b): a + b d, or d itself.  Its `knots` are the units where it bends (the
    nondecreasing points'; at 0 it steps, and below the first point it is the max of 0 and an affine piece, convex),
    which `_split` reads off the model itself."""
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
    at.knots = [d for d, _ in known]
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


def least(total, grain, low, high, stretches, value, knots, stop=None, budget=None):
    """Units of `total` at `grain` within [low_i, high_i] (each bound taken inward onto the grain) minimising
    sum_s c_s max_i f_si(u_i) over `stretches` [(c_s, [f_si, nondecreasing])], or `value(u)` where given (a call's
    latency path, _path), exactly: one stretch by min_max (exact [Ibaraki & Katoh 1988]); two ranks by the pairwise
    descent (one split, exact); more by a branch and bound [Land & Doig 1960] over boxes of the lattice from the
    descent's parts (_descend).  The value is nondecreasing in every rank's units (the models are, and max, + and
    nonnegative counts keep it so; _path is max-plus), so a box's every point is no less than the value at its least
    corner, each bound tightened by the total (u_i at least total less the others' highs, at most total less their
    lows); with `stretches` the bound is the sum of each stretch's own min_max over the box, each stretch's least
    largest time, which is no less.  Boxes are taken least bound first, each split in half along its widest rank, and
    one whose bound reaches the best value found is dropped: the best value is the lattice's least.  With `stop`, a
    question rather than a search: whether any point's value is below it, the boxes whose bound reaches it dropped too
    and the search ended at the first point below it (the Balancer's stand test).  `budget` (None: none) bounds the boxes
    the search makes (metal-microbench docs/acceptance-20261014.md A37: at six ranks of 4,096 units it made up to 54
    thousand where the descent priced 6 thousand): past it the search ends at the best point found, which is then not
    known to be the least.  (units, the value, whether it is the least: the search ended within its budget.)"""
    n = len(low)
    low, high = [-(-v // grain) * grain for v in low], [v // grain * grain for v in high]
    if len(stretches) == 1 and value is None:
        c, fs = stretches[0]
        units, top = min_max(total, grain, low, high, fs)
        return units, c * top, True
    value = value or (lambda u: sum(c * max(fs[i](u[i]) for i in range(n)) for c, fs in stretches))
    seen = {}

    def priced(u):
        key = tuple(u)
        if key not in seen:
            seen[key] = value(list(u))
        return seen[key]

    best = _descend(total, grain, low, high, priced, knots)
    if n <= 2 or (stop is not None and best[1] < stop):
        return best[0], best[1], True
    best, spent = (best[1], tuple(best[0])), 0

    def box(lo, hi):
        lo, hi = list(lo), list(hi)
        while True:
            under, over = total - sum(hi), total - sum(lo)
            tight = ([max(lo[i], -(-(under + hi[i]) // grain) * grain) for i in range(n)],
                     [min(hi[i], (over + lo[i]) // grain * grain) for i in range(n)])
            if tight == (lo, hi):
                break
            lo, hi = tight
        if any(a > b for a, b in zip(lo, hi)) or sum(lo) > total or sum(hi) < total:
            return None
        if stretches:
            bound = sum(c * min_max(total, grain, lo, hi, fs)[1] for c, fs in stretches)
        else:
            bound = priced(lo)
        return bound, lo, hi

    def done(bound):
        reach = best[0] if stop is None else min(best[0], stop)
        return bound >= reach - 1e-12 * abs(reach)

    first, order = box(low, high), 0
    heap = [(first[0], order, first[1], first[2])] if first else []
    while heap:
        bound, _, lo, hi = heapq.heappop(heap)
        if done(bound):
            break
        if budget is not None and spent >= budget:
            return list(best[1]), best[0], False
        spent += 1
        if sum(lo) == total:
            best = min(best, (priced(lo), tuple(lo)))
            if stop is not None and best[0] < stop:
                break
            continue
        i = max(range(n), key=lambda i: (hi[i] - lo[i], -i))
        cut = lo[i] + grain * ((hi[i] - lo[i]) // grain // 2)
        for part in ((lo, hi[:i] + [cut] + hi[i + 1:]), (lo[:i] + [cut + grain] + lo[i + 1:], hi)):
            made = box(*part)
            if made and not done(made[0]):
                order += 1
                heapq.heappush(heap, (made[0], order, made[1], made[2]))
    return list(best[1]), best[0], True


def _remembered(value):
    """`value` remembering each parts' value, so a decision's searches (its stand test, its step, a half step) price a
    point once."""
    seen = {}

    def priced(u):
        key = tuple(u)
        if key not in seen:
            seen[key] = value(u)
        return seen[key]
    return priced


def _descend(total, grain, low, high, value, knots):
    """`least`'s start: (two ranks: `low`, the rest to the first rank with room; more: `_fill`), each two ranks' split
    moved to its least (`_split`, exact for the two with the others held: `knots` each rank's) while that lowers the
    value, or keeps it and gives the lower rank less (as `_split` breaks ties), until every pair stands at its least
    (block coordinate descent [Tseng 2001]; the value and then the parts in rank order fall at every move, so it ends;
    two ranks are one pair, so one split, exact).  A sweep costs n (n - 1) / 2 splits of O(knots log(total / grain))
    evaluations each.  (units, the value.)"""
    n = len(low)
    if n > 2:
        u = _fill(total, grain, low, high, value)
    else:
        u = list(low)
        for i in range(n):
            u[i] += max(0, min(total - sum(u), high[i] - u[i]))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    v, k, settled = value(u), 0, 0
    while settled < len(pairs):
        i, j = pairs[k % len(pairs)]
        w, x = _split(u[i] + u[j], grain, [low[i], low[j]], [high[i], high[j]],
                      lambda p, i=i, j=j, u=list(u): value(u[:i] + [p[0]] + u[i + 1:j] + [p[1]] + u[j + 1:]),
                      [knots[i], knots[j]])
        k, settled = k + 1, settled + 1
        if (x, w[0]) < (v, u[i]):
            u[i], u[j], v, settled = w[0], w[1], x, 1
    return u, v


def _fill(total, grain, low, high, value):
    """A start for `least` of more than two ranks, the greedy fill [Ibaraki & Katoh 1988] coarse to fine: from `low`,
    each chunk to the rank whose value with it is least (ties to the lower rank) while a chunk fits within `high` and
    `total`; the first chunk the grain times the largest power of two that places four to eight a rank, then each
    half the last, each rank first handing back two chunks of the coarser size (down to its `low`), down to the grain.
    It fills from below, so ranks tied at the slowest are filled past together, which moves between two ranks alone
    cannot do (a rank joining with none: the pairwise descent from parts [100, 100, 0] of three equal ranks stands).
    The first size places at most 8 n chunks and each later one at most 4 n + 2: O(n^2 log(total / grain))
    evaluations where a grain at a time took n total / grain."""
    n, u, chunk = len(low), list(low), grain
    while chunk * 8 * n <= total - sum(low):
        chunk *= 2
    while True:
        while sum(u) + chunk <= total:
            fits = [i for i in range(n) if u[i] + chunk <= high[i]]
            if not fits:
                break
            i = min(fits, key=lambda i: value(u[:i] + [u[i] + chunk] + u[i + 1:]))
            u[i] += chunk
        if chunk == grain:
            return u
        u, chunk = [max(b, p - 2 * chunk) for p, b in zip(u, low)], chunk // 2


def _split(total, grain, low, high, value, knots):
    """The split [a, total - a] of two ranks of least `value`, the least a among equal values, a on the grain from
    lo = max(low_0, total - high_1) up to min(high_0, total - low_1) (else (low, value(low))).  `knots` are each rank's
    units where its models bend (the `time` models' own `knots`: rank 0's at a, rank 1's at total - a).  Between two
    consecutive knots every model is affine in a (below its first point the max of 0 and an affine piece, convex), and
    `value` composes the models by max, + and nonnegative counts alone (_path's longest path is max-plus [Baccelli,
    Cohen, Olsder & Quadrat 1992]), so it is convex there [Boyd & Vandenberghe 2004, 3.2.3]; a rank's units 0 (it
    publishes nothing) is a knot of its own.  The grain points at and beside each knot are evaluated, and each run of
    grain points between two by a bisection on the sign of its forward difference, which on a convex sequence ends at
    its least minimiser: O(knots log(total / grain)) evaluations where trying every split took total / grain, with the
    same minimum."""
    lo = max(low[0], total - high[1])
    hi = lo + grain * ((min(high[0], total - low[1]) - lo) // grain)
    if lo > hi:
        return list(low), value(low)
    seen = {}

    def at(a):
        if a not in seen:
            seen[a] = value([a, total - a])
        return seen[a]

    marks = {lo, hi}
    for a in [0, *knots[0], total, *(total - u for u in knots[1])]:
        if lo < a < hi:
            under = lo + grain * ((a - lo) // grain)
            marks |= {under, min(hi, under + grain)}
    marks = sorted(marks)
    best = min((at(a), a) for a in marks)
    for p, q in zip(marks, marks[1:]):
        i, j = p + grain, q - grain
        if i > j:
            continue
        while i < j:
            m = i + grain * ((j - i) // grain // 2)
            if at(m + grain) < at(m):
                i = m + grain
            else:
                j = m
        best = min(best, (at(i), i))
    return [best[1], total - best[1]], best[0]


def _path(rows):
    """A call's time from its stretches `rows` [(c, order, latency, [a rank's seconds, None where it publishes none])]:
    each program's stretches (order: (program, place)) in place order as a longest path [the dependency graph's
    critical path; Baccelli, Cohen, Olsder & Quadrat 1992, max-plus], a stretch starting on a rank once its own
    previous one has ended and every other rank that published there has ended and crossed (the previous stretch's
    latency, seconds, or, where it is a function, its value at whether each rank publishes there, a tuple of booleans: a
    crossing's time over the ranks it then joins, so a rank that holds none of the parts publishing there leaves it),
    the program's count times its end and its last crossing; a stretch without an order its count times its slowest
    rank.  With every latency 0 it is the sum of each stretch's slowest rank."""
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
            # the latest crossed end of the ranks that sent, and the next latest (a rank waits on the others' alone)
            first, at, second = -math.inf, -1, -math.inf
            for j in range(n):
                if sent[j]:
                    t = end[j] + crossed
                    if t > first:
                        first, at, second = t, j, first
                    elif t > second:
                        second = t
            end = [max(end[i], second if i == at else first) + (times[i] or 0.0) for i in range(n)]
            sent = [t is not None for t in times]
            crossed = latency(tuple(sent)) if callable(latency) else latency or 0.0
        total += max(c for _, c, _, _ in seq) * (max(end) + crossed)
    return total


# The rule a vertex is priced within (Balancer._vertex), part of a weighed decision's key, so a decision weighed under
# another rule is weighed again at its parts.
VERTEX_RULE = 'joined+unattributed'


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
    one's work there), less the work it names no decision for (`fixed`, the same units, and `joined`, work a rank does
    there only while it holds some decision's units there: its share's own norms, finishes and crossing dispatches,
    which a rank holding none there does not run).  A decision's parts are its groups' (`groups`, the ranks holding each part,
    default one rank a part): a part some coordinate's ranks hold together (a pipeline stage's layers), its time a
    stretch their slowest's; a rank in none of its groups is constant to it, as is a rank holding none of it, and a
    part none of whose ranks run a stretch is no part of that stretch.  A stretch a decision has seen and no rank now
    runs (its parts' ranks hold none: an emptied stage) stays in its models at the count, remainders and floor it
    last had, so a move back is priced on what it measured.

    The models are DFPA's [Lastovetsky & Reddy 2010, partial estimation of functional performance models]: each
    (decision, stretch, rank)'s measured points (units, seconds an invocation), up to `window` at a part, their median
    [LB-BSP 2020, §3.2.1]; the time interpolated between them, nondecreasing (`time`), within a doubling trust region
    past the largest part the holder was measured at in the stretches the call ran, else in any the decision has seen
    (a part that moves the work into other stretches, a whole layer's onto one member, does not widen the region of the
    stretches it left), else past its part (a holder never measured: a grain past it), never less than a grain past it
    (a region that has halved to the measured parts cannot move them, and a decision whose models promise a gain then
    stood without a step) [Conn, Gould & Toint 2000]; a rank with none at a stretch read at the measured ranks' median
    time a unit.  Every stretch a decision observes is remembered (its count, each holder's remainder and the outside
    ranks' time), the call a move is rejected at too, so a move measured worse is priced by what it measured once its
    stretches are no longer run.  A model's median at a part moving by more than the resolution drops its other points
    (its performance changed [Clarke, Lastovetsky & Rychkov 2011]).  A move the next call measures worse by more than
    the resolution is rejected: the parts go back, the trust region halves and the measurement stays in the models
    (a trust-region method's ratio test [Conn, Gould & Toint 2000, §6.1]); it doubles again, to twice, after a move
    that held.  The test reads the whole call (`measured`, the call's own time where the caller gives it, `measure_name`
    its kind: a new kind starts the decisions' histories again, as a new measure does; else every
    stretch's slowest rank on the models' path, which leaves out what no stretch times; a decision none of whose stretches
    the call ran, a move that left it no crossing, is tested on `measured` alone): a move can carry work across stretches
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
    calls, takes at least twice the spread of the measured sums of repeated calls at the same parts of every decision
    (what changes between calls, a peer program's load on a GPU, which no one call shows; a call after another
    decision's move is not a repeat), so a peer's load does not reject a move.  Every rank given the same evidence gets the same
    parts (R4).

      Balancer({'tp': {'parts': [8, 8], 'grain': 1, 'low': [1, 1], 'high': [15, 15]}})
      .observe({'step:3': {'scales': {'tp': [1.0, 1.0]}, 'fixed': [0.2, 0.1], 'times': [[s, ...], [s, ...]]}, ...})
      -> {'tp': [5, 11]}

    What a decision keeps is bounded, so nothing a call does grows with the calls before it (metal-microbench
    docs/acceptance-20261014.md A37): with `memory`, each model's points at its `memory` latest parts and the
    measured calls of the `memory` latest placements (every decision's parts: the ratio test's spread, one for the
    program) and each decision's measured parts (its vertices' history), the older dropped; with `budget`,
    each search (least) makes at most that many boxes, past which it takes the best parts it found.

    Of the moves reaching the models' least, the one of fewest moved parts (_fewest: a neighbouring pair's split where
    it reaches it).  A rank whose whole work at a stretch is a decision's (no other decision it holds weighs there and
    no fixed work, its joined work leaving with its units) publishes nothing there at none of it, so the crossing after it, where its latency is a function of
    the ranks publishing (_path), is priced without it: a rank whose crossings cost more than its work saves moves to
    none on the calls' evidence.

    `state()` and `Balancer.of(state)` carry it between processes."""

    def __init__(self, decisions, window=3, memory=None, budget=None):
        self.window, self.memory, self.decisions, self.points = int(window), memory and int(memory), {}, {}
        self.budget, self.measure, self.calls = budget and int(budget), 'sum', {}
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

    def observe(self, stretches, measured=None, measure_name='measured'):
        import statistics
        seen = {}
        for s, obs in stretches.items():
            scales = {d: w for d, w in (obs.get('scales') or {}).items() if d in self.decisions}
            ranks = {i: _robust([float(t) for t in ts]) for i, ts in enumerate(obs['times']) if ts}
            if scales and ranks:
                order = tuple(obs['order']) if obs.get('order') is not None else None
                latency = obs.get('latency')
                standing, joined = list(obs.get('fixed') or []), list(obs.get('joined') or [])
                fixed = [(standing[i] if i < len(standing) else 0.0) + (joined[i] if i < len(joined) else 0.0)
                         for i in range(max(len(standing), len(joined)))]
                seen[s] = (max(len(ts) for ts in obs['times'] if ts), ranks, scales, fixed, order,
                           latency if callable(latency) else float(latency or 0.0), len(obs['times']), standing)
        priced = any(v[4] is not None for v in seen.values())
        measure = ('path' if priced else 'sum') + (' joined' if any(any(v[3][i] != (v[7][i] if i < len(v[7]) else 0.0) for i in range(len(v[3])))
                                                                  for v in seen.values()) else '') + (f' {measure_name}' if measured is not None else '')
        if not seen and measured is not None:
            measure = self.measure

        def rows(times, keep=None):
            return [(c, order, latency, times(s, ranks, n)) for s, (c, ranks, _, _, order, latency, n, _) in seen.items()
                    if keep is None or s in keep]

        medians = lambda s, ranks, n: [ranks[i][0] if i in ranks else None for i in range(n)]
        call = _path(rows(medians))
        whole = max(RESOLUTION, 2 * math.sqrt(sum((c * max(r.values())[1]) ** 2 for c, r, *_ in seen.values())) / call if call else 0.0)
        tested = float(measured) if measured is not None else call
        placed = ';'.join(f'{name}:' + ','.join(map(str, d['parts'])) for name, d in sorted(self.decisions.items()))
        moved = self.measure != measure
        if moved:
            self.calls.clear()
        self.measure = measure
        self.calls[placed] = (self.calls.pop(placed, []) + [tested])[-5:]
        while self.memory and len(self.calls) > self.memory:
            self.calls.pop(next(iter(self.calls)))
        spread = [abs(f / statistics.median(fs) - 1) for fs in self.calls.values() if len(fs) > 1 and statistics.median(fs) > 0
                  for f in fs]
        between = max(whole, 2 * 1.4826 * statistics.median(spread) if len(spread) > 2 else 0.0)
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
                if measured is not None:
                    history, key = d.setdefault('history', {}), ','.join(map(str, d['parts']))
                    if moved:
                        history.clear()
                        d['last'] = None
                    history[key] = (history.pop(key, []) + [tested])[-5:]
                    while self.memory and len(history) > self.memory:
                        history.pop(next(iter(history)))
                    last, d['last'] = d.get('last'), [list(d['parts']), tested]
                    if last and last[0] != d['parts'] and tested > last[1] * (1 + between):
                        proposals.append((math.inf, name, last))
                        runs[name] = set()
                continue
            path = {s for s, v in seen.items() if priced and v[4] is not None and any(seen[t][4] and seen[t][4][0] == v[4][0] for t in mine)}
            counted = runs[name] = set(mine) | path
            summed = sum(c * max(med for med, _ in r.values()) for c, r, *_ in mine.values())
            slowest = [(c, max(r.values())) for c, r, *_ in mine.values()]
            within = max(RESOLUTION, 2 * math.sqrt(sum((c * se) ** 2 for c, (_, se) in slowest)) / summed if summed else 0.0)
            base = _path(rows(medians, counted)) if path else summed
            history, changed = d.setdefault('history', {}), moved
            if moved:
                history.clear()
                d['last'] = None
            key = ','.join(map(str, d['parts']))
            history[key] = (history.pop(key, []) + [tested])[-5:]
            while self.memory and len(history) > self.memory:
                history.pop(next(iter(history)))
            epsilon = within
            last, d['last'] = d.get('last'), [list(d['parts']), tested]
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
                    kept = [(u, t) for u, t in own if u != units] + [(units, t) for t in now]
                    recent = list(dict.fromkeys(u for u, _ in reversed(kept)))[:self.memory] if self.memory else None
                    self.points[(name, s, g)] = [(u, t) for u, t in kept if recent is None or u in recent]
            remembered = d.setdefault('seen', {})
            for s, (c, ranks, scales, fixed, view, other) in mine.items():
                remembered[s] = [c, [[g, view[g][0][0] - self._model(name, s, g)(d['parts'][g])] for g in view if d['parts'][g]], other]
            if last and last[0] != d['parts'] and tested > last[1] * (1 + between):
                proposals.append((math.inf, name, last))
                continue
            if last and last[0] != d['parts']:
                d['reach'] = min(2.0, d.get('reach', 2.0) * 2)
            if d['before'] is not None and d['before'] == d['parts'] and not changed and (d.get('weighed') == f'{key} {VERTEX_RULE}' or not self._vertices(d, history)):
                d['stands'] = True
                continue
            n = len(d['parts'])
            kept, ranked = [], {}
            for s, (c, rests, other) in remembered.items():
                if s in mine:
                    continue
                rests = dict((int(g), r) for g, r in rests)
                kept.append((c, [(lambda u, o=other: o) if d['parts'][g] or g not in rests else
                                 (lambda u, m=self._model(name, s, g), r=rests[g], o=other: max(o, m(u) + r) if u else o)
                                 for g in range(n)]))
            built, alone = list(kept), {}
            for s, (c, ranks, scales, fixed, view, other) in mine.items():
                standing = seen[s][7]
                alone[s] = {i for i in ranks if not max(0.0, float(standing[i] if i < len(standing) else 0.0))
                            and not any(max(0.0, float(w[i] or 0.0)) and self._holds(e, i) for e, w in scales.items() if e != name)}
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
                built.append((c, fs))
                ranked[s] = gs
            value = None
            if path:
                member = {i: g for g, members in enumerate(groups) for i in members}

                def value(u, ranked=ranked, member=member, kept=kept, counted=counted, alone=alone):
                    def at(s, ranks, n):
                        if s not in ranked:
                            return medians(s, ranks, n)
                        return [(ranked[s][member[i]](u[member[i]]) if (i in ranks or u[member[i]])
                                 and (u[member[i]] or i not in alone.get(s, ())) else None)
                                if member.get(i) in ranked[s] else (ranks[i][0] if i in ranks and i not in member else None)
                                for i in range(n)]
                    return _path(rows(at, counted)) + sum(c * max(f(u[g]) for g, f in enumerate(fs)) for c, fs in kept)
                value = _remembered(value)
            total = sum(d['parts'])
            knots = [sorted({int(k) for (e, s, h) in self.points if e == name and h == g for k in self._model(name, s, g).knots})
                     for g in range(n)]
            target = base - epsilon * summed
            if least(total, d['grain'], d['low'], d['high'], [] if value else built, value, knots, stop=target,
                     budget=self.budget)[1] >= target:
                joined = sum(c * sum(med * max(0.0, fixed[i] - (seen[s][7][i] if i < len(seen[s][7]) else 0.0)) / total_i
                                     for i, (med, _) in ranks.items() if i < len(fixed)
                                     for total_i in [sum(max(0.0, float(w[i] or 0.0)) for w in scales.values()) + max(0.0, fixed[i])] if total_i)
                             for s, (c, ranks, scales, fixed, _, _) in mine.items())
                vertex = self._vertex(d, history, value or (lambda u, built=built: sum(c * max(f(u[g]) for g, f in enumerate(fs))
                                                                                     for c, fs in built)), joined + max(0.0, tested - call))
                if vertex:
                    proposals.append(((base - vertex[1]) / summed if summed else 0.0, name, vertex[0]))
                    continue
                d['before'], d['stands'], d['weighed'] = list(d['parts']), True, f'{key} {VERTEX_RULE}'
                continue
            grow = d.get('reach', 2.0)
            failed = d.get('failed') or [None] * n
            here, anywhere = [None] * n, [None] * n
            for (e, s, g), own in self.points.items():
                if e == name and own:
                    anywhere[g] = max([anywhere[g] or 0] + [u for u, _ in own])
                    here[g] = max([here[g] or 0] + [u for u, _ in own]) if s in mine else here[g]
            measured_ = [h if h is not None else a if a is not None else d['parts'][g] for g, (h, a) in enumerate(zip(here, anywhere))]
            reach = [min(d['high'][g], max(d['parts'][g], d['grain'] * math.ceil(grow * u / d['grain']), u + d['grain']))
                     for g, u in enumerate(measured_)]
            reach = [r if f is None else max(min(r, f - d['grain']), d['parts'][g]) for g, (r, f) in enumerate(zip(reach, failed))]
            nxt, promise, _ = least(total, d['grain'], d['low'], reach, [] if value else built, value, knots, budget=self.budget)
            if sum(nxt) < total:
                reach = [d['high'][g] if f is None else max(f - d['grain'], d['parts'][g]) for g, f in enumerate(failed)]
                nxt, promise, _ = least(total, d['grain'], d['low'], reach, [] if value else built, value, knots, budget=self.budget)
            nxt, promise = self._fewest(d, nxt, promise, value or (
                lambda u, built=built: sum(c * max(f(u[g]) for g, f in enumerate(fs)) for c, fs in built)), reach, knots)
            if d['before'] is not None and any((c - p) * (p - b) < 0 for c, p, b in zip(nxt, d['parts'], d['before'])):
                half = [(p + c) / 2 for c, p in zip(nxt, d['parts'])]
                nxt, promise, _ = least(total, d['grain'], [math.floor(h / d['grain']) * d['grain'] for h in half],
                                        [math.ceil(h / d['grain']) * d['grain'] for h in half], [] if value else built, value, knots,
                                        budget=self.budget)
            proposals.append(((base - promise) / summed if summed else 0.0, name, nxt))
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

    def _vertex(self, d, history, value, joined):
        """A decision about to stand: the vertex where one group leaves it (its units to the others in proportion to
        theirs) that no call at these parts' history measured, where the models price it within `joined` (the measured
        time they attribute to the stretches' joined work, which a leaving group takes with it and a crossing no longer
        joined costs no one, and the call's measured time their path leaves out, a session's start where a vertex leaves
        the call no crossing: what interior points cannot price) of the parts', else None.  Such a vertex is measured
        once, and the ratio test returns a worse one; a stretch's work changes in kind where a group leaves it (its
        crossing goes), so the vertex is a design point of its own [Box, Hunter & Hunter 2005, the factorial's corner
        points].  (parts, value.)"""
        best = None
        for moved in self._vertices(d, history):
            v = value(moved)
            if best is None or v < best[1]:
                best = (moved, v)
        return best if best and best[1] <= value(d['parts']) + joined else None

    @staticmethod
    def _vertices(d, history):
        """The decision's vertices no call at its history measured: each group's leaving it, its units to the others in
        proportion to theirs on the grain, within the bounds."""
        parts, grain, n, out = d['parts'], d['grain'], len(d['parts']), []
        for g in range(n):
            others = sum(p for h, p in enumerate(parts) if h != g)
            if not parts[g] or d['low'][g] > 0 or not others:
                continue
            moved = [0 if h == g else (p + parts[g] * p // others) // grain * grain for h, p in enumerate(parts)]
            moved[max(range(n), key=lambda h: moved[h])] += sum(parts) - sum(moved)
            if not any(m > hi or m < lo for m, lo, hi in zip(moved, d['low'], d['high'])) and ','.join(map(str, moved)) not in history:
                out.append(moved)
        return out

    def _fewest(self, d, nxt, promise, value, reach, knots):
        """Of the decision's moves whose value is the models' least within `reach` (nxt, its value `promise`), the one
        that moves fewest parts: nxt, or a neighbouring pair's least split with the others held (_split, exact for the
        two) where it reaches the same value, so where the least is not one point (a holder whose time is flat in its
        part, a share whose slowest holder alone sets it) the move leaves the other parts where they are (a part a
        member holds as a range of an axis, its run's start the parts before it: a move between neighbours moves no
        other member's range, and a member whose range moves reloads it; metal-microbench docs/elastic.md The engine's
        calls).  A move of fewer parts at a worse value is not taken: it reaches the least in more calls, each a
        reload of its own (metal-microbench output_data/planner-20261015/pass2 allocate2-after.json moves_neighbours:
        8 reloads in 3 moves after a holder slowed 30 %, where the least's one move reloaded 3).  (parts, value.)"""
        parts, grain, moves = d['parts'], d['grain'], [(nxt, promise)]
        for g in range(len(parts) - 1):
            pair, _ = _split(parts[g] + parts[g + 1], grain, d['low'][g:g + 2], [max(parts[g], reach[g]), max(parts[g + 1], reach[g + 1])],
                             lambda p, g=g: value(parts[:g] + list(p) + parts[g + 2:]), knots[g:g + 2])
            moved = parts[:g] + pair + parts[g + 2:]
            moves.append((moved, value(moved)))
        reach_ = promise + 1e-12 * abs(promise)
        eligible = [(sum(a != b for a, b in zip(u, parts)), v, u) for u, v in moves if v <= reach_ and u != parts]
        if not eligible:
            return nxt, promise
        _, v, u = min(eligible, key=lambda m: (m[0], m[1]))
        return u, v

    def _model(self, name, s, i):
        return time(_median_points(self.points.get((name, s, i), [])))

    def state(self):
        return {'window': self.window, 'memory': self.memory, 'budget': self.budget, 'decisions': self.decisions,
                'points': [[d, s, i, own] for (d, s, i), own in self.points.items()], 'measure': self.measure,
                'calls': self.calls}

    @classmethod
    def of(cls, state):
        b = cls({}, state.get('window', 3), state.get('memory'), state.get('budget'))
        b.decisions = {name: {'grain': 1, 'low': [0] * len(d['parts']), 'high': [sum(d['parts'])] * len(d['parts']), 'before': None,
                              'stands': False, **{k: v for k, v in d.items() if k != 'measure'},
                              'groups': d.get('groups') or [[i] for i in range(len(d['parts']))]}
                       for name, d in state['decisions'].items()}
        b.points = {(d, s, int(i)): [tuple(p) for p in own] for d, s, i, own in state.get('points', [])}
        b.measure = state.get('measure') or next((d['measure'] for d in state['decisions'].values() if d.get('measure')), 'sum')
        b.calls = {k: list(v) for k, v in (state.get('calls') or {}).items()}
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
            if abs(now - self.stood) <= max(RESOLUTION, 2 * self._error(e['calls'])) * self.stood:
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
        resolution = max(RESOLUTION, 2 * self._error(measured[proposal]['calls']))
        if proposal_time - best_time > resolution * best_time:
            chosen = best
        self.stands, self.stood = True, self._median(measured[chosen]['calls'])
        return chosen
