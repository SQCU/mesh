"""AOT allocation math, measured-point models and report-only Pareto summaries.

The online policy lives exclusively in controller.WholeFunctionController.
min_max/equal_finish quantize a declared physical model; Points retains measured
resource evidence; project_parts realizes fractional operands on a new extent.
These helpers do not accept/reject whole-call executions or choose an online
standing configuration. Historical Balancer/Coupled interfaces were removed.
"""
import heapq
import math

RESOLUTION = 0.01  # Relative measurement-noise floor for retained resource points; not an acceptance gate.


def min_max(total, grain, low, high, times):
    """Shares of `total` units at `grain` within [low_i, high_i] (multiples of grain, a high may be math.inf) minimising
    the largest of times[i](c_i), each nondecreasing: each grain to the rank finishing earliest with it, ties to the
    lower rank [Ibaraki & Katoh 1988; Beaumont et al. 2001, Alg. 3.1].  That greedy takes the (total - sum low) / grain
    least of the ranks' grain times (rank i's k-th grain's times[i](low_i + k grain), a sorted list a rank), so it
    is found as that order statistic of the sorted lists [Frederickson & Johnson 1982]: a pivot the weighted median of
    the lists' middle times, every list cut at it by bisection, until the pivot is the statistic; O(n log^2(total /
    grain)) evaluations where a grain at a time took total / grain.  A time that dips is the caller's to pass as the
    largest it has reached (its least nondecreasing majorant, which the grain-at-a-time greedy took; metal-microbench
    tools/mesh/balance_sim.py envelope): every time here is read as sorted.  (c, T), T the largest time, each rank's at
    its low counted; grains no rank can hold are left unplaced."""
    n = len(low)
    want = max(0, (total - sum(low)) // grain)
    top = [max(0, want if h == math.inf else min(want, (h - lo) // grain)) for lo, h in zip(low, high)]
    seen = {}

    def at(i, k):
        if (i, k) not in seen:
            seen[i, k] = times[i](low[i] + k * grain)
        return seen[i, k]

    def cut(i, v, a, b, strict):
        while a < b:
            m = (a + b + 1) // 2
            if at(i, m) < v or not strict and at(i, m) == v:
                a = m
            else:
                b = m - 1
        return a

    floor = max((times[i](low[i]) for i in range(n)), default=0.0)
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


def equal_finish(total, grain, low, high, a, b):
    """Affine equal-finish shares, restored from fa36bea6^; min_max realizes tiles.

    a includes measured fixed cost and dependency delivery tail; b is measured
    work cost per unit. Neither is an assumed price for DMA/compute interference.
    """
    if not (len(a) == len(b) == len(low) == len(high)) or not a:
        raise ValueError('equal-finish dimensions')
    if any(not math.isfinite(v) or v <= 0 for v in b):
        raise ValueError('equal-finish slopes must be positive')
    if not sum(low) <= total <= sum(high):
        raise ValueError('infeasible equal-finish bounds')
    held = {}
    while True:
        free = [i for i in range(len(a)) if i not in held]
        T = ((total - sum(held.values()) + sum(a[i] / b[i] for i in free)) /
             sum(1 / b[i] for i in free) if free else max(a[i] + b[i] * held[i] for i in held))
        s = [held[i] if i in held else (T - a[i]) / b[i] for i in range(len(a))]
        below = [i for i in free if s[i] < low[i]]
        above = [i for i in free if s[i] > high[i]]
        if not below and not above:
            break
        if sum(low[i] - s[i] for i in below) >= sum(s[i] - high[i] for i in above):
            held.update((i, float(low[i])) for i in below)
        else:
            held.update((i, float(high[i])) for i in above)
    c, _ = min_max(total, grain, low, high, [lambda u, i=i: a[i] + b[i] * u for i in range(len(a))])
    return c, s, T


def time(points):
    """A rank's time for d units from its measured points [(units, seconds)], made nondecreasing in units (pool
    adjacent violators [Ayer et al. 1955]: more units never take less time), the time interpolated linearly between
    them and extended along the line through the two nearest past either end (between none and the first's time below
    it); with one point, in proportion below it and its own time above (the least a nondecreasing time can be there:
    optimism where nothing is known [Auer et al. 2002], which the trust region bounds); with none, 0 (a Points holds
    none only where every time it was given was 0: its decision does no work there).  Its `knots` are the units where
    it bends (the nondecreasing points'; at 0 it steps, and below the first point it is the max of 0 and an affine
    piece, convex), which `_split` reads off it."""
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
            return 0.0
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


class Points:
    """A holder's time on a stretch at each part it ran: DFPA's partial functional performance model [Lastovetsky &
    Reddy 2010] whose points are latent, each part's time a Kalman filter of a random walk observed in noise [Kalman
    1960], so the model has a posterior and not only a value.  A call at u units observes its time t with its own
    standard error `se` (the call's invocations') and the variance between calls `r` (the innovations' excess over
    the filters' and the calls' own, estimated as they come [Mehra 1970]); every point's variance grows by 1 / `forget`
    a call (the drift the forgetting factor states [Haykin 2014]), so a holder whose performance changed moves its
    points by its calls' own evidence, every call kept.  The `keep` latest parts are held.  The model is `time` over the
    points' means (more units never take less time, interpolated between them)."""

    def __init__(self, forget, keep=8):
        self.forget, self.keep, self.r, self.seen, self.at = float(forget), int(keep), 0.0, 0, {}
        self._made = None

    def update(self, u, t, se=0.0):
        """One call: `t` seconds at `u` units, its standard error `se`."""
        u, t = int(u), float(t)
        if u <= 0 or t <= 0:
            return
        self.seen, self._made = self.seen + 1, None
        for point in self.at.values():
            point[1] /= self.forget
        own = max(se * se, (RESOLUTION * t) ** 2)
        if u in self.at:
            m, v = self.at.pop(u)
            e, spread = t - m, v + own + self.r
            self.r += (1 - self.forget) * (max(0.0, e * e - v - own) - self.r)
            k = v / spread
            self.at[u] = [m + k * e, (1 - k) * v]
        else:
            self.at[u] = [t, own + self.r]
        while len(self.at) > self.keep:
            self.at.pop(next(iter(self.at)))

    def _model(self):
        if self._made is None:
            self._made = time([(u, m) for u, (m, v) in self.at.items()])
        return self._made

    def __call__(self, u):
        return self._model()(u)

    @property
    def knots(self):
        return self._model().knots

    def state(self):
        return {'forget': self.forget, 'keep': self.keep, 'r': self.r, 'seen': self.seen,
                'at': [[u, m, v] for u, (m, v) in self.at.items()]}

    @classmethod
    def of(cls, state):
        model = cls(state['forget'], state['keep'])
        model.r, model.seen, model.at = state['r'], state['seen'], {int(u): [m, v] for u, m, v in state['at']}
        return model


def _robust(samples):
    """(median, its standard error) of seconds an invocation: 1.2533 x 1.4826 MAD / sqrt(n), which a tail of slow
    invocations does not inflate."""
    import statistics
    med = statistics.median(samples)
    mad = statistics.median(abs(t - med) for t in samples)
    return med, 1.2533 * 1.4826 * mad / math.sqrt(len(samples))


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
    thousand where the descent evaluated 6 thousand): past it the search ends at the best point found, which is then not
    known to be the least.  (units, the value, whether it is the least: the search ended within its budget.)"""
    n = len(low)
    low, high = [-(-v // grain) * grain for v in low], [v // grain * grain for v in high]
    if len(stretches) == 1 and value is None:
        c, fs = stretches[0]
        units, top = min_max(total, grain, low, high, fs)
        return units, c * top, True
    value = value or (lambda u: sum(c * max(fs[i](u[i]) for i in range(n)) for c, fs in stretches))
    seen = {}

    def valued(u):
        key = tuple(u)
        if key not in seen:
            seen[key] = value(list(u))
        return seen[key]

    best = _descend(total, grain, low, high, valued, knots)
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
            bound = valued(lo)
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
            best = min(best, (valued(lo), tuple(lo)))
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
    """`value` remembering each parts' value, so a decision's searches (its stand test, its step, a half step) evaluate a
    point once."""
    seen = {}

    def valued(u):
        key = tuple(u)
        if key not in seen:
            seen[key] = value(u)
        return seen[key]
    return valued


def _descend(total, grain, low, high, value, knots):
    """`least`'s start: (two ranks: `low`, the rest to the first rank with room; more: `_fill`), each two ranks' split
    moved to its least (`_split`, exact for the two with the others held: `knots` each rank's) while that lowers the
    value, or keeps it and gives the lower rank less (as `_split` breaks ties), until every pair stands at its least
    (block coordinate descent [Tseng 2001]; the value and then the parts in rank order fall at every move, so it ends;
    two ranks are one pair, so one split, exact).  A sweep is n (n - 1) / 2 splits of O(knots log(total / grain))
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


def pareto(rows, coordinate=('ctx_in', 'ctx_out')):
    """The two throughputs' frontier (metal-microbench docs/principles.md §3, "Two throughputs, a frontier, and the
    placer on it"), from the calls' own records alone (no solo run, no fit): each row one call's, its configuration
    (`program`, `backend`, batch `bs`), its workload coordinate (`coordinate`'s keys, by default ctx_in and ctx_out)
    and its two measured throughputs, `aggregate_tok_s` (its tokens over its wall seconds) and `per_sequence_tok_s`
    (that over its streams).  Per coordinate, the rows no other row of it beats in both throughputs (at least as high
    in each and higher in one), highest aggregate first; and of two backends of one program, whether one's aggregate is
    higher at every (bs, coordinate) both ran.  (frontier {"ctx_in/ctx_out": [row...]}, dominance [...]).
    metal-microbench 4efbbff^ tools/mesh/grid.py pareto; its frontier 16384/1024 at bs 4: 564.6 aggregate, 141.1 per
    sequence (output_data/confirm-20260924/summary.json)."""
    key = lambda r: tuple(r[k] for k in coordinate)
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    frontier = {}
    for at in sorted(groups):
        group = groups[at]
        keep = [r for r in group if not any(
            o['aggregate_tok_s'] >= r['aggregate_tok_s'] and o['per_sequence_tok_s'] >= r['per_sequence_tok_s']
            and (o['aggregate_tok_s'] > r['aggregate_tok_s'] or o['per_sequence_tok_s'] > r['per_sequence_tok_s'])
            for o in group)]
        frontier['/'.join(str(v) for v in at)] = sorted(
            [{'program': r['program'], 'backend': r['backend'], 'bs': r['bs'], 'aggregate_tok_s': r['aggregate_tok_s'],
              'per_sequence_tok_s': r['per_sequence_tok_s']} for r in keep], key=lambda r: -r['aggregate_tok_s'])
    table = {(r['program'], r['backend'], r['bs'], *key(r)): r['aggregate_tok_s'] for r in rows}
    labels = sorted({(r['program'], r['backend']) for r in rows})
    dominance = []
    for (pa, a), (pb, b) in ((x, y) for x in labels for y in labels if x != y):
        if pa != pb:
            continue
        shared = [k[2:] for k in table if k[:2] == (pa, a) and (pa, b, *k[2:]) in table]
        if not shared:
            continue
        wins = sum(1 for k in shared if table[(pa, a, *k)] > table[(pa, b, *k)])
        dominance.append({'program': pa, 'backend': a, 'over': b, 'coordinates': len(shared), 'higher_at': wins,
                          'dominates': wins == len(shared)})
    return frontier, dominance


def project_parts(parts, total, grain):
    """Configuration-only proportional projection onto a changed extent."""
    from whole import project
    scale = total / sum(parts) if sum(parts) else 0
    values = [p * scale for p in parts] if scale else [total / len(parts)] * len(parts)
    return project(values, total, grain, [0] * len(parts), [total] * len(parts))
