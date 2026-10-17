"""Share allocation on measured times: the parts of a mesh function's next call from its calls' own evidence.

metal-microbench docs/principles.md §4, "Adaptive configuration without synchronization or cold start".  A share
group of W units at grain g is split over n ranks; after each call every rank passes the same measurements and gets the
same parts back (plain floats in a fixed order: identical inputs give identical parts on every rank), so the next
call's configuration is one more value in the dataflow: no root, no lock.

`min_max` divides units among ranks whose times are given nondecreasing functions of their units, each grain to the
rank finishing earliest with it [Ibaraki & Katoh 1988; Beaumont et al. 2001, Alg. 3.1].  `time` and `Points` are a
rank's measured points (units, seconds) as DFPA reads them [Lastovetsky & Reddy 2010, partial estimation of functional
performance models]: nondecreasing, interpolated between the points, each point a filter of its calls with a change
test.  `least` minimises a call's value over the lattice of parts (`min_max` for one stretch, a pairwise descent and a
branch and bound for more), and `_path` is a call's time from its stretches as a longest path.

`Balancer` is DFPA over a program's measured stretches (design/heterogeneity.md R3, R5, R10, R11).  `Coupled` divides
one decision among members that run at once and slow each other (a node's engines) by a search over allocations, each
measured before any is chosen.
"""
import heapq
import math

RESOLUTION = 0.01  # the least relative accuracy a decision resolves (R11): a gain below it is no gain [DFPA 2011]


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
    a call (the drift the forgetting factor states [Haykin 2014]); a two-sided cumulative sum of its standardised
    innovations past 5, each less a slack of 0.5 [Page 1954; Basseville & Nikiforov 1993], drops the other points (its
    holder's performance changed [Clarke, Lastovetsky & Rychkov 2011]): a shift that persists, not one call's outlier,
    which a single innovation's chi-square took for one (metal-microbench docs/placement.md#moves).  The
    `keep` latest parts are held.  The model is `time` over the points' means (more units never take less time,
    interpolated between them)."""

    def __init__(self, forget, keep=8):
        self.forget, self.keep, self.r, self.seen, self.at = float(forget), int(keep), 0.0, 0, {}
        self.cusum, self._made = [0.0, 0.0], None

    def update(self, u, t, se=0.0):
        """One call: `t` seconds at `u` units, its standard error `se`; whether it found the holder changed."""
        u, t = int(u), float(t)
        if u <= 0 or t <= 0:
            return False
        self.seen, self._made = self.seen + 1, None
        for point in self.at.values():
            point[1] /= self.forget
        own = max(se * se, (RESOLUTION * t) ** 2)
        changed = False
        if u in self.at:
            m, v = self.at.pop(u)
            e, spread = t - m, v + own + self.r
            z = e / math.sqrt(spread)
            self.cusum = [max(0.0, self.cusum[0] + z - 0.5), max(0.0, self.cusum[1] - z - 0.5)]
            if max(self.cusum) > 5.0:
                self.at, self.cusum, changed = {}, [0.0, 0.0], True
                self.at[u] = [t, own + self.r]
            else:
                self.r += (1 - self.forget) * (max(0.0, e * e - v - own) - self.r)
                k = v / spread
                self.at[u] = [m + k * e, (1 - k) * v]
        else:
            self.at[u] = [t, own + self.r]
        while len(self.at) > self.keep:
            self.at.pop(next(iter(self.at)))
        return changed

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
        return {'forget': self.forget, 'keep': self.keep, 'r': self.r, 'seen': self.seen, 'cusum': self.cusum,
                'at': [[u, m, v] for u, (m, v) in self.at.items()]}

    @classmethod
    def of(cls, state):
        model = cls(state['forget'], state['keep'])
        model.r, model.seen, model.at = state['r'], state['seen'], {int(u): [m, v] for u, m, v in state['at']}
        model.cusum = list(state.get('cusum') or [0.0, 0.0])
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
    decision's parts minimise its stretches' part of that time (its programs' whole paths where ordered, every
    decision on one path then moving alone in a call, its gain weighed against its own stretches' sum), a stretch's
    time on a rank divided among its decisions by the weights the program gives (any units: its own estimate of each
    one's work there), less the work it names no decision for (`fixed`, the same units, and `joined`, work a rank does
    there only while it holds some decision's units there: its share's own norms, finishes and crossing dispatches,
    which a rank holding none there does not run).  A decision's parts are its groups' (`groups`, the ranks holding each part,
    default one rank a part): a part some coordinate's ranks hold together (a pipeline stage's layers), its time a
    stretch their slowest's; a rank in none of its groups is constant to it, as is a rank holding none of it, and a
    part none of whose ranks run a stretch is no part of that stretch.  A stretch a decision has seen and no rank now
    runs (its parts' ranks hold none: an emptied stage) stays in its models at the count, remainders and floor it
    last had, so a move back is evaluated on what it measured.

    The models are DFPA's [Lastovetsky & Reddy 2010, partial estimation of functional performance models]: each
    (decision, stretch, rank)'s measured points (units, seconds an invocation), up to `window` at a part, their median
    [LB-BSP 2020, §3.2.1]; the time interpolated between them, nondecreasing (`time`), within a doubling trust region
    past the largest part the holder was measured at in the stretches the call ran, else in any the decision has seen
    (a part that moves the work into other stretches, a whole layer's onto one member, does not widen the region of the
    stretches it left), else past its part (a holder never measured: a grain past it), never less than a grain past it
    (a region that has halved to the measured parts cannot move them, and a decision whose models promise a gain then
    stood without a step) [Conn, Gould & Toint 2000]; a rank with none at a stretch read at the measured ranks' median
    time a unit.  Every stretch a decision observes is remembered (its count, each holder's remainder and the outside
    ranks' time), the call a move is rejected at too, so a move measured worse is evaluated by what it measured once its
    stretches are no longer run.  A model's median at a part moving by more than the resolution drops its other points
    (its performance changed [Clarke, Lastovetsky & Rychkov 2011]).  A move is judged by the models, every member's
    measured within the same call, never by comparing one call's measured time with another's: two calls run under
    whatever load their nodes' users put on them, so a comparison across calls charges a slower call to its placement,
    and a member made slower by its own load would see the split charged and its work handed back to it, the reverse of
    balancing (metal-microbench output_data/monotone-20261016: a one-row split returned to its first member alone after
    a call 0.611 s against an earlier 0.407 s; the operator, 2026-10-16: a loaded member must lead to more work on an
    unloaded one, never a zero share).  The parts go back only where the call failed (`reject`).  The trust region
    doubles, to twice, after a move.  `measured`, the call's own time where the caller gives it (`measure_name` its kind:
    a new kind starts the decisions' histories again, as a new measure does; else every stretch's slowest rank on the
    models' path), is kept as each placement's history, which names the vertices measured.  Of the decisions sharing a
    stretch one moves a call, the one whose models promise the most (block coordinate descent [Tseng 2001]): another's
    move would change the stretches its models read; one that waits is weighed again at the next call, not read as
    standing.  A move left pending (below) changes no stretch the call reads, so it holds no other decision back: a
    ragged decision moves while a prepared one waits for its preparation.  A decision's parts stand where the models' unbounded optimum promises less than its resolution under the
    measured sum [DFPA; Meta-Balancer 2012], and stay standing while none of its models changed; once a part moves back
    the way it came, every part of the decision moves half as far [LB-BSP 2020, §3.3.2].  The resolution is twice the
    measured sum's relative standard error (each stretch's slowest rank's median's, from its invocations, in
    quadrature), at least 1 %; a model changed where its window's median moved past that and past three of its own
    standard errors.  Every rank given the same evidence gets the same
    parts (R4).

    A move takes, in the call that takes it, its members' preparation of the placement it makes (metal-microbench
    docs/placement.md#moves: S), unless they prepared it before or the decision is `ragged` (its parts the calls' data,
    which no preparation reads: taken at once, whatever `prepare`): with `prepare`, a move observe takes waits as `pending`
    until every member has prepared the placement it makes, when `prepared` takes it, a move measured worse going back
    the same way (make before break [RFC 3209]: the placement it returns to need not be prepared still, its nested
    divisions moved to its parts, and a call never runs one that is not, which had taken a call 19.7 s of Neural Engine
    shares built inside it); without it a move, and its return, is taken at once.  Every move the models propose is
    taken, and a pending move is a commitment: a later proposal waits until it is taken or its preparation fails
    (`unprepared`), so a decision proposing at every call never outruns its own preparation (docs/placement.md#moves).

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
    the ranks publishing (_path), is evaluated without it: a rank whose crossings take longer than its work saves moves to
    none on the calls' evidence.

    `state()` and `Balancer.of(state)` carry it between processes."""

    def __init__(self, decisions, window=3, memory=None, budget=None):
        self.window, self.memory, self.decisions, self.lines = int(window), memory and int(memory), {}, {}
        self.budget, self.measure = budget and int(budget), 'sum'
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

    def observe(self, stretches, measured=None, measure_name='measured', prepare=False):
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
        ordered = any(v[4] is not None for v in seen.values())
        measure = ('path' if ordered else 'sum') + (' joined' if any(any(v[3][i] != (v[7][i] if i < len(v[7]) else 0.0) for i in range(len(v[3])))
                                                                  for v in seen.values()) else '') + (f' {measure_name}' if measured is not None else '')
        if not seen and measured is not None:
            measure = self.measure

        def rows(times, keep=None):
            return [(c, order, latency, times(s, ranks, n)) for s, (c, ranks, _, _, order, latency, n, _) in seen.items()
                    if keep is None or s in keep]

        medians = lambda s, ranks, n: [ranks[i][0] if i in ranks else None for i in range(n)]
        tested = float(measured) if measured is not None else _path(rows(medians))
        moved = self.measure != measure
        self.measure = measure
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
                    d['last'] = [list(d['parts']), tested]
                continue
            path = {s for s, v in seen.items() if ordered and v[4] is not None and any(seen[t][4] and seen[t][4][0] == v[4][0] for t in mine)}
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
                    line = self.lines.setdefault((name, s, g), Points(self.forget, self.memory or 8))
                    changed = line.update(units, share, se * share / med if med > 0 else 0.0) or changed
            remembered = d.setdefault('seen', {})
            for s, (c, ranks, scales, fixed, view, other) in mine.items():
                remembered[s] = [c, [[g, view[g][0][0] - self._model(name, s, g)(d['parts'][g])] for g in view if d['parts'][g]], other]
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
                    if (name, s, g) in self.lines:
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
            knots = [sorted({int(k) for (e, s, h), line in self.lines.items() if e == name and h == g for k in line.knots})
                     for g in range(n)]
            target = base - epsilon * summed
            if least(total, d['grain'], d['low'], d['high'], [] if value else built, value, knots, stop=target,
                     budget=self.budget)[1] >= target:
                joined = [sum(c * sum(med * max(0.0, fixed[i] - (seen[s][7][i] if i < len(seen[s][7]) else 0.0)) / total_i
                                      for i, (med, _) in ranks.items() if i < len(fixed) and i in groups[g]
                                      for total_i in [sum(max(0.0, float(w[i] or 0.0)) for w in scales.values()) + max(0.0, fixed[i])] if total_i)
                              for s, (c, ranks, scales, fixed, _, _) in mine.items()) for g in range(n)]
                vertex = self._vertex(d, history, value or (lambda u, built=built: sum(c * max(f(u[g]) for g, f in enumerate(fs))
                                                                                     for c, fs in built)), joined)
                if vertex:
                    proposals.append(((base - vertex[1]) / summed if summed else 0.0, name, vertex[0]))
                    continue
                d['before'], d['stands'], d['weighed'] = list(d['parts']), True, f'{key} {VERTEX_RULE}'
                continue
            grow = d.get('reach', 2.0)
            failed = d.get('failed') or [None] * n
            here, anywhere = [None] * n, [None] * n
            for (e, s, g), line in self.lines.items():
                if e == name and line.knots:
                    anywhere[g] = max([anywhere[g] or 0] + line.knots)
                    here[g] = max([here[g] or 0] + line.knots) if s in mine else here[g]
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
            # metal-microbench docs/placement.md#ragged-shares-a-move-is-data: a ragged decision's parts are each call's data,
            # so its move prepares nothing and is taken at once; a move left pending changes no stretch this call, so it
            # holds no other decision back
            prepare_ = prepare and not d.get('ragged')
            if reverting and prepare_:
                d['reach'] = max(1.0, d.get('reach', 2.0) / 2)
                d['stands'], d['last'], d['pending'], d['reverting'] = False, value, list(value[0]), True
            elif reverting:
                taken |= runs[name]
                d['reach'] = max(1.0, d.get('reach', 2.0) / 2)
                d['before'], d['parts'], d['stands'], d['last'], d['pending'] = None, list(value[0]), False, value, None
            elif not prepare_ or list(value) == d['parts'] and not d.get('pending'):
                taken |= runs[name]
                d['before'], d['parts'], d['stands'] = list(d['parts']), list(value), False
            elif not d.get('pending'):
                d['pending'], d['stands'] = list(value), False
        return self.parts

    @property
    def forget(self):
        """The models' forgetting factor: a memory of `window` calls at each of `memory` parts (8 where unset)."""
        return 1.0 - 1.0 / (self.window * (self.memory or 8))

    @property
    def pending(self):
        """Each decision's move taken and not yet prepared {name: parts} (observe with `prepare`: it waits as `pending`
        until every member has prepared the placement it makes, then `prepared` takes it)."""
        return {name: list(d['pending']) for name, d in self.decisions.items() if d.get('pending')}

    def unprepared(self, expected=None):
        """The pending moves `expected` names ({name: parts}; every pending one where None) given up: their preparation
        failed, so a later proposal may take their place."""
        for name, d in self.decisions.items():
            if d.get('pending') and (expected is None or expected.get(name) == d['pending']):
                d['pending'] = None
                d.pop('reverting', None)
        return self.parts

    def prepared(self, expected=None):
        """Every pending move taken (where `expected`, {name: parts}, names the pending moves that were prepared: only
        those still pending as prepared): the parts become the pending ones, as observe's move made them."""
        for name, d in self.decisions.items():
            if d.get('pending') and (expected is None or expected.get(name) == d['pending']):
                d['before'] = None if d.pop('reverting', False) else list(d['parts'])
                d['parts'], d['stands'], d['pending'] = list(d['pending']), False, None
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
        """A decision about to stand: the vertex where one group g leaves it (its units to the others in proportion to
        theirs) that no call at these parts' history measured, where the models price it within joined[g] of the
        parts' (the measured time they attribute to g's own joined work, which g takes with it when it leaves: never
        another group's work, nor the call's time the models do not explain, which would credit a group's leaving with
        the load the others carry), else None; a group whose bounds keep it holding a part (its `low`) never leaves.  A
        stretch's work changes in kind where a group leaves it (its crossing goes), so the vertex is a design point of
        its own [Box, Hunter & Hunter 2005, the factorial's corner points].  (parts, value.)"""
        best, parts = None, value(d['parts'])
        for g, moved in self._vertices(d, history):
            v = value(moved)
            if v <= parts + joined[g] and (best is None or v < best[1]):
                best = (moved, v)
        return best

    @staticmethod
    def _vertices(d, history):
        """The decision's vertices no call at its history measured: (g, parts) of each group g's leaving it, its units
        to the others in proportion to theirs on the grain, within the bounds."""
        parts, grain, n, out = d['parts'], d['grain'], len(d['parts']), []
        for g in range(n):
            others = sum(p for h, p in enumerate(parts) if h != g)
            if not parts[g] or d['low'][g] > 0 or not others:
                continue
            moved = [0 if h == g else (p + parts[g] * p // others) // grain * grain for h, p in enumerate(parts)]
            moved[max(range(n), key=lambda h: moved[h])] += sum(parts) - sum(moved)
            if not any(m > hi or m < lo for m, lo, hi in zip(moved, d['low'], d['high'])) and ','.join(map(str, moved)) not in history:
                out.append((g, moved))
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
        return self.lines.get((name, s, i)) or time([])

    def state(self):
        return {'window': self.window, 'memory': self.memory, 'budget': self.budget, 'decisions': self.decisions,
                'lines': [[d, s, i, line.state()] for (d, s, i), line in self.lines.items()], 'measure': self.measure}

    @classmethod
    def of(cls, state):
        b = cls({}, state.get('window', 3), state.get('memory'), state.get('budget'))
        b.decisions = {name: {'grain': 1, 'low': [0] * len(d['parts']), 'high': [sum(d['parts'])] * len(d['parts']), 'before': None,
                              'stands': False, **{k: v for k, v in d.items() if k != 'measure'},
                              'groups': d.get('groups') or [[i] for i in range(len(d['parts']))]}
                       for name, d in state['decisions'].items()}
        b.lines = {(d, s, int(i)): Points.of(line) for d, s, i, line in state.get('lines', [])}
        b.measure = state.get('measure') or next((d['measure'] for d in state['decisions'].values() if d.get('measure')), 'sum')
        return b


class Coupled:
    """A decision among members that run at once and slow each other (a node's engines, sharing its power, cores and
    memory: torch_mesh/engines.py), moved only between allocations its calls measured: a compass search over the
    lattice of allocations [Hooke & Jeeves 1961; Kolda, Lewis & Torczon 2003, Optimization by direct search, §3],
    whose measure is the call's median time, each member's slowing of the others in it as the allocation ran.

    From a centre (at first the given parts) the search runs every allocation a step of units away (one member given
    the step and another giving it, every ordered pair of members, within their bounds), each `window` calls before it
    counts; the centre moves to the one measured fastest where it beats the centre by more than the resolution (twice
    the larger of the two call times' relative standard errors, at least 1 %), else the step halves, from the largest
    power-of-two count of grains the total holds down to one grain, where the parts stand at the centre.  No allocation
    is chosen before it is measured, and none by a time it was not measured at.  A standing allocation whose last
    window's median call time moves past the resolution from the median it stood at starts the search again from it at
    the largest step, every other allocation's calls dropped (its performance changed [Clarke, Lastovetsky & Rychkov
    2011]).

    The total.  A decision whose total is another decision's part (a member's units of a decision among members,
    divided among the member's engines) follows it (`resize`): the search starts again at the old parts scaled to the
    new total, or, where members are `kept` (those whose units are built per range, a Neural Engine's compiled shares:
    metal-microbench docs/placement.md#moves), with their units kept where those still fit and the others taking the
    difference, so a move of the total rebuilds nothing of theirs.  The calls are kept by allocation, so a return to a
    total runs again none of the allocations measured there.

      c = Coupled(3456, 128, [0, 0, 0], [3456, 3456, 3456], [1152, 1152, 1152])
      c.observe([[t, ...], [t, ...], [t, ...]], [call, ...])   # each member's seconds, the calls' seconds
      c.parts -> the next allocation
      c.resize(4096) -> the next allocation of 4096 units
      Coupled.of(c.state()) -> the same search (JSON between processes)"""

    def __init__(self, total, grain, low, high, parts, window=3):
        self.total, self.grain, self.window = int(total), int(grain), int(window)
        self.low, self.high = [int(v) for v in low], [int(v) for v in high]
        self.parts = self.centre = tuple(int(p) for p in parts)
        self.able = [i for i, h in enumerate(self.high) if h > 0]
        self.seen, self.stands, self.stood, self.pending, self.kept = {}, False, None, None, []
        self.step = self._widest()

    def state(self):
        """The search as JSON data (Coupled.of)."""
        return {'total': self.total, 'grain': self.grain, 'window': self.window, 'low': self.low, 'high': self.high,
                'parts': list(self.parts), 'centre': list(self.centre), 'step': self.step, 'able': self.able,
                'stands': self.stands, 'stood': self.stood, 'pending': list(self.pending) if self.pending else None,
                'kept': list(self.kept), 'seen': [[list(p), e] for p, e in self.seen.items()]}

    @classmethod
    def of(cls, state):
        c = cls.__new__(cls)
        c.total, c.grain, c.window = int(state['total']), int(state['grain']), int(state['window'])
        c.low, c.high = [int(v) for v in state['low']], [int(v) for v in state['high']]
        c.able = list(state.get('able') or [i for i, h in enumerate(c.high) if h > 0])
        c.parts = tuple(state['parts'])
        c.centre = tuple(state.get('centre') or state['parts'])
        c.pending = tuple(state['pending']) if state.get('pending') else None
        c.kept = list(state.get('kept') or [])
        c.stands, c.stood = state['stands'], state['stood']
        c.seen = {tuple(p): e for p, e in state['seen']}
        c.step = int(state.get('step') or c._widest())
        return c

    def entries(self):
        """This decision's measured allocations: each its parts, its members' medians and the call's median."""
        return [{'parts': list(p), 'members': [self._median(m) if m else None for m in e['members']],
                 'call': self._median(e['calls'])} for p, e in self._measured().items()]

    def prior(self):
        """The parts the decision starts at: its own (nothing but its measured allocations moves them)."""
        return list(self.parts)

    def resize(self, total):
        """The total moved to `total` units (a multiple of the grain): each bound that was the whole total becomes the
        new one, and the search starts again at the parts moved to it (class docstring, The total)."""
        total = int(total)
        if total == self.total:
            return list(self.parts)
        if total % self.grain:
            raise ValueError(f'coupled: total {total} is not in grains of {self.grain}')
        old, self.total = self.total, total
        self.high = [total if h >= old else min(h, total) for h in self.high]
        self.low = [min(lo, total) for lo in self.low]
        kept = [i for i in self.kept if i in self.able]
        others = [i for i in self.able if i not in kept]
        held = sum(self.parts[i] for i in kept)
        parts = None
        if kept and others and held <= total - sum(self.low[i] for i in others):
            parts = list(self.parts)
            for i in others:
                parts[i] = 0
            parts[max(others, key=lambda i: (self.parts[i], -i))] = total - held
            if not all(self.low[i] <= parts[i] <= self.high[i] for i in range(len(parts))):
                parts = None
        if parts is None:
            parts = [self.grain * round(p * total / old / self.grain) if old else 0 for p in self.parts]
            parts[max(self.able, key=lambda i: (parts[i], -i))] += total - sum(parts)
            for i in self.able:
                parts = self._give(parts, i)
        return self._restart(parts)

    def bound(self, i, high):
        """Member i holds at most `high` units from now on (an allocation giving it more failed a requirement the times
        do not show: an operation's accuracy, a share that overran its wait, either growing with its units): the
        allocations past it leave the evidence, the current parts give their excess to the others (_give), and the
        search starts again there; a bound of none excludes it."""
        high = int(high) // self.grain * self.grain
        if high <= 0:
            return self.exclude(i)
        self.high[i] = min(self.high[i], high)
        self.seen = {p: e for p, e in self.seen.items() if p[i] <= self.high[i]}
        return self._restart(self._give(self.parts, i))

    def exclude(self, i):
        """Member i holds no units from now on (it failed a requirement the times do not show: an operation's
        accuracy): its bound goes to 0, its units to the others (_give), the allocations it held leave the evidence,
        and the search starts again there."""
        self.high[i], self.able = 0, [j for j in self.able if j != i]
        self.seen = {p: e for p, e in self.seen.items() if p[i] == 0}
        return self._restart(self._give(self.parts, i))

    def _give(self, parts, i):
        """`parts` with member i's units past its bound given to the other able members, each up to its own bound, the
        one holding the most first (ties to the lower member)."""
        parts = list(parts)
        excess = parts[i] - self.high[i]
        if excess > 0:
            parts[i] = self.high[i]
            for j in sorted((j for j in self.able if j != i), key=lambda j: (-parts[j], j)):
                moved = min(excess, self.high[j] - parts[j])
                parts[j], excess = parts[j] + moved, excess - moved
            parts[i] += excess
        return parts

    def _restart(self, parts):
        self.parts = self.centre = tuple(parts)
        self.step, self.stands, self.stood = self._widest(), False, None
        return list(self.parts)

    def observe(self, times, calls, prepare=False):
        """The calls at the current parts: `times` each member's seconds a call (empty where it holds nothing),
        `calls` the calls' seconds.  With `prepare`, the next allocation waits as `pending` until `prepared`."""
        entry = self.seen.setdefault(self.parts, {'members': [[] for _ in self.low], 'calls': []})
        for i, ts in enumerate(times):
            entry['members'][i].extend(float(t) for t in ts or [])
        entry['calls'].extend(float(t) for t in calls or [])
        chosen = self._next()
        if not prepare or chosen == self.parts and self.pending is None:
            self.parts, self.pending = chosen, None
        elif self.pending is None:
            self.pending = chosen
        return list(self.parts)

    def unprepared(self, expected=None):
        """The pending allocation given up (where `expected` names it: only that one): its preparation failed."""
        if self.pending and (expected is None or tuple(expected) == tuple(self.pending)):
            self.pending = None
        return list(self.parts)

    def prepared(self, expected=None):
        """The pending allocation taken (where `expected` names it: only that one)."""
        if self.pending and (expected is None or tuple(expected) == tuple(self.pending)):
            self.parts, self.pending = tuple(self.pending), None
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

    def _widest(self):
        """The search's first step: the largest power-of-two count of grains the total holds."""
        step = self.grain
        while step * 2 <= self.total:
            step *= 2
        return step

    def _around(self, centre):
        """The allocations a step from `centre`: member i given it and member j giving it, every ordered pair of able
        members, within the bounds."""
        out = []
        for i in self.able:
            for j in self.able:
                if i != j and centre[i] + self.step <= self.high[i] and centre[j] - self.step >= self.low[j]:
                    out.append(tuple(v + self.step if k == i else v - self.step if k == j else v for k, v in enumerate(centre)))
        return out

    def _faster(self, a, b):
        """Whether calls `a` measured faster than calls `b` by more than the resolution."""
        resolution = max(RESOLUTION, 2 * max(self._error(a), self._error(b)))
        return self._median(a) < self._median(b) * (1 - resolution)

    def _next(self):
        measured = self._measured()
        if self.parts not in measured:
            return self.parts
        if self.stands:
            calls = measured[self.centre]['calls'] if self.centre in measured else []
            now = self._median(calls[-self.window:])
            if now is not None and abs(now - self.stood) <= max(RESOLUTION, 2 * self._error(calls)) * self.stood:
                return self.centre
            e = self.seen.get(self.centre)
            self.seen = {self.centre: {'members': [m[-self.window:] for m in e['members']], 'calls': e['calls'][-self.window:]}} if e else {}
            self.stands, self.step = False, self._widest()
            measured = self._measured()
        while True:
            if self.centre not in measured:
                return self.centre
            around = self._around(self.centre)
            waiting = [p for p in around if p not in measured]
            if waiting:
                return waiting[0]
            best = min(around, key=lambda p: (self._median(measured[p]['calls']), p), default=None)
            if best is not None and self._faster(measured[best]['calls'], measured[self.centre]['calls']):
                self.centre = best
            elif self.step > self.grain:
                self.step //= 2
            else:
                self.stands, self.stood = True, self._median(measured[self.centre]['calls'])
                return self.centre
