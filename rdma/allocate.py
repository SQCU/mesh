# design/whole-function.md
import math
import itertools

RESOLUTION = 0.01  # Configuration-time memory-fit resolution.


def partition_bounds(parts, extent):
    """Exact adjacent cuts of a declared extent, weighted by rank shares.

    Integer prefix rounding preserves coverage, including tails and empty
    shares. Divisible extents retain the original capacity-block boundaries.
    This relation is consumed during preparation, never by numerical replay.
    """
    total = sum(parts)
    return tuple(extent * prefix // total for prefix in itertools.accumulate(parts, initial=0))


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
    scale = total / sum(parts) if sum(parts) else 0
    values = [p * scale for p in parts] if scale else [total / len(parts)] * len(parts)
    return project(values, total, grain, [0] * len(parts), [total] * len(parts))


SHARE_TICKS = 1 << 40


def project(values, total, grain, low, high):
    """design/whole-function.md#finite-differences"""
    grain, total = int(grain), int(total)
    if grain <= 0 or not values or len(values) != len(low) or len(values) != len(high):
        raise ValueError('allocation dimensions and grain must be valid')
    lower = [math.ceil(x / grain) for x in low]
    upper = [math.floor(x / grain) for x in high]
    units = total // grain
    if total % grain or not sum(lower) <= units <= sum(upper):
        raise ValueError('infeasible allocation lattice')
    values = [float(x) / grain for x in values]
    left = min(x - hi for x, hi in zip(values, upper)) - 1
    right = max(x - lo for x, lo in zip(values, lower)) + 1
    for _ in range(64):
        middle = (left + right) / 2
        if sum(max(lo, min(hi, x - middle)) for x, lo, hi in zip(values, lower, upper)) > units:
            left = middle
        else:
            right = middle
    real = [max(lo, min(hi, x - right)) for x, lo, hi in zip(values, lower, upper)]
    result = [math.floor(x) for x in real]
    order = sorted(range(len(result)), key=lambda i: (-(real[i] - result[i]), i))
    for i in order:
        if sum(result) == units:
            break
        if result[i] < upper[i]:
            result[i] += 1
    if sum(result) != units:
        raise ArithmeticError('allocation projection lost units')
    return [x * grain for x in result]


def simplex(values, total, low, high):
    """Project linear shares onto a bounded simplex with 40 fractional bits.

    The precision is explicit (9.1e-13 of a decision), with exact zero/one and
    conservation, without exponential tails or a positive allocation floor.
    The transport's coarser allocation is separate.
    """
    if not total > 0 or not all(math.isfinite(v) for v in [total, *values, *low, *high]):
        raise ValueError('simplex operands must be finite with a positive total')
    ticks = SHARE_TICKS
    from fractions import Fraction
    ratio = Fraction(ticks) / Fraction(total)
    # Tiny operands are explicitly below representation precision. Do not
    # divide them into subnormals on the way to an inevitable rounded zero.
    scaled = [0.0 if abs(v) < total / ticks else v / total * ticks for v in values]
    result = project(scaled, ticks, 1,
                     [math.ceil(Fraction(v) * ratio) for v in low],
                     [math.floor(Fraction(v) * ratio) for v in high])
    return [v / ticks * total for v in result]
