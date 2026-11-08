"""design/joint-compilation.md §3, row 9"""
import math


class Allocator:

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
    c, _ = min_max(total, grain, c, high, [lambda x, i=i: a[i] + b[i] * x for i in range(n)])
    return c, s, T


def min_max(total, grain, low, high, cost, shared=None):
    n = len(low)

    def fill(cap):
        c = list(low)
        for _ in range((total - sum(low)) // grain):
            fits = [i for i in range(n) if c[i] + grain <= min(high[i], cap)]
            if not fits:
                break
            i = min(fits, key=lambda i: (cost[i](c[i] + grain), i))
            c[i] += grain
        return c, max(cost[i](c[i]) for i in range(n))

    if shared is None:
        return fill(math.inf)
    fills = [fill(cap) for cap in range(max(low), total + 1, grain)]
    c, t = min((f for f in fills if sum(f[0]) == sum(fills[-1][0])), key=lambda f: f[1] + shared(f[0]))
    return c, t + shared(c)


def capacity(total, grain, low, high):
    return [min(high[i], total - sum(low) + low[i]) // grain * grain for i in range(len(low))]
