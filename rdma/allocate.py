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
   s'_i = s_i - eta (g_i - mu) / (b_i kappa'), mu keeping sum s' = W.  A rank that falls below
   its lower bound is fixed there and mu re-solved [Bitran & Hax 1981].  eta = 1 is the
   equal-finish solve.
4. c_i = low_i + g floor((s'_i - low_i) / g), the leftover grains to the largest remainders,
   ties to the lower rank.  s' stays the iterate; c is the next call's operand and regressor.

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
        free = [True] * n
        while True:
            held = sum(self.low[i] for i in range(n) if not free[i])
            mu = ((self.total - held - sum(self.s[i] for i in range(n) if free[i])
                   + eta * sum(w[i] * g[i] for i in range(n) if free[i]))
                  / (eta * sum(w[i] for i in range(n) if free[i])))
            new = [self.s[i] - eta * w[i] * (g[i] - mu) if free[i] else float(self.low[i]) for i in range(n)]
            below = [i for i in range(n) if free[i] and new[i] < self.low[i]]
            if not below:
                break
            for i in below:
                free[i] = False
        self.s = new
        c = [self.low[i] + self.grain * math.floor((new[i] - self.low[i]) / self.grain) for i in range(n)]
        order = sorted(range(n), key=lambda i: (c[i] - new[i], i))
        for j in range((self.total - sum(c)) // self.grain):
            c[order[j % n]] += self.grain
        self.shares = c
        return list(c)
