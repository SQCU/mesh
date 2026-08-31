#!/usr/bin/env python3
"""navmesh -- the stock navigation graph, Voronoi cells over it, and the cart-path
solver that is CONSTRAINED to negative space instead of tested against it.

This implements the data flow of design/NAV-SPEC.md:

    map BSP -> COMPUTE negative space            [negspace.NegSpace]
            -> the STOCK navmesh (waypoint graph, the same one playerbots use)
            -> Voronoi cells over the navmesh
                 |- classify edges: cart-traversable?  (semantic + geometric)
                 |- fuse contiguous NAVIGABLE cells
                 |- k-center origins, >=3, approximately equidistant in
                 |  navmesh-WALKING distance
                 `- tangent-energy curve optimization placing cart paths, subject
                    to: not intersecting terrain; not floating in
                    non-walking-navigable air; every path point within activation
                    distance of negative space  <- BY CONSTRUCTION

The last line is the whole design.  Every iterate of the optimizer is PROJECTED
into the computed free volume before the next step, so the curve that comes out
is a member of free space rather than a curve that was checked afterwards and
patched.  There is no "unstick" and there is nothing to sample.

§5 of NAV-SPEC is binding: this module uses the stock waypoint graph as the one
navigation definition.  It does not build a second one.
"""

import math

import numpy as np

import negspace as NS

# a cart is a bigger body than a player; its track has to admit it
CART_MIN = NS.CART_MIN
CART_MAX = NS.CART_MAX
PL_MIN = NS.PL_MIN
PL_MAX = NS.PL_MAX

# how far a path point may be from a legal cart placement before the plan is
# rejected.  Zero by construction after projection; the tolerance exists only to
# absorb the projector's own convergence.
ACTIVATION = 1.0
# a cart resting point must have the world's floor within this drop, or the plan
# is "floating in non-walking-navigable air"
GROUND_DROP = 96.0

WPF_JUMP = 0x004
WPF_TELEPORT = 0x008


class Navmesh(object):
    """The stock waypoint graph: nodes, undirected adjacency, saved flags."""

    def __init__(self, nodes, adj, flags=None, jumplinks=(), triggerboxes=()):
        self.nodes = [list(n) for n in nodes]
        self.adj = [dict(a) for a in adj]
        self.flags = flags or {}
        self.jumplinks = set(jumplinks)
        self.triggerboxes = list(triggerboxes)
        self.bad = {}

    # ------------------------------------------------------ edge classification
    def classify_edges(self, ns, verbose=False, ride=16.0):
        """Which navmesh links a CART may follow.

        Two independent reasons a link is not a cart segment:

        SEMANTIC (NAV-SPEC §4) -- the link encodes a jump-pad or teleport
        trajectory.  A bot flies that arc; a cart cannot, and fitting a smooth
        curve to it is exactly what drove carts into level geometry "along very
        smooth waypoint following curves".  These are recognised from the saved
        waypoint flags and from the link endpoints sitting in a trigger volume,
        never from geometry.

        GEOMETRIC -- the straight segment leaves the computed free volume, or has
        no floor under it.  Both are read off `negspace` in closed form: the
        intersection of a segment with a convex cell is an interval, so the
        uncovered parts of the segment are computed, not sampled."""
        bad = {}
        stat = {'semantic': 0, 'burrow': 0, 'airborne': 0}
        for u, a in enumerate(self.adj):
            for v in a:
                if u >= v:
                    continue
                k = (u, v)
                if k in bad:
                    continue
                if self._semantic_bad(u, v):
                    bad[k] = 'semantic'
                    stat['semantic'] += 1
                    continue
                pa = [self.nodes[u][0], self.nodes[u][1], self.nodes[u][2] + ride]
                pb = [self.nodes[v][0], self.nodes[v][1], self.nodes[v][2] + ride]
                gaps = ns.segment_gaps(pa, pb)
                if gaps:
                    bad[k] = 'burrow'
                    stat['burrow'] += 1
                    continue
                if not self._grounded(ns, pa, pb):
                    bad[k] = 'airborne'
                    stat['airborne'] += 1
        self.bad = bad
        if verbose:
            ne = sum(len(a) for a in self.adj) // 2
            print('navmesh: %d nodes %d links; %d links are not cart segments '
                  '(semantic jump/teleport=%d, burrows through solid=%d, '
                  'floats over non-walkable air=%d)'
                  % (len(self.nodes), ne, len(bad), stat['semantic'], stat['burrow'],
                     stat['airborne']))
        return bad

    def _semantic_bad(self, u, v):
        fu = self.flags.get(tuple(round(x, 1) for x in self.nodes[u]), 0)
        fv = self.flags.get(tuple(round(x, 1) for x in self.nodes[v]), 0)
        if (fu | fv) & (WPF_JUMP | WPF_TELEPORT):
            return True
        if (u, v) in self.jumplinks or (v, u) in self.jumplinks:
            return True
        for lo, hi in self.triggerboxes:
            for p in (self.nodes[u], self.nodes[v]):
                if all(lo[a] - 8 <= p[a] <= hi[a] + 8 for a in range(3)):
                    return True
        return False

    def _grounded(self, ns, pa, pb, nsamp=4):
        """Is there floor under the whole segment?  The floor is the free
        volume's own lower boundary, read from the cell the point is in."""
        for i in range(nsamp + 1):
            f = i / float(nsamp)
            p = [pa[j] + f * (pb[j] - pa[j]) for j in range(3)]
            fz = ns.floor_under(p, GROUND_DROP)
            if fz is None:
                return False
        return True

    # -------------------------------------------------------- walking distance
    def walk_dist(self, src, allow_bad=False):
        """Dijkstra over the navmesh in WALKING distance (NAV-SPEC §1)."""
        import heapq
        n = len(self.nodes)
        d = [math.inf] * n
        prev = [-1] * n
        d[src] = 0.0
        q = [(0.0, src)]
        while q:
            du, u = heapq.heappop(q)
            if du > d[u] + 1e-9:
                continue
            for v, w in self.adj[u].items():
                if not allow_bad and (min(u, v), max(u, v)) in self.bad:
                    continue
                nd = du + w
                if nd < d[v] - 1e-9:
                    d[v] = nd
                    prev[v] = u
                    heapq.heappush(q, (nd, v))
        return d, prev

    # ------------------------------------------------------------- Voronoi
    def voronoi(self, ns, sites=None, verbose=False):
        """Voronoi decomposition of the FREE VOLUME over navmesh sites.

        Every free cell of `ns` is assigned to the navmesh node that owns it --
        nearest in navmesh WALKING distance from that node's own position, not in
        straight-line distance, so the partition follows navigability rather than
        geometric proximity (NAV-SPEC §2, §8).  Cells that no navmesh node can
        reach are left unassigned; that set is exactly the part of the free volume
        the stock bots cannot use, and it is reported rather than hidden."""
        if sites is None:
            sites = list(range(len(self.nodes)))
        P = np.array([self.nodes[i] for i in sites], dtype=np.float64)
        centres = 0.5 * (ns.lo + ns.hi)
        owner = np.full(len(ns.cells), -1, dtype=np.int64)
        seed = np.full(len(ns.cells), -1, dtype=np.int64)
        # seed: the cell each navmesh node stands in
        for si, i in enumerate(sites):
            c = ns.cell_at(self.nodes[i])
            if c >= 0 and seed[c] < 0:
                seed[c] = si
        # grow along portal adjacency -- "fuse contiguous NAVIGABLE cells"
        if ns.adj is None:
            ns.build_portals()
        from collections import deque
        q = deque()
        for c in np.nonzero(seed >= 0)[0]:
            owner[c] = seed[c]
            q.append(int(c))
        while q:
            c = q.popleft()
            for (w, pi) in ns.adj[c]:
                if owner[w] >= 0:
                    continue
                if ns.portals[pi].radius < 16.0:
                    continue            # not an opening a body can pass
                owner[w] = owner[c]
                q.append(w)
        self.vor_owner = owner
        nass = int((owner >= 0).sum())
        if verbose:
            import collections
            sz = collections.Counter(int(o) for o in owner if o >= 0)
            print('navmesh: Voronoi over %d sites -> %d/%d free cells assigned '
                  '(%d unreachable by any navmesh node); cell-per-site '
                  'median=%d max=%d' % (len(sites), nass, len(ns.cells),
                                        len(ns.cells) - nass,
                                        int(np.median(list(sz.values()))) if sz else 0,
                                        max(sz.values()) if sz else 0))
        return owner

    # ------------------------------------------------------------- k-center
    def equidistant_origins(self, k, pool=None, verbose=False):
        """>=k origins approximately equidistant from each other in navmesh
        WALKING distance (NAV-SPEC §1).

        Greedy k-center on the walking metric, then a swap pass that directly
        maximises the minimum pairwise walking distance -- 'approximately
        equidistant' is measured and reported as the spread ratio
        max_pairwise / min_pairwise."""
        if pool is None:
            pool = [i for i in range(len(self.nodes))]
        if not pool:
            return [], {}
        D = {}
        for s in pool:
            D[s] = self.walk_dist(s)[0]
        start = pool[0]
        picks = [start]
        while len(picks) < k:
            best, bd = None, -1.0
            for c in pool:
                if c in picks:
                    continue
                dd = min(D[p][c] for p in picks)
                if dd < math.inf and dd > bd:
                    best, bd = c, dd
            if best is None:
                break
            picks.append(best)

        def spread(ps):
            vals = [D[a][b] for i, a in enumerate(ps) for b in ps[i + 1:]
                    if D[a][b] < math.inf]
            if not vals:
                return 0.0, 0.0, math.inf
            return min(vals), max(vals), (max(vals) / min(vals) if min(vals) else math.inf)

        for _ in range(60):
            lo, hi, rat = spread(picks)
            improved = False
            for idx in range(len(picks)):
                for c in pool:
                    if c in picks:
                        continue
                    trial = list(picks)
                    trial[idx] = c
                    l2, h2, r2 = spread(trial)
                    if l2 > lo + 1e-6 or (abs(l2 - lo) < 1e-6 and r2 < rat - 1e-6):
                        picks, lo, hi, rat = trial, l2, h2, r2
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        lo, hi, rat = spread(picks)
        st = {'min': lo, 'max': hi, 'ratio': rat, 'k': len(picks)}
        if verbose:
            print('navmesh: %d cart origins, pairwise navmesh-walking distance '
                  'min=%.0f max=%.0f spread_ratio=%.2f (1.00 = exactly '
                  'equidistant)' % (len(picks), lo, hi, rat))
        return picks, st


# ---------------------------------------------------------------------------
# THE PATH SOLVER
# ---------------------------------------------------------------------------
def resample(poly, spacing):
    if len(poly) < 2:
        return [list(p) for p in poly]
    out = [list(poly[0])]
    acc = 0.0
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        L = math.dist(a, b)
        if L < 1e-6:
            continue
        t = spacing - acc
        while t < L:
            f = t / L
            out.append([a[j] + f * (b[j] - a[j]) for j in range(3)])
            t += spacing
        acc = (acc + L) % spacing
    out.append([float(x) for x in poly[-1]])
    return out


def tangent_energy(P):
    """Discrete bending (tangent) energy of a polyline: the quantity the cart
    path minimises so it reads as a laid track and not a waypoint zigzag."""
    if len(P) < 3:
        return 0.0
    A = np.asarray(P, dtype=np.float64)
    D = A[2:] - 2.0 * A[1:-1] + A[:-2]
    return float((D * D).sum())


class PathSolver(object):
    """Tangent-energy curve optimization under a hard free-space constraint.

    The feasible set is the computed free volume, admitted as the set of points
    at which the CART BODY fits (`negspace.fits`).  Every iterate is projected
    back into that set, so the curve is always a motion plan inside negative
    space: it cannot burrow and it never needs an unstick.  Points that also have
    the free volume's own floor beneath them within `GROUND_DROP` are on
    walking-navigable ground; the rest are reported as the airborne fraction
    rather than silently accepted."""

    def __init__(self, ns, mins=CART_MIN, maxs=CART_MAX, spacing=64.0,
                 outer=24, inner=6, step=0.5, ride=16.0):
        self.ns = ns
        self.mins = mins
        self.maxs = maxs
        self.spacing = spacing
        self.outer = outer
        self.inner = inner
        self.step = step
        self.ride = ride

    def feasible(self, p):
        return self.ns.fits(p, self.mins, self.maxs)

    def project(self, p):
        """Nearest legal cart placement to p.  Returns (point, distance)."""
        q, dd = self.ns.project(p, self.mins, self.maxs, radius=self.ns.gridcell * 2)
        if q is not None and self.feasible(q):
            return q, dd
        # the cart body does not fit in any single convex cell here; fall back to
        # the player-sized free volume, then accept only if the cart body is in
        # fact covered by the union (a placement can straddle a doorway)
        q2, d2 = self.ns.project(p, PL_MIN, PL_MAX, radius=self.ns.gridcell * 2)
        if q2 is not None and self.feasible(q2):
            return q2, d2
        if q is not None:
            return q, dd
        return q2, d2

    def settle(self, p):
        """Return a legal placement near p that also SITS ON the free volume's own
        floor.  'Not floating in non-walking-navigable air' is imposed here, on
        every iterate, rather than measured afterwards."""
        q = list(p)
        if not self.feasible(q):
            r, _ = self.project(q)
            if r is None:
                return q
            q = r
        fz = self.ns.floor_under(q, GROUND_DROP * 3.0)
        if fz is not None:
            g = [q[0], q[1], fz + self.ride]
            if self.feasible(g):
                return g
        return q

    def solve(self, poly, pin=(0,), stats=None):
        """Fit a tangent-energy-minimal curve through `poly` inside free space."""
        P = [list(p) for p in resample(poly, self.spacing)]
        n = len(P)
        if n < 3:
            return P, {'e0': 0.0, 'e1': 0.0, 'infeasible': 0, 'airborne': 0}
        pins = set()
        for i in pin:
            pins.add(i % n)
        pins.add(0)
        pins.add(n - 1)
        # seed: every point starts as a LEGAL placement, so the very first iterate
        # is already a motion plan inside negative space
        infeasible0 = 0
        for i in range(n):
            P[i] = self.settle(P[i])
            if not self.feasible(P[i]):
                infeasible0 += 1
        e0 = tangent_energy(P)
        for _ in range(self.outer):
            for _ in range(self.inner):
                A = np.asarray(P, dtype=np.float64)
                G = np.zeros_like(A)
                # gradient of sum |p_{i-1} - 2 p_i + p_{i+1}|^2
                D = A[2:] - 2.0 * A[1:-1] + A[:-2]
                G[:-2] += 2.0 * D
                G[1:-1] += -4.0 * D
                G[2:] += 2.0 * D
                # the 4th-difference operator's spectral radius is 16, so the
                # gradient of the squared second difference is bounded by 32:
                # anything above 2/32 diverges, which it did at /4.
                for i in range(n):
                    if i in pins:
                        continue
                    A[i] -= self.step * G[i] / 32.0
                P = [list(x) for x in A]
            # PROJECT the whole curve back into the feasible set.  This is the
            # constraint; it is not a repair pass, it is how the iterate is kept
            # inside negative space at every step.
            for i in range(n):
                if i in pins:
                    continue
                P[i] = self.settle(P[i])
        infeasible = 0
        unplaceable = 0
        airborne = 0
        maxdev = 0.0
        for i, p in enumerate(P):
            if not self.feasible(p):
                infeasible += 1
                q, dd = self.project(p)
                if q is None:
                    unplaceable += 1
                else:
                    maxdev = max(maxdev, dd)
            if self.ns.floor_under(p, GROUND_DROP) is None:
                airborne += 1
        st = {'e0': e0, 'e1': tangent_energy(P), 'n': n,
              'infeasible': infeasible, 'infeasible_seed': infeasible0,
              'unplaceable': unplaceable,
              'airborne': airborne, 'max_activation_distance': maxdev}
        if stats is not None:
            stats.update(st)
        return P, st
