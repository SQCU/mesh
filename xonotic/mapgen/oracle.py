#!/usr/bin/env python3
"""oracle.py -- the world-space geometry oracle over the ASSEMBLED world.

design/AGENDA.md R30 zeroed G15 with: "EVERY fusion validator works on a proxy
rather than on the assembled geometry... Nothing can answer 'is this point, in the
assembled fused world, inside geometry?'" -- solid_brush_at is source-space, the
void audit asks "is the screen black", the flood-fill answers WAYPOINT-GRAPH
connectivity, joinview measures path length, fusegraph measures abstract topology.

It was only a hard tool while the world was something we carved and then had to
interrogate. When the world is AUTHORED the oracle is exact and nearly free: every
brush is a convex intersection of halfspaces we placed ourselves, so

    solid(p)  ==  any brush has  n_i . p <= d_i  for all its faces

is a decision procedure, not an estimate. No ray marching, no point-sample that
can miss, no second definition of solidity to disagree with the first.

This matches negspace.py's law (`solid(p) == ns.cell_at(p) < 0`) rather than
competing with it: negspace derives free volume from a COMPILED bsp, this decides
solidity on the SOURCE brushes before compilation. They must agree, and where a
map is generated the source is authoritative because it is what produced the bsp.
"""
import math

EPS = 0.1


def _planes(brush):
    """Outward halfspaces of an authored Brush: (normal, d) with interior n.p <= d."""
    pts = [p for f, _ in brush.faces for p in f]
    cx = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
    out = []
    for f, _ in brush.faces:
        a, b, c = f[0], f[1], f[2]
        u = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
        v = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
        n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        L = math.sqrt(sum(x*x for x in n))
        if L < 1e-9:
            continue
        n = tuple(x / L for x in n)
        d = sum(n[i]*a[i] for i in range(3))
        if sum(n[i]*cx[i] for i in range(3)) > d:   # flip to face outward
            n, d = tuple(-x for x in n), -d
        out.append((n, d))
    return out


class Oracle:
    """Solidity of an assembled world. `offsets` places each tile, as placement.py decides."""

    def __init__(self, tiles):
        # tiles: [(brushes, offset)] -- offset is how the tile is placed in the world
        self.brushes = []
        for brushes, off in tiles:
            for br in brushes:
                pl = _planes(br)
                if not pl:
                    continue
                pl = [(n, d + n[0]*off[0] + n[1]*off[1] + n[2]*off[2]) for n, d in pl]
                lo = [min(p[i] + off[i] for f, _ in br.faces for p in f) for i in range(3)]
                hi = [max(p[i] + off[i] for f, _ in br.faces for p in f) for i in range(3)]
                self.brushes.append((pl, lo, hi))

    def solid_at(self, p):
        for pl, lo, hi in self.brushes:
            if any(p[i] < lo[i] - EPS or p[i] > hi[i] + EPS for i in range(3)):
                continue
            if all(n[0]*p[0] + n[1]*p[1] + n[2]*p[2] <= d + EPS for n, d in pl):
                return True
        return False

    def fits(self, p, half=(16.0, 16.0, 24.0)):
        """Does a player box centred at p clear geometry? Xonotic's hull is 32x32x48."""
        for dx in (-half[0], 0.0, half[0]):
            for dy in (-half[1], 0.0, half[1]):
                for dz in (-half[2], 0.0, half[2]):
                    if self.solid_at((p[0]+dx, p[1]+dy, p[2]+dz)):
                        return False
        return True

    def clearance(self, p, cap=512.0, step=8.0):
        """Distance straight down to solid -- how far above a floor p is."""
        t = 0.0
        while t < cap:
            if self.solid_at((p[0], p[1], p[2] - t)):
                return t
            t += step
        return cap

    def floor_under(self, p, cap=512.0, step=4.0, tol=0.25):
        """Z of the floor SURFACE below p, or None.

        The coarse march finds a solid sample, then bisects to the surface. A
        quantized answer is not good enough: the caller rests a player box on
        this value, and a coarse hit lies up to `step` BELOW the true surface, so
        the box gets placed intersecting the floor and the point is reported
        unstandable. That alone condemned 38% of a walkable corridor.
        """
        t = 0.0
        while t < cap:
            if self.solid_at((p[0], p[1], p[2] - t)):
                lo, hi = t, max(0.0, t - step)      # solid at lo, free at hi
                while lo - hi > tol:
                    mid = 0.5 * (lo + hi)
                    if self.solid_at((p[0], p[1], p[2] - mid)):
                        lo = mid
                    else:
                        hi = mid
                return p[2] - lo
            t += step
        return None

    def standable(self, p, hover=64.0, half=(16.0, 16.0, 24.0), lift=2.0):
        """Can a player STAND at p: floor under the whole footprint, and head room.

        A standing player's box rests on the HIGHEST surface its footprint
        touches, so the floor height is sampled under every corner and the max is
        taken -- then only the volume ABOVE that is required to be free. The
        earlier version tested the box centred at an arbitrary probe height and
        called 36% of a perfectly walkable helical corridor unnavigable; lifting
        it a fixed 2 units then called 100% of it unnavigable. Both were the same
        mistake: this floor is a HELICOID, its height varies across a 32x32
        footprint, and no single sample point stands in for the surface. The
        engine resolves a stand to the highest contacted surface, so the oracle
        does too, or it condemns geometry players walk on every match.
        """
        zs = []
        for dx in (-half[0], 0.0, half[0]):
            for dy in (-half[1], 0.0, half[1]):
                z = self.floor_under((p[0] + dx, p[1] + dy, p[2]), cap=hover + half[2])
                if z is None:
                    return False
                zs.append(z)
        base = max(zs) + lift
        # head room: the box above the floor must be clear at every corner
        for dx in (-half[0], 0.0, half[0]):
            for dy in (-half[1], 0.0, half[1]):
                for h in (0.0, half[2], half[2] * 2.0):
                    if self.solid_at((p[0] + dx, p[1] + dy, base + h)):
                        return False
        return True

    def trace(self, a, b, step=8.0):
        """First solid point along a->b, or None. The sightline test."""
        d = [b[i] - a[i] for i in range(3)]
        L = math.sqrt(sum(x*x for x in d))
        if L < 1e-6:
            return None
        u = [x / L for x in d]
        t = 0.0
        while t <= L:
            p = (a[0]+u[0]*t, a[1]+u[1]*t, a[2]+u[2]*t)
            if self.solid_at(p):
                return p
            t += step
        return None
