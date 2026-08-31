#!/usr/bin/env python3
"""nav.py -- navigability as COMPOSABLE predicates over player-shaped colliders.

A ray is not a player. Casting one answers "is this line of points free", which
is not "can a body get from here to there" -- and the harness that asked the
first question while reporting the second had the test inlined into it, so there
was one way to be right and no way to ask a different question.

Everything here is built from a BODY (a collider with real hull extents) and a
MODE (a way that body may move), both of which are values. A route test is the
composition of modes over consecutive points, so adding "can it crouch through"
or "can the cart make this corner" is a new value, not a new copy of the loop.

Solidity is negspace's, unchanged: `solid(p) == ns.cell_at(p) < 0`. The swept
tests are `ns.segment_intervals(a, b, mins, maxs)`, which returns the EXACT
parametric intervals of the segment where the swept hull is free -- a closed-form
answer for a convex cell, not a march. This module adds no geometry and no
second definition of solidity; it only composes.

    bodies   STAND / CROUCH / CART, from Xonotic's own hull constants
    modes    walk, fall, jump_gap  -- each (ns, body, a, b) -> bool
    compose  traversable(...modes)  first mode that succeeds
             route(points, ...)     per-edge, reports WHICH mode carried it
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'payload', 'tools'))
import negspace as NS

__all__ = ['Body', 'STAND', 'CROUCH', 'CART', 'ground', 'walk', 'fall',
           'jump_gap', 'DEFAULT_MODES', 'traversable', 'route', 'occupancy']


class Body(object):
    """A collider. Hull extents are Xonotic's, not a symmetric approximation."""

    def __init__(self, name, mins, maxs, step=18.0, jump=88.0):
        self.name, self.mins, self.maxs = name, tuple(mins), tuple(maxs)
        self.step = step        # how high it can walk up without jumping
        self.jump = jump        # how high a standing jump clears

    @property
    def height(self):
        return self.maxs[2] - self.mins[2]


# PL_MIN/PL_MAX are the engine's standing hull (32 x 32 x 69).
STAND = Body('stand', NS.PL_MIN, NS.PL_MAX)
CROUCH = Body('crouch', NS.PL_MIN, (NS.PL_MAX[0], NS.PL_MAX[1], NS.PL_MAX[2] - 24.0))
CART = Body('cart', getattr(NS, 'CART_MIN', (-30.0, -30.0, -24.0)),
            getattr(NS, 'CART_MAX', (30.0, 30.0, 40.0)), step=0.0, jump=0.0)


def _swept_clear(ns, body, a, b, tol=1e-3):
    """Is the whole segment free to the SWEPT hull? Exact, via negspace.

    Free space is a CELL COMPLEX, so a segment crossing a cell boundary comes
    back as several abutting intervals. What matters is whether they leave a
    GAP, not how many there are -- requiring a single interval called every
    multi-cell crossing blocked, which was every edge of a helical corridor.
    """
    iv = ns.segment_intervals(a, b, body.mins, body.maxs)
    if not iv:
        return False
    if iv[0][0] > tol or iv[-1][1] < 1.0 - tol:
        return False
    end = iv[0][1]
    for t0, t1 in iv[1:]:
        if t0 > end + tol:          # a real gap: solid between two free spans
            return False
        end = max(end, t1)
    return True


# ------------------------------------------------------------------ modes
# Each mode is (ns, body, a, b) -> bool. They are values; add one, don't edit
# a caller.

def ground(ns, body, p, lift=1.0, maxdrop=512.0):
    """Put `body` ON the floor under p: the position a standing player occupies.

    Hull extents are asymmetric (mins.z -24, maxs.z +45), so a standing centre is
    floor - mins.z, not floor + half-height. Guessing that offset put the box two
    units off the ground and every swept test then clipped the floor.
    """
    z = ns.floor_under(p, maxdrop=maxdrop, footprint=(body.maxs[0], body.maxs[1]))
    if z is None:
        return None
    return (p[0], p[1], z - body.mins[2] + lift)


def walk(ns, body, a, b, sub=4):
    """Travel following the floor, with the engine's own step tolerance.

    This is what Quake-family movement IS: try to move, and if blocked, lift by
    the step height and move -- which is also how a player walks up a ramp or a
    stair. So there is no separate "step_up" mode; splitting them produced two
    predicates for one physics, and on a helical corridor the strict-level half
    matched nothing at all while the other claimed all 204 edges. One mode, and
    the sweep is done with the hull lifted by the step tolerance and then
    re-settled, exactly as the mover does it.

    A straight sweep between two grounded points on a curved surface dips THROUGH
    it, so the path is re-grounded at each substep and swept between consecutive
    grounded positions. Rises steeper than `body.step` per substep are not
    walking, and are left for fall/jump_gap to claim.
    """
    pts = []
    for i in range(sub + 1):
        t = i / float(sub)
        probe = (a[0] + (b[0] - a[0]) * t,
                 a[1] + (b[1] - a[1]) * t,
                 max(a[2], b[2]) + body.step + 1.0)
        g = ground(ns, body, probe)
        if g is None or not ns.fits(g, body.mins, body.maxs):
            return False
        pts.append(g)
    lift = max(1.0, body.step)
    for i in range(sub):
        if pts[i + 1][2] - pts[i][2] > body.step + 1e-6:
            return False
        up0 = (pts[i][0], pts[i][1], pts[i][2] + lift)
        up1 = (pts[i + 1][0], pts[i + 1][1], pts[i + 1][2] + lift)
        if not _swept_clear(ns, body, up0, up1):
            return False
    return True


def fall(ns, body, a, b, maxdrop=512.0):
    """b is below a: step off and land, provided the drop is survivable."""
    drop = a[2] - b[2]
    if drop <= 0.0 or drop > maxdrop:
        return False
    over = (b[0], b[1], a[2])
    return _swept_clear(ns, body, a, over) and _swept_clear(ns, body, over, b)


def jump_gap(ns, body, a, b, arc=5):
    """Ballistic hop: sweep the hull along a parabola, not along a ray.

    The arc is tested as consecutive SWEPT segments, so the body is what has to
    fit at every stage -- a ray would sail over a lip the hull would clip.
    """
    if body.jump <= 0.0:
        return False
    pts = []
    for i in range(arc + 1):
        t = i / float(arc)
        z = a[2] + (b[2] - a[2]) * t + body.jump * 4.0 * t * (1.0 - t)
        pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, z))
    return all(_swept_clear(ns, body, pts[i], pts[i + 1]) for i in range(arc))


DEFAULT_MODES = (walk, fall, jump_gap)


# ------------------------------------------------------------- composition

def traversable(ns, body, a, b, modes=DEFAULT_MODES):
    """First mode that carries `body` from a to b, or None."""
    for m in modes:
        if m(ns, body, a, b):
            return m.__name__
    return None


def route(ns, points, body=STAND, modes=DEFAULT_MODES):
    """Per-edge traversal of a polyline. Returns [(i, mode-or-None)].

    Reporting WHICH mode carried each edge is the point: "94% navigable" hides
    whether a route walks or needs a jump at every third step.
    """
    return [(i, traversable(ns, body, points[i], points[i + 1], modes))
            for i in range(len(points) - 1)]


def occupancy(ns, points, bodies=(STAND, CROUCH, CART)):
    """Which bodies can stand at each point. A corridor that fits a player and
    not the cart is navigable and still fails the game."""
    return {b.name: [ns.standable(p, b.mins, b.maxs) for p in points]
            for b in bodies}
