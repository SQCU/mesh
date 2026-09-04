#!/usr/bin/env mesh-python
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'payload', 'tools'))
import negspace as NS

__all__ = ['Body', 'STAND', 'CROUCH', 'CART', 'ground', 'walk', 'fall',
           'jump_gap', 'DEFAULT_MODES', 'traversable', 'route', 'occupancy']

class Body(object):

    def __init__(self, name, mins, maxs, step=18.0, jump=88.0):
        self.name, self.mins, self.maxs = name, tuple(mins), tuple(maxs)
        self.step = step
        self.jump = jump

    @property
    def height(self):
        return self.maxs[2] - self.mins[2]

STAND = Body('stand', NS.PL_MIN, NS.PL_MAX)
CROUCH = Body('crouch', NS.PL_MIN, (NS.PL_MAX[0], NS.PL_MAX[1], NS.PL_MAX[2] - 24.0))
CART = Body('cart', getattr(NS, 'CART_MIN', (-30.0, -30.0, -24.0)),
            getattr(NS, 'CART_MAX', (30.0, 30.0, 40.0)), step=0.0, jump=0.0)

def _swept_clear(ns, body, a, b, tol=1e-3):
    iv = ns.segment_intervals(a, b, body.mins, body.maxs)
    if not iv:
        return False
    if iv[0][0] > tol or iv[-1][1] < 1.0 - tol:
        return False
    end = iv[0][1]
    for t0, t1 in iv[1:]:
        if t0 > end + tol:
            return False
        end = max(end, t1)
    return True

def ground(ns, body, p, lift=1.0, maxdrop=512.0):
    z = ns.floor_under(p, maxdrop=maxdrop, footprint=(body.maxs[0], body.maxs[1]))
    if z is None:
        return None
    return (p[0], p[1], z - body.mins[2] + lift)

def walk(ns, body, a, b, sub=4):
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
    drop = a[2] - b[2]
    if drop <= 0.0 or drop > maxdrop:
        return False
    over = (b[0], b[1], a[2])
    return _swept_clear(ns, body, a, over) and _swept_clear(ns, body, over, b)

def jump_gap(ns, body, a, b, arc=5):
    if body.jump <= 0.0:
        return False
    pts = []
    for i in range(arc + 1):
        t = i / float(arc)
        z = a[2] + (b[2] - a[2]) * t + body.jump * 4.0 * t * (1.0 - t)
        pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, z))
    return all(_swept_clear(ns, body, pts[i], pts[i + 1]) for i in range(arc))

DEFAULT_MODES = (walk, fall, jump_gap)

def traversable(ns, body, a, b, modes=DEFAULT_MODES):
    for m in modes:
        if m(ns, body, a, b):
            return m.__name__
    return None

def route(ns, points, body=STAND, modes=DEFAULT_MODES):
    return [(i, traversable(ns, body, points[i], points[i + 1], modes))
            for i in range(len(points) - 1)]

def occupancy(ns, points, bodies=(STAND, CROUCH, CART)):
    return {b.name: [ns.standable(p, b.mins, b.maxs) for p in points]
            for b in bodies}
