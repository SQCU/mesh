#!/usr/bin/env python3
"""e2e.py -- apply the real generator and the real oracle to a real map.

Not a re-simulation and not a unit test: it calls the SAME spiralgen.Spiral.build()
that writes the shipped .map and the SAME Oracle the fuser uses, through their
public surface, and measures the world they actually produce.

What it measures, all on the ASSEMBLED WORLD rather than on a proxy:
  navigability   fraction of the corridor centerline where a 32x32x48 player box
                 fits AND has floor under it -- not waypoint-graph connectivity
  aperture open  the mouth is free volume when the plug is dropped
  aperture seal  the same mouth is solid when the plug ships (so the standalone
                 map that compiled leak-free is the same geometry a join mates to)
  sightline      the 'in' vantage can see the 'out' vantage through the mouth,
                 which is exactly the frame joinshot renders
"""
import argparse, sys, math
import spiralgen as SG
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'payload', 'tools'))
# ONE definition of solidity.  `negspace.from_brushes` is the SOURCE entry point
# of the same object the fuser uses on compiled worlds, so a generated map and a
# shipped one are judged by the same law.  `oracle.py` was a second exact
# implementation of it and is deleted.
from negspace import from_brushes as Oracle
import nav as NAV


def run(args):
    args = SG.randomize(args)
    sp = SG.Spiral(args)
    brushes, centers, radials = sp.build()
    aps, plugs = sp.apertures, sp.plugs
    plugged = brushes + [b for pl in plugs for b in pl]

    open_world = Oracle([(brushes, (0, 0, 0))])
    seal_world = Oracle([(plugged, (0, 0, 0))])
    print('brushes %d  apertures %d  plug brushes %d'
          % (len(brushes), len(aps), sum(len(p) for p in plugs)))

    # 1. NAVIGABILITY, composed -- not a raycast, and not inlined here.
    # nav.route sweeps a real player-shaped collider along every consecutive
    # pair and reports WHICH mode carried each edge; nav.occupancy asks which
    # BODIES can stand at each point. "94% navigable" hides whether a route
    # walks or needs a jump every third step, and hides a corridor that fits a
    # player but not the cart -- which is navigable and still fails the game.
    # stand the body on the floor rather than guessing an eye height: the hull
    # is asymmetric, so a standing centre is floor - mins.z.
    eye = [g for g in (NAV.ground(open_world, NAV.STAND, (c[0], c[1], c[2] + 40.0))
                       for c in centers) if g is not None]
    edges = NAV.route(open_world, eye, NAV.STAND)
    carried = [m for _, m in edges if m]
    by_mode = {m: carried.count(m) for m in set(carried)}
    frac = len(carried) / float(max(1, len(edges)))
    print('navigable edges: %d/%d = %.1f%%  by mode %s'
          % (len(carried), len(edges), 100 * frac, by_mode))
    occ = NAV.occupancy(open_world, eye)
    for name, hits in occ.items():
        print('  body %-6s stands at %d/%d' % (name, sum(hits), len(hits)))

    # 2/3. APERTURE open vs plugged, at the mouth itself.
    nopen = nseal = 0
    for a in aps:
        m, n = a['origin'], a['normal']
        # probe INSIDE the shell volume the plug occupies (the mouth origin is
        # the plug's OUTER face, so stepping outward from it leaves the plug
        # entirely and reports every plug as non-sealing).
        probe = tuple(m[i] - n[i] * (args.thickness * 0.5) for i in range(3))
        probe = (probe[0], probe[1], probe[2] + 40.0)
        o, s = open_world.fits(probe), seal_world.fits(probe)
        nopen += o; nseal += (not s)
        print('  aperture %d  mouth free unplugged=%s  sealed when plugged=%s'
              % (a['id'], o, not s))
    # 4. SIGHTLINE through each mouth, the frame joinshot shoots.
    # a sightline is a visibility question and stays a ray; whether a PLAYER can
    # pass through the mouth is a swept-collider question and is asked as one.
    npass = 0
    for a in aps:
        vin = [v for v in a['vantages'] if v['side'] == 'in'][0]['origin']
        vout = [v for v in a['vantages'] if v['side'] == 'out'][0]['origin']
        thru = NAV.traversable(open_world, NAV.STAND, tuple(vin), tuple(vout))
        shut = NAV.traversable(seal_world, NAV.STAND, tuple(vin), tuple(vout))
        npass += (thru is not None and shut is None)
        print('  aperture %d  body passes unplugged=%s (%s)  blocked plugged=%s'
              % (a['id'], thru is not None, thru, shut is None))

    nsight = 0
    for a in aps:
        vin = [v for v in a['vantages'] if v['side'] == 'in'][0]['origin']
        vout = [v for v in a['vantages'] if v['side'] == 'out'][0]['origin']
        hit = open_world.trace(tuple(vin), tuple(vout))
        blocked = seal_world.trace(tuple(vin), tuple(vout))
        nsight += (hit is None and blocked is not None)
        print('  aperture %d  sightline clear unplugged=%s  blocked when plugged=%s'
              % (a['id'], hit is None, blocked is not None))

    bad = []
    if frac < 0.90: bad.append('navigable centerline %.1f%% < 90%%' % (100 * frac))
    if aps and nopen != len(aps): bad.append('%d/%d mouths not free' % (nopen, len(aps)))
    if aps and nseal != len(aps): bad.append('%d/%d mouths not sealed by plug' % (nseal, len(aps)))
    if aps and nsight != len(aps): bad.append('%d/%d sightlines wrong' % (nsight, len(aps)))
    if aps and npass != len(aps): bad.append('%d/%d bodies cannot pass' % (npass, len(aps)))
    print(('FAIL: ' + '; '.join(bad)) if bad else 'PASS')
    return 1 if bad else 0


if __name__ == '__main__':
    p = SG.build_parser() if hasattr(SG, 'build_parser') else None
    sys.exit(run(SG.parse_args(sys.argv[1:]) if p is None else p.parse_args()))
