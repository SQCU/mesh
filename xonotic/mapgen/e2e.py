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
from oracle import Oracle


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

    # 1. NAVIGABILITY, on the world.
    ok = sum(1 for c in centers if open_world.standable((c[0], c[1], c[2] + 26.0)))
    frac = ok / float(len(centers))
    print('navigable centerline: %d/%d = %.1f%%' % (ok, len(centers), 100 * frac))

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
    print(('FAIL: ' + '; '.join(bad)) if bad else 'PASS')
    return 1 if bad else 0


if __name__ == '__main__':
    p = SG.build_parser() if hasattr(SG, 'build_parser') else None
    sys.exit(run(SG.parse_args(sys.argv[1:]) if p is None else p.parse_args()))
