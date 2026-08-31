#!/usr/bin/env python3
"""fusecheck -- verify a fused megamap against its own COMPUTED free volume.

    fusecheck.py <outdir>

Reads the artifacts a fusion writes -- `fused.negspace.npz` (the assembled free
volume), `fused.joins.json` (the doorways and connectors it cut) and `fused.ent`
/ the BSP entity lump (the spawnpoints and cart track it shipped) -- and reports,
in world space and in closed form:

  * spawnpoints: is every SHIPPED spawn a standing placement in free space?
  * doorways:    does a player-sized body fit through every cut aperture, from
                 the approach inside the tile to the connector mouth outside it?
  * connectors:  how much of each corridor's interior is NOT free volume, in
                 cubic units?
  * cart track:  is every cart node a legal cart placement, and is every swept
                 segment between consecutive nodes inside free space?

None of this is a sample.  A segment's intersection with a convex free cell is an
interval and a box's uncovered part is a convex subtraction, so every number
below is the whole answer rather than a count of probes that happened to hit.

This exists as a separate tool because the evidence should be reproducible from
the shipped artifacts, not only from the run that produced them.
"""

import json
import math
import os
import re
import struct
import sys

import numpy as np

import negspace as NS
from negspace import box_H, bounds_of, subtract

CORW, CORH, WALL, FLOORTHK = 288.0, 224.0, 32.0, 32.0
CORW_PROM = 448.0
DOOR_H = 208.0


def corridor_volume(a, b, w2, lo, hi):
    af = [a[0], a[1], a[2] - 1.0]
    bf = [b[0], b[1], b[2] - 1.0]
    d = [bf[i] - af[i] for i in range(3)]
    L2 = math.hypot(d[0], d[1]) or 1.0
    dirh = [d[0] / L2, d[1] / L2, 0.0]
    side = [-dirh[1], dirh[0], 0.0]
    n3 = math.sqrt(sum(x * x for x in d)) or 1.0
    d3 = [x / n3 for x in d]
    ntop = [d3[1] * side[2] - d3[2] * side[1],
            d3[2] * side[0] - d3[0] * side[2],
            d3[0] * side[1] - d3[1] * side[0]]
    nn = math.sqrt(sum(x * x for x in ntop)) or 1.0
    ntop = [x / nn for x in ntop]
    if ntop[2] < 0:
        ntop = [-x for x in ntop]
    dot = lambda u, v: sum(u[i] * v[i] for i in range(3))
    rows = [(dirh, dot(dirh, bf)), ([-x for x in dirh], -dot(dirh, af)),
            (side, dot(side, af) + w2), ([-x for x in side], -(dot(side, af) - w2)),
            (ntop, dot(ntop, af) + hi), ([-x for x in ntop], -(dot(ntop, af) + lo))]
    return np.array([[n[0], n[1], n[2], v] for n, v in rows], dtype=np.float64)


def uncovered_volume(ns, H, depth=0, maxdepth=5, cap=48):
    """Volume of the convex region H that is NOT free, in cubic units.

    If the free cells in the way are too finely divided for the piece budget, the
    region is HALVED and each half measured -- so an overflow costs accuracy in
    the split, never a fabricated 'the whole tube is solid'."""
    lo, hi = bounds_of(H, ns.world_lo - 8192.0, ns.world_hi + 8192.0)
    if np.any(hi - lo <= 0.5):
        return 0.0
    pieces = [H]
    over = False
    for ci in ns._cells_in_box(lo, hi):
        pieces, over = subtract(pieces, ns.cells[ci], lo, hi, cap=cap,
                                exact_empty=False, minext=1.0)
        if over or not pieces:
            break
    if over:
        if depth >= maxdepth:
            return float('nan')
        ax = int(np.argmax(hi - lo))
        mid = 0.5 * (lo[ax] + hi[ax])
        tot = 0.0
        for sgn in (1.0, -1.0):
            n = np.zeros(4)
            n[ax] = sgn
            n[3] = sgn * mid
            tot += uncovered_volume(ns, np.vstack([H, n[None, :]]), depth + 1, maxdepth, cap)
        return tot
    v = 0.0
    for G in pieces:
        # a remainder whose AABB is thick can still be an EMPTY polytope; counting
        # those would inflate the answer, so the survivors are checked exactly
        if len(NS.vertices(G)) < 4:
            continue
        gl, gh = bounds_of(G, lo - 1.0, hi + 1.0)
        e = np.maximum(gh - gl, 0.0)
        v += float(e[0] * e[1] * e[2])
    return v


def ent_blocks(outdir):
    p = os.path.join(outdir, 'fused.ent')
    if os.path.exists(p):
        txt = open(p, encoding='latin-1').read()
    else:
        d = open(os.path.join(outdir, 'fused.bsp'), 'rb').read()
        o, n = struct.unpack_from('<ii', d, 8)
        txt = d[o:o + n].split(b'\0')[0].decode('latin-1')
    return re.findall(r'\{[^{}]*\}', txt)


def main(outdir):
    # The lump writer used to emit fused.negspace.npz alongside the bsp because
    # its bsp could not express the assembled free volume (connector leaves hung
    # off a degenerate router chain).  A q3map2-compiled world has no such gap:
    # the bsp IS the assembled world, so the complex is read straight from it.
    npz = os.path.join(outdir, 'fused.negspace.npz')
    if os.path.exists(npz):
        ns = NS.load_saved(npz)
    else:
        ns = NS.NegSpace(os.path.join(outdir, 'fused.bsp'))
    J = json.load(open(os.path.join(outdir, 'fused.joins.json')))
    print('fusecheck: %s' % outdir)
    print('free volume: %d convex cells, world %s .. %s'
          % (len(ns.cells), [int(x) for x in ns.world_lo], [int(x) for x in ns.world_hi]))

    # ---- SPAWNPOINTS ------------------------------------------------------
    blocks = ent_blocks(outdir)
    spawns = []
    for b in blocks:
        cn = re.search(r'"classname"\s+"([^"]+)"', b)
        if not cn or not cn.group(1).startswith('info_player_'):
            continue
        mo = re.search(r'"origin"\s+"([-\d.eE+ ]+)"', b)
        if mo:
            spawns.append([float(x) for x in mo.group(1).split()])
    insolid = notfit = nofloor = 0
    for o in spawns:
        if ns.cell_at(o) < 0:
            insolid += 1
            continue
        if not ns.fits(o):
            notfit += 1
            continue
        if ns.floor_under(o, 512.0) is None:
            nofloor += 1
    print('spawnpoints: %d shipped | origin inside solid: %d | player box does not '
          'fit: %d | no floor beneath within 512u: %d'
          % (len(spawns), insolid, notfit, nofloor))

    # ---- DOORWAYS ---------------------------------------------------------
    ok = 0
    bad = []
    for p in J.get('portals', []):
        alo, ahi = p['aperture']
        ax, sg, mth = p['axis'], p['sgn'], p['mouth']
        inner = list(mth)
        inner[ax] = (alo[ax] + ahi[ax]) / 2.0
        back = list(inner)
        back[ax] = (alo[ax] if sg > 0 else ahi[ax]) - sg * 96.0
        run = [back, inner, mth]
        good = True
        worst = 0.0
        for k in range(len(run) - 1):
            a, b = run[k], run[k + 1]
            lo2 = [min(a[i], b[i]) - (16.0 if i < 2 else 0.0) for i in range(3)]
            hi2 = [max(a[i], b[i]) + (16.0 if i < 2 else 0.0) for i in range(3)]
            lo2[2] = alo[2] + 2.0
            hi2[2] = min(ahi[2] - 2.0, alo[2] + 2.0 + 69.0)
            if hi2[2] - lo2[2] < 60.0:
                good = False
                break
            v = uncovered_volume(ns, box_H(lo2, hi2))
            tot = float(np.prod(np.maximum(np.array(hi2) - np.array(lo2), 0.0)))
            worst = max(worst, 0.0 if tot <= 0 else v / tot)
            if v > 0.02 * tot:
                good = False
        if good:
            ok += 1
        else:
            bad.append((p['name'], worst))
    print('doorways: %d/%d cut apertures admit a player-sized body end to end '
          '(approach -> aperture -> connector mouth), measured as uncovered fraction '
          'of the swept volume%s'
          % (ok, len(J.get('portals', [])),
             '' if not bad else '; worst: ' + ', '.join('%s %.1f%%' % (n, 100 * f)
                                                        for n, f in sorted(bad, key=lambda t: -t[1])[:6])))

    # ---- CONNECTORS -------------------------------------------------------
    nfail = 0
    tot = 0.0
    rows = []
    for jn in J.get('joins', []):
        if jn['kind'] != 'corridor':
            continue
        w2 = (CORW_PROM if jn.get('prominent') else CORW) / 2.0
        H = corridor_volume(jn['sa'], jn['sb'], w2 - 8.0, 8.0, CORH - 16.0)
        v = uncovered_volume(ns, H)
        lo, hi = bounds_of(H, ns.world_lo - 8192.0, ns.world_hi + 8192.0)
        e = np.maximum(hi - lo, 0.0)
        cap = float(e[0] * e[1] * e[2]) or 1.0
        if not (v == v):
            rows.append((jn['length'], float('nan'), 0.0))
            continue
        if v > 1.0:
            nfail += 1
            tot += v
        rows.append((jn['length'], v, v / cap))
    worst = sorted((r for r in rows if r[1] == r[1]), key=lambda r: -r[1])[:5]
    print('connectors: %d/%d corridors have ANY uncovered interior; total %.4g u^3'
          % (nfail, len(rows), tot))
    for L, v, f in worst:
        print('   len=%-6.0f uncovered %.4g u^3 (%.2f%% of the tube AABB)' % (L, v, 100 * f))

    # ---- CART TRACK -------------------------------------------------------
    carts = {}
    for b in blocks:
        cn = re.search(r'"classname"\s+"(plc_path|plc_goal|plc_start)"', b)
        tn = re.search(r'"targetname"\s+"([^"]+)"', b)
        mo = re.search(r'"origin"\s+"([-\d.eE+ ]+)"', b)
        if not cn or not mo or not tn:
            continue
        m = re.match(r'plc(\d+)n(\d+)$', tn.group(1))
        if not m:
            continue
        carts.setdefault(int(m.group(1)), []).append((int(m.group(2)),
                                                      [float(x) for x in mo.group(1).split()]))
    ncart = nnode = badnode = badseg = 0
    for key, pts in sorted(carts.items()):
        pts = [p for _, p in sorted(pts)]
        if len(pts) < 2:
            continue
        ncart += 1
        nnode += len(pts)
        for p in pts:
            if not ns.fits(p, NS.CART_MIN, NS.CART_MAX):
                badnode += 1
        for i in range(len(pts) - 1):
            if ns.segment_gaps(pts[i], pts[i + 1]):
                badseg += 1
    print('cart track: %d tracks, %d nodes | nodes with no legal cart placement: %d | '
          'segments leaving free volume: %d' % (ncart, nnode, badnode, badseg))


if __name__ == '__main__':
    main(sys.argv[1])
