#!/usr/bin/env mesh-python
import json
import os
import re
import struct
import sys

import numpy as np

import negspace as NS
from negspace import box_H, bounds_of, subtract

def uncovered_volume(ns, H):
    lo, hi = bounds_of(H, ns.world_lo - 8192.0, ns.world_hi + 8192.0)
    if np.any(hi - lo <= 0.5):
        return 0.0
    pieces = [H]
    for ci in ns._cells_in_box(lo, hi):
        pieces = subtract(pieces, ns.cells[ci], lo, hi,
                          exact_empty=False, minext=1.0)
        if not pieces:
            break
    v = 0.0
    for G in pieces:

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

    npz = os.path.join(outdir, 'fused.negspace.npz')
    if os.path.exists(npz):
        ns = NS.load_saved(npz)
        if ns.schema != NS.NEGSPACE_SCHEMA:
            ns = NS.NegSpace(os.path.join(outdir, 'fused.bsp'))
    else:
        ns = NS.NegSpace(os.path.join(outdir, 'fused.bsp'))
    J = json.load(open(os.path.join(outdir, 'fused.joins.json')))
    print('fusion geometry measures: %s' % outdir)
    print('free volume: %d convex cells, world %s .. %s'
          % (len(ns.cells), [int(x) for x in ns.world_lo], [int(x) for x in ns.world_hi]))

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

    fully_covered = 0
    uncovered = []
    for p in J.get('portals', []):
        alo, ahi = p['aperture']
        ax, sg, mth = p['axis'], p['sgn'], p['mouth']
        inner = list(mth)
        inner[ax] = (alo[ax] + ahi[ax]) / 2.0
        back = list(inner)
        back[ax] = (alo[ax] if sg > 0 else ahi[ax]) - sg * 96.0
        run = [back, inner, mth]
        positive_uncovered_segment_mass = 0
        maximum_uncovered_fraction = 0.0
        for k in range(len(run) - 1):
            a, b = run[k], run[k + 1]
            lo2 = [min(a[i], b[i]) - (16.0 if i < 2 else 0.0) for i in range(3)]
            hi2 = [max(a[i], b[i]) + (16.0 if i < 2 else 0.0) for i in range(3)]
            lo2[2] = alo[2] + 2.0
            hi2[2] = min(ahi[2] - 2.0, alo[2] + 2.0 + 69.0)
            if hi2[2] - lo2[2] < 60.0:
                positive_uncovered_segment_mass += 1
                maximum_uncovered_fraction = 1.0
                break
            v = uncovered_volume(ns, box_H(lo2, hi2))
            tot = float(np.prod(np.maximum(np.array(hi2) - np.array(lo2), 0.0)))
            fraction = 0.0 if tot <= 0 else v / tot
            maximum_uncovered_fraction = max(maximum_uncovered_fraction, fraction)
            positive_uncovered_segment_mass += v > 0
        if positive_uncovered_segment_mass == 0:
            fully_covered += 1
        else:
            uncovered.append((p['name'], positive_uncovered_segment_mass,
                              maximum_uncovered_fraction))
    print('doorways: %d/%d fully covered player-body sweeps; %d positive-uncovered '
          'portal rows%s'
          % (fully_covered, len(J.get('portals', [])), len(uncovered),
             '' if not uncovered else '; largest fractions: ' + ', '.join(
                 '%s %.1f%%' % (name, 100 * fraction)
                 for name, _, fraction in sorted(uncovered, key=lambda row: -row[2])[:6])))

    clearance_gap_mass = 0
    support_gap_mass = 0
    rows = []
    for jn in J.get('joins', []):
        if jn['kind'] != 'corridor':
            continue
        clearance = support = 0
        for left, right in zip(jn['chain'], jn['chain'][1:]):
            clearance += len(ns.segment_gaps(
                left, right, NS.CART_RIDER_MIN, NS.CART_RIDER_MAX,
            ))
            support += len(ns.support_gaps(
                left, right, NS.CART_RIDER_MIN, NS.CART_RIDER_MAX,
            ))
        clearance_gap_mass += clearance
        support_gap_mass += support
        rows.append((jn['length'], clearance, support))
    print('connectors: %d corridors | swept-hull clearance gaps %d | floor-support gaps %d'
          % (len(rows), clearance_gap_mass, support_gap_mass))
    for length, clearance, support in rows:
        if clearance or support:
            print('   len=%-6.0f clearance_gaps=%d support_gaps=%d'
                  % (length, clearance, support))

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
    ncart = nnode = unrepresented_node_mass = unrepresented_segment_mass = 0
    for key, pts in sorted(carts.items()):
        pts = [p for _, p in sorted(pts)]
        if len(pts) < 2:
            continue
        ncart += 1
        nnode += len(pts)
        for p in pts:
            if not ns.fits(p, NS.CART_MIN, NS.CART_MAX):
                unrepresented_node_mass += 1
        for i in range(len(pts) - 1):
            if ns.segment_gaps(pts[i], pts[i + 1]):
                unrepresented_segment_mass += 1
    print('cart track: %d tracks, %d nodes | nodes with no legal cart placement: %d | '
          'segments leaving free volume: %d' % (ncart, nnode, unrepresented_node_mass,
                                                 unrepresented_segment_mass))

if __name__ == '__main__':
    main(sys.argv[1])
