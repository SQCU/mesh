import struct, sys, os, math, json, heapq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M

PLAYERCLIP, BOTCLIP, MONSTERCLIP = 0x10000, 0x400000, 0x20000


def clip_brushes(d):
    L = lambda i: struct.unpack_from('<ii', d, 8 + i * 8)
    to, tl = L(1)
    po, pl = L(2)
    bo, bl = L(8)
    so, sl = L(9)
    ntex = tl // 72
    clipflag = []
    for i in range(ntex):
        ct = struct.unpack_from('<i', d, to + i * 72 + 68)[0]
        clipflag.append(bool(ct & (PLAYERCLIP | BOTCLIP | MONSTERCLIP)))
    planes = [struct.unpack_from('<4f', d, po + i * 16) for i in range(pl // 16)]
    sides = [struct.unpack_from('<2i', d, so + i * 8) for i in range(sl // 8)]
    boxes = []
    for i in range(bl // 12):
        fs, ns, tx = struct.unpack_from('<3i', d, bo + i * 12)
        if tx < 0 or tx >= ntex or not clipflag[tx]:
            continue
        lo, hi = [-1e18] * 3, [1e18] * 3
        for k in range(fs, fs + ns):
            nx, ny, nz, dd = planes[sides[k][0]]
            for a, c in enumerate((nx, ny, nz)):
                if c > 0.999:
                    hi[a] = min(hi[a], dd)
                elif c < -0.999:
                    lo[a] = max(lo[a], -dd)
        if all(hi[a] - lo[a] < 4096 for a in range(3)):
            boxes.append((lo, hi))
    return boxes


def seg_hits_box(pa, pb, lo, hi):
    t0, t1 = 0.0, 1.0
    for a in range(3):
        dd = pb[a] - pa[a]
        if abs(dd) < 1e-9:
            if pa[a] < lo[a] or pa[a] > hi[a]:
                return False
            continue
        u0, u1 = (lo[a] - pa[a]) / dd, (hi[a] - pa[a]) / dd
        if u0 > u1:
            u0, u1 = u1, u0
        t0, t1 = max(t0, u0), min(t1, u1)
        if t0 > t1:
            return False
    return True


def dijkstra_nodes(adj, srcs):
    D = {}
    pq = [(0.0, s) for s in srcs]
    for s in srcs:
        D[s] = 0.0
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > D.get(u, 1e18):
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < D.get(v, 1e18):
                D[v] = nd
                heapq.heappush(pq, (nd, v))
    return D


def nearest_k(nodes, p, k):
    return [i for _, i in sorted((math.dist(nodes[i], p), i) for i in range(len(nodes)))[:k]]


def build_dir_adj(nodes, adj, bot_jumps):
    key = lambda q: tuple(round(x, 1) for x in q)
    idx = {key(nodes[i]): i for i in range(len(nodes))}
    dadj = [dict(a) for a in adj]
    for near, far in bot_jumps:
        u, v = idx.get(key(near)), idx.get(key(far))
        if u is not None and v is not None:
            w = math.dist(nodes[u], nodes[v])
            dadj[u][v] = min(dadj[u].get(v, 1e18), w)
    return dadj, idx


def contortion(nodes, dadj, sa, sb, k=6):
    ea = nearest_k(nodes, sa, k)
    eb = set(nearest_k(nodes, sb, k))
    D = dijkstra_nodes(dadj, ea)
    ratios = []
    for b in eb:
        if b in D and D[b] < 1e17:
            sl = math.dist(sa, sb)
            if sl > 1:
                ratios.append(D[b] / sl)
    if not ratios:
        return None
    return min(ratios), sum(ratios) / len(ratios), max(ratios), len(ratios)


def sightline(ns, clipgrid, sa, sb):
    """Is the join's straight line actually OPEN, and is it clip-blocked?

    DELETED here: `occlusion_probe`, which stepped 24 units at a time along nine
    rays asking `mkentfile.Bsp.inside` -- a point sampler over an
    AABB-from-plane-distance grid, i.e. exactly the class of proxy this toolchain
    no longer contains.  A segment's intersection with a convex free cell is an
    interval, so the OPEN part of each ray is computed in closed form and what is
    reported is the fraction of the line that is genuinely open, in units."""
    mid = [(sa[i] + sb[i]) / 2 for i in range(3)]
    d = [sb[i] - sa[i] for i in range(3)]
    L = math.hypot(d[0], d[1]) or 1.0
    perp = [-d[1] / L, d[0] / L, 0.0]
    occluded = 0
    samples = 0
    for side, base in ((-1, sa), (1, sb)):
        eye = [base[0], base[1], base[2] + 40]
        for lat in (-64, 0, 64):
            tgt = [mid[0] + perp[0] * lat, mid[1] + perp[1] * lat, mid[2] + 40]
            samples += 1
            if ns.segment_gaps(eye, tgt):
                occluded += 1
    clipblocked = 0
    pa = (sa[0], sa[1], sa[2] + 32)
    pb = (sb[0], sb[1], sb[2] + 32)
    for lo, hi in clipgrid:
        if seg_hits_box(pa, pb, lo, hi):
            clipblocked += 1
    return occluded, samples, clipblocked


def svg_floorplan(path, maps, nodes, adj, joins, lights):
    allpts = [m['mins'][:2] for m in maps] + [m['maxs'][:2] for m in maps]
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    x0, x1, y0, y1 = min(xs) - 256, max(xs) + 256, min(ys) - 256, max(ys) + 256
    W = 1400.0
    sc = W / (x1 - x0)
    H = (y1 - y0) * sc
    tx = lambda x: (x - x0) * sc
    ty = lambda y: H - (y - y0) * sc
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" '
           'viewBox="0 0 %.0f %.0f" font-family="sans-serif">' % (W, H, W, H)]
    out.append('<rect width="%.0f" height="%.0f" fill="#12141a"/>' % (W, H))
    palette = ['#3a4a6a', '#4a6a3a', '#6a3a4a', '#6a5a3a', '#3a6a6a']
    for mi, m in enumerate(maps):
        col = palette[mi % len(palette)]
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s22" stroke="%s" stroke-width="2"/>'
                   % (tx(m['mins'][0]), ty(m['maxs'][1]),
                      (m['maxs'][0] - m['mins'][0]) * sc, (m['maxs'][1] - m['mins'][1]) * sc, col, col))
        out.append('<text x="%.1f" y="%.1f" fill="%s" font-size="16">%s</text>'
                   % (tx(m['mins'][0]) + 6, ty(m['maxs'][1]) + 18, col, m['name']))
    seen = set()
    for u in range(len(adj)):
        for v in adj[u]:
            e = (min(u, v), max(u, v))
            if e in seen:
                continue
            seen.add(e)
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#2a3550" stroke-width="0.5"/>'
                       % (tx(nodes[u][0]), ty(nodes[u][1]), tx(nodes[v][0]), ty(nodes[v][1])))
    for lp in lights:
        out.append('<circle cx="%.1f" cy="%.1f" r="4" fill="#ffdd55" opacity="0.8"/>' % (tx(lp[0]), ty(lp[1])))
    for jn in joins:
        sa, sb = jn['sa'], jn['sb']
        col = {'corridor': '#55ff88', 'jumppad': '#ffaa33', 'teleporter': '#ff55cc'}.get(jn['kind'], '#ffffff')
        w = 5 if jn['prominent'] else 2.5
        dash = '' if jn['prominent'] else ' stroke-dasharray="8 6"'
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%.1f"%s/>'
                   % (tx(sa[0]), ty(sa[1]), tx(sb[0]), ty(sb[1]), col, w, dash))
        for s in (sa, sb):
            out.append('<circle cx="%.1f" cy="%.1f" r="6" fill="%s"/>' % (tx(s[0]), ty(s[1]), col))
        mx, my = (tx(sa[0]) + tx(sb[0])) / 2, (ty(sa[1]) + ty(sb[1])) / 2
        tag = '%s %s' % (jn['kind'], 'EXCL' if jn['exclusive'] else 'redund')
        out.append('<text x="%.1f" y="%.1f" fill="%s" font-size="13">%s</text>' % (mx + 6, my, col, tag))
    out.append('<text x="10" y="%.0f" fill="#8892a6" font-size="12">green=corridor orange=jumppad '
               'pink=teleporter; solid+thick=prominent/exclusive, dashed=subtle/redundant; yellow=light</text>'
               % (H - 10))
    out.append('</svg>')
    open(path, 'w').write('\n'.join(out))


def analyze(outdir):
    d = open(os.path.join(outdir, 'fused.bsp'), 'rb').read()
    joins = json.load(open(os.path.join(outdir, 'fused.joins.json')))
    nodes, adj = M.parse_cache(open(os.path.join(outdir, 'fused.waypoints.cache')).read())
    import negspace as NS
    nspath = os.path.join(outdir, 'fused.negspace.npz')
    if os.path.exists(nspath):
        ns = NS.load_saved(nspath)
    else:
        # a single map's own tree does cover its whole world, so this is exact there
        ns = NS.NegSpace(d, mask=NS.MASK_PLAYERSOLID)
    clip = clip_brushes(d)
    dadj, idx = build_dir_adj(nodes, adj, joins.get('bot_jumps', []))
    names = [m['name'] for m in joins['maps']]
    svg = os.path.join(outdir, 'fused.floorplan.svg')
    # collect light origins from entities
    lo, ln = struct.unpack_from('<ii', d, 8)
    ents = d[lo:lo + ln].split(b'\0')[0].decode('latin-1')
    import re
    lights = []
    for b in re.findall(r'\{[^{}]*\}', ents):
        if re.search(r'"classname"\s+"light"', b):
            mo = re.search(r'"origin"\s+"([-\d. ]+)"', b)
            if mo:
                v = [float(x) for x in mo.group(1).split()]
                lights.append(v)
    svg_floorplan(svg, joins['maps'], nodes, adj, joins['joins'], lights)
    print('wrote %s (%d maps, %d nav nodes, %d joins, %d lights, %d clip brushes)'
          % (svg, len(names), len(nodes), len(joins['joins']), len(lights), len(clip)))
    def dest_ok(p):
        # standable in the ASSEMBLED world's computed free volume
        return ns.standing_point([p[0], p[1], p[2] + 24]) is not None
    print('per-edge diagnostics (contortion = walk/straight-line, fuzzed over nearest entry/exit waypoints;')
    print('clip/occlusion along the straight line gate CORRIDORS only -- teleport/pad transport is instant/ballistic,')
    print('so for those the meaningful check is whether both endpoints are clear & standable):')
    for jn in joins['joins']:
        sa, sb = jn['sa'], jn['sb']
        ct = contortion(nodes, dadj, sa, sb)
        occ, nsmp, clipn = sightline(ns, clip, sa, sb)
        ctstr = ('min=%.2f mean=%.2f max=%.2f (n=%d)' % ct) if ct else 'UNREACHABLE'
        tag = ' [EXCL/prominent]' if jn['exclusive'] else ' [redundant]'
        if jn['kind'] == 'corridor':
            phys = 'clip-blocked=%s | sightlines blocked %d/%d (exact free-volume intervals)' % ('YES(%d)' % clipn if clipn else 'no', occ, nsmp)
        else:
            phys = 'endpoints-clear=%s (near=%s far=%s) | straight-line clip/occ n/a (instant transport)' % (
                'YES' if dest_ok(sa) and dest_ok(sb) else 'NO',
                'ok' if dest_ok(sa) else 'BLOCKED', 'ok' if dest_ok(sb) else 'BLOCKED')
        print('  %s<->%s %s%s len=%.0f: contortion %s | %s'
              % (names[jn['a']], names[jn['b']], jn['kind'], tag, jn['length'], ctstr, phys))


if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/fuse_v4/data/maps'
    analyze(outdir)
