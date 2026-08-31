#!/usr/bin/env python3
"""fusegraph.py -- connectivity solvers, navmesh metrics and trivial visualizers
for a fused megamap.

Offline stats of the form "3 maps, 3 joins, flood-fill OK" were not sufficient
evidence that a fusion works.  This tool answers the harder questions, over the
REAL artifacts (fused.joins.json + fused.waypoints.cache + fused.bsp), and draws
them:

  SOLVERS (region graph: tiles = nodes, joins = edges)
    * connected components
    * articulation points  -> chokepoint TILES
    * cut edges            -> chokepoint JOINS (lose one and the megamap splits)
    * 2-edge-connected blocks, hop diameter, degree distribution

  SOLVERS (navmesh: the real fused bot-waypoint graph)
    * weighted (euclidean) Dijkstra between region representatives
    * per-region bot-reachable coverage, region<->region walking distance matrix,
      the megamap WALKING DIAMETER (the commitment cost a strategy has to pay)
    * join BETWEENNESS: the fraction of region-pair shortest bot paths that use
      each join -- the quantitative version of "chokepoint"
    * detour ratio per join: bot walking distance vs straight-line distance

  VIEWERS (SVG, no dependencies)
    * fused.graph.svg    -- the region graph: tiles sized by waypoint count, joins
      coloured by kind, thickened by betweenness, dashed when subtle, red when a
      cut edge; bridge tiles drawn as diamonds
    * fused.navmesh.svg  -- the real fused waypoint graph in plan view, coloured by
      region, with the join sockets and the megamap diameter path drawn on top

Usage:  fusegraph.py <fused_map_dir> [--out DIR]
"""
import os, sys, json, math, heapq, argparse
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M


# --------------------------------------------------------------------------- io
def load(mapdir):
    J = json.load(open(os.path.join(mapdir, 'fused.joins.json')))
    cache = os.path.join(mapdir, 'fused.waypoints.cache')
    nodes, adj = M.parse_cache(open(cache, encoding='latin-1').read())
    dadj = [set(a.keys()) for a in adj]
    key = lambda p: tuple(round(x, 1) for x in p)
    idx = {key(nodes[i]): i for i in range(len(nodes))}
    for near, far in J.get('bot_jumps', []):
        u, v = idx.get(key(near)), idx.get(key(far))
        if u is not None and v is not None:
            dadj[u].add(v)
    return J, nodes, dadj, idx, key


def region_of(J, nodes):
    """Assign every fused waypoint to the tile whose world AABB contains it; points
    in a connector fall to the nearest tile centre."""
    boxes = [(m['mins'], m['maxs']) for m in J['maps']]
    cent = [[(m['mins'][a] + m['maxs'][a]) / 2 for a in range(3)] for m in J['maps']]
    reg = []
    for p in nodes:
        r = None
        for i, (lo, hi) in enumerate(boxes):
            if all(lo[a] - 1 <= p[a] <= hi[a] + 1 for a in range(3)):
                r = i
                break
        if r is None:
            r = min(range(len(cent)), key=lambda i: math.dist(p, cent[i]))
        reg.append(r)
    return reg


# ---------------------------------------------------------------- region solver
def region_solve(n, edges):
    adj = [[] for _ in range(n)]
    for ei, (a, b) in enumerate(edges):
        adj[a].append((b, ei))
        adj[b].append((a, ei))
    disc, low, arts, cuts, comps = [-1] * n, [0] * n, set(), [], []
    t = 0
    for s0 in range(n):
        if disc[s0] != -1:
            continue
        comp, rootkids = [s0], 0
        disc[s0] = low[s0] = t
        t += 1
        st = [(s0, iter(adj[s0]), -1)]
        while st:
            u, it, pe = st[-1]
            adv = False
            for v, ei in it:
                if ei == pe:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = t
                    t += 1
                    comp.append(v)
                    if u == s0:
                        rootkids += 1
                    st.append((v, iter(adj[v]), ei))
                    adv = True
                    break
                low[u] = min(low[u], disc[v])
            if adv:
                continue
            st.pop()
            if st:
                pu = st[-1][0]
                low[pu] = min(low[pu], low[u])
                if low[u] > disc[pu]:
                    cuts.append(pe)
                if pu != s0 and low[u] >= disc[pu]:
                    arts.add(pu)
        if rootkids > 1:
            arts.add(s0)
        comps.append(sorted(comp))
    big = max(comps, key=len) if comps else []
    bs = set(big)
    diam = 0
    for s0 in big:
        d = {s0: 0}
        q = deque([s0])
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if v not in d and v in bs:
                    d[v] = d[u] + 1
                    q.append(v)
        diam = max(diam, max(d.values()))
    # 2-edge-connected blocks: components after deleting the cut edges
    cutset = set(cuts)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for ei, (a, b) in enumerate(edges):
        if ei in cutset:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    blocks = {}
    for i in range(n):
        blocks.setdefault(find(i), []).append(i)
    return dict(adj=adj, components=comps, articulation=sorted(arts), cutedges=sorted(cuts),
                degree=[len(adj[i]) for i in range(n)], hop_diameter=diam,
                blocks=sorted(blocks.values(), key=len, reverse=True))


# --------------------------------------------------------------- navmesh solver
def dijkstra(W, src):
    dist = [float('inf')] * len(W)
    prev = [-1] * len(W)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] + 1e-9:
            continue
        for v, w in W[u]:
            nd = d + w
            if nd < dist[v] - 1e-9:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def navmesh_metrics(J, nodes, dadj, idx, key, reg):
    W = [[(v, math.dist(nodes[u], nodes[v])) for v in dadj[u]] for u in range(len(nodes))]
    n = len(J['maps'])
    # a representative node per region: the reachable node nearest the region centroid
    reps = {}
    for m in range(n):
        pts = [i for i in range(len(nodes)) if reg[i] == m]
        if not pts:
            reps[m] = None
            continue
        cx = sum(nodes[i][0] for i in pts) / len(pts)
        cy = sum(nodes[i][1] for i in pts) / len(pts)
        reps[m] = min(pts, key=lambda i: math.hypot(nodes[i][0] - cx, nodes[i][1] - cy))
    # socket node ids per join, for betweenness
    sock = []
    for jn in J['joins']:
        sock.append((idx.get(key(jn['sa'])), idx.get(key(jn['sb']))))
    D, P, cover = {}, {}, {}
    for m, r in reps.items():
        if r is None:
            continue
        dist, prev = dijkstra(W, r)
        cover[m] = sum(1 for i in range(len(nodes)) if dist[i] < float('inf') and reg[i] == m)
        P[m] = prev
        for m2, r2 in reps.items():
            if r2 is None or m2 == m:
                continue
            D[(m, m2)] = dist[r2] if dist[r2] < float('inf') else None
    # betweenness of each join over region-pair shortest paths
    use = [0] * len(J['joins'])
    pairs = 0
    for (m, m2), d in D.items():
        if d is None or m > m2:
            continue
        pairs += 1
        prev = P[m]
        cur = reps[m2]
        path = set()
        while cur != -1:
            path.add(cur)
            cur = prev[cur]
        for ji, (sa, sb) in enumerate(sock):
            if sa in path and sb in path:
                use[ji] += 1
    fin = [v for v in D.values() if v is not None]
    total = {m: sum(1 for i in range(len(nodes)) if reg[i] == m) for m in range(n)}
    return dict(reps=reps, walk=D, prevs=P, coverage=cover, region_nodes=total,
                betweenness=[u / max(1, pairs) for u in use], pairs=pairs,
                walk_diameter=max(fin) if fin else 0.0,
                walk_median=sorted(fin)[len(fin) // 2] if fin else 0.0,
                unreachable=sum(1 for v in D.values() if v is None), W=W)


# --------------------------------------------------------------------- viewers
def col(i):
    p = ['#4f8fd6', '#e0803a', '#5aa469', '#c05a6e', '#8a6fbf', '#9c7b5a',
         '#cf74b4', '#7f7f7f', '#b7b24a', '#4bb0c0']
    return p[i % len(p)]


def svg_graph(path, J, RG, NM):
    maps, joins = J['maps'], J['joins']
    xs = [(m['mins'][0] + m['maxs'][0]) / 2 for m in maps]
    ys = [(m['mins'][1] + m['maxs'][1]) / 2 for m in maps]
    W, H, PADP = 1400, 1000, 90
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = (W - 2 * PADP) / max(1.0, x1 - x0)
    sy = (H - 2 * PADP) / max(1.0, y1 - y0)
    sc = min(sx, sy)
    px = lambda x: PADP + (x - x0) * sc
    py = lambda y: H - PADP - (y - y0) * sc
    cut = set(RG['cutedges'])
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
           'viewBox="0 0 %d %d" font-family="monospace">' % (W, H, W, H),
           '<rect x="0" y="0" width="%d" height="%d" fill="#12151a"/>' % (W, H)]
    kindcol = {'corridor': '#5aa469', 'teleporter': '#8a6fbf', 'jumppad': '#e0803a'}
    for ji, jn in enumerate(joins):
        a, b = jn['a'], jn['b']
        bw = 1.5 + 9.0 * NM['betweenness'][ji]
        c = kindcol.get(jn['kind'], '#888')
        dash = '' if jn.get('prominent') else ' stroke-dasharray="7,5"'
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                   'stroke-width="%.1f" opacity="0.9"%s/>' %
                   (px(xs[a]), py(ys[a]), px(xs[b]), py(ys[b]), c, bw, dash))
        if ji in cut:
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e05a5a" '
                       'stroke-width="%.1f" opacity="0.45"/>' %
                       (px(xs[a]), py(ys[a]), px(xs[b]), py(ys[b]), bw + 6))
        mx, my = (px(xs[a]) + px(xs[b])) / 2, (py(ys[a]) + py(ys[b])) / 2
        out.append('<text x="%.1f" y="%.1f" fill="#cfd6e0" font-size="10" '
                   'text-anchor="middle">%s %du b=%.2f</text>' %
                   (mx, my - 4, jn['kind'][:4], int(jn['length']), NM['betweenness'][ji]))
    arts = set(RG['articulation'])
    for i, m in enumerate(maps):
        n = NM['region_nodes'].get(i, 0)
        r = 9 + min(26, 0.9 * math.sqrt(max(1, n)) * 2)
        fill = col(i)
        X, Y = px(xs[i]), py(ys[i])
        if m.get('bridge'):
            out.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f" '
                       'fill="%s" stroke="%s" stroke-width="3"/>' %
                       (X, Y - r, X + r, Y, X, Y + r, X - r, Y,
                        fill, '#ffcf5a' if i in arts else '#0d0f13'))
        else:
            out.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="%s" '
                       'stroke-width="3"/>' % (X, Y, r, fill,
                                               '#ffcf5a' if i in arts else '#0d0f13'))
        out.append('<text x="%.1f" y="%.1f" fill="#eef2f7" font-size="12" '
                   'text-anchor="middle">%s</text>' % (X, Y + r + 14, m['name']))
        out.append('<text x="%.1f" y="%.1f" fill="#9fb0c4" font-size="10" '
                   'text-anchor="middle">deg %d / %d wp</text>' %
                   (X, Y + r + 26, m.get('degree', RG['degree'][i]), n))
    L = ['tiles %d (%d bridge)  joins %d  components %d  hop-diameter %d' %
         (len(maps), sum(1 for m in maps if m.get('bridge')), len(joins),
          len(RG['components']), RG['hop_diameter']),
         'chokepoint tiles (articulation): %d   chokepoint joins (cut edges): %d' %
         (len(RG['articulation']), len(RG['cutedges'])),
         '2-edge-connected blocks: %s' % [len(b) for b in RG['blocks'][:8]],
         'navmesh walking: median %.0fu  DIAMETER %.0fu  unreachable pairs %d' %
         (NM['walk_median'], NM['walk_diameter'], NM['unreachable']),
         'edge width = betweenness (share of region-pair bot paths); dashed = subtle/'
         'redundant; red halo = cut edge; diamond = procedural bridge tile']
    for k, t in enumerate(L):
        out.append('<text x="18" y="%d" fill="#cfd6e0" font-size="14">%s</text>' %
                   (24 + 18 * k, t))
    out.append('</svg>')
    open(path, 'w').write('\n'.join(out))
    return path


def svg_navmesh(path, J, nodes, dadj, reg, NM):
    W, H, PADP = 1500, 1100, 40
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sc = min((W - 2 * PADP) / max(1.0, x1 - x0), (H - 2 * PADP) / max(1.0, y1 - y0))
    px = lambda x: PADP + (x - x0) * sc
    py = lambda y: H - PADP - (y - y0) * sc
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
           'viewBox="0 0 %d %d" font-family="monospace">' % (W, H, W, H),
           '<rect x="0" y="0" width="%d" height="%d" fill="#0d1014"/>' % (W, H)]
    for m in J['maps']:
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="none" '
                   'stroke="#2b323c" stroke-width="1"/>' %
                   (px(m['mins'][0]), py(m['maxs'][1]),
                    (m['maxs'][0] - m['mins'][0]) * sc, (m['maxs'][1] - m['mins'][1]) * sc))
    seen = set()
    for u in range(len(nodes)):
        for v in dadj[u]:
            if (v, u) in seen:
                continue
            seen.add((u, v))
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                       'stroke-width="0.6" opacity="0.55"/>' %
                       (px(nodes[u][0]), py(nodes[u][1]), px(nodes[v][0]), py(nodes[v][1]),
                        col(reg[u])))
    # the megamap diameter path, drawn on top
    best = None
    for (m, m2), d in NM['walk'].items():
        if d is not None and (best is None or d > best[0]):
            best = (d, m, m2)
    if best:
        d, m, m2 = best
        prev = NM['prevs'][m]
        cur = NM['reps'][m2]
        pts = []
        while cur != -1:
            pts.append('%.1f,%.1f' % (px(nodes[cur][0]), py(nodes[cur][1])))
            cur = prev[cur]
        out.append('<polyline points="%s" fill="none" stroke="#ffd24a" stroke-width="3" '
                   'opacity="0.95"/>' % ' '.join(pts))
        out.append('<text x="18" y="44" fill="#ffd24a" font-size="15">walking DIAMETER '
                   '%s -> %s = %.0fu (%d hops of waypoint graph)</text>' %
                   (J['maps'][m]['name'], J['maps'][m2]['name'], d, len(pts)))
    for jn in J['joins']:
        for s in (jn['sa'], jn['sb']):
            out.append('<circle cx="%.1f" cy="%.1f" r="4" fill="none" stroke="#ff6b6b" '
                       'stroke-width="1.6"/>' % (px(s[0]), py(s[1])))
    out.append('<text x="18" y="24" fill="#cfd6e0" font-size="15">fused navmesh: %d '
               'waypoints, %d regions, join sockets ringed red</text>' %
               (len(nodes), len(J['maps'])))
    out.append('</svg>')
    open(path, 'w').write('\n'.join(out))
    return path


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mapdir')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    d = os.path.abspath(args.mapdir)
    out = args.out or d
    os.makedirs(out, exist_ok=True)
    J, nodes, dadj, idx, key = load(d)
    reg = region_of(J, nodes)
    names = [m['name'] for m in J['maps']]
    RG = region_solve(len(J['maps']), [(j['a'], j['b']) for j in J['joins']])
    NM = navmesh_metrics(J, nodes, dadj, idx, key, reg)

    print('== region graph ==')
    print('tiles %d (%d procedural bridge)  joins %d' %
          (len(names), sum(1 for m in J['maps'] if m.get('bridge')), len(J['joins'])))
    print('components %s  hop-diameter %d' % ([len(c) for c in RG['components']],
                                              RG['hop_diameter']))
    print('chokepoint TILES  (articulation): %s' % [names[i] for i in RG['articulation']])
    print('chokepoint JOINS  (cut edges)   : %s' %
          ['%s<->%s' % (names[J['joins'][e]['a']], names[J['joins'][e]['b']])
           for e in RG['cutedges']])
    print('2-edge-connected blocks (sizes) : %s' % [len(b) for b in RG['blocks']])
    print('degrees: %s' % dict(zip(names, RG['degree'])))
    print('== navmesh ==')
    print('waypoints %d  region-pairs solved %d  unreachable pairs %d' %
          (len(nodes), NM['pairs'], NM['unreachable']))
    print('walking distance median %.0fu  DIAMETER %.0fu' %
          (NM['walk_median'], NM['walk_diameter']))
    print('per-region waypoints / reachable-from-own-rep:')
    for i, n in enumerate(names):
        print('  %-16s %5d wp   reachable %5d' %
              (n, NM['region_nodes'].get(i, 0), NM['coverage'].get(i, 0)))
    print('== joins ==')
    for ji, jn in enumerate(J['joins']):
        print('  %-14s<->%-14s %-11s len=%6.0f cart=%-5s prominent=%-5s cut=%-5s '
              'betweenness=%.2f' %
              (names[jn['a']], names[jn['b']], jn['kind'], jn['length'],
               jn.get('cart_navigable', jn['kind'] == 'corridor'), jn.get('prominent'),
               ji in set(RG['cutedges']), NM['betweenness'][ji]))
    g1 = svg_graph(os.path.join(out, 'fused.graph.svg'), J, RG, NM)
    g2 = svg_navmesh(os.path.join(out, 'fused.navmesh.svg'), J, nodes, dadj, reg, NM)
    rep = {'names': names,
           'region_graph': {k: v for k, v in RG.items() if k != 'adj'},
           'navmesh': {'n_nodes': len(nodes), 'walk_median': NM['walk_median'],
                       'walk_diameter': NM['walk_diameter'],
                       'unreachable_pairs': NM['unreachable'],
                       'coverage': NM['coverage'], 'region_nodes': NM['region_nodes'],
                       'walk': {'%d-%d' % k: v for k, v in NM['walk'].items()}},
           'joins': [{'a': names[j['a']], 'b': names[j['b']], 'kind': j['kind'],
                      'length': j['length'], 'prominent': j.get('prominent'),
                      'cart_navigable': j.get('cart_navigable', j['kind'] == 'corridor'),
                      'cut_edge': ji in set(RG['cutedges']),
                      'betweenness': NM['betweenness'][ji]}
                     for ji, j in enumerate(J['joins'])]}
    rp = os.path.join(out, 'fused.connectivity.json')
    json.dump(rep, open(rp, 'w'), indent=1)
    print('wrote %s' % g1)
    print('wrote %s' % g2)
    print('wrote %s' % rp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
