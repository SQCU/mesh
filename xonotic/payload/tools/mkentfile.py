import struct, sys, re, math, os, glob, subprocess, heapq


def load_cache(mapname, bsp, pk3arg, depth=0):
    cands = []
    if pk3arg:
        cands.append(pk3arg)
    if os.environ.get('XON_MAPS_PK3'):
        cands.append(os.environ['XON_MAPS_PK3'])
    cands += sorted(glob.glob(os.path.expanduser('~/dox/xonotic/Xonotic/data/*maps*.pk3')), reverse=True)
    cands += sorted(glob.glob(os.path.expanduser('~/dox/xonotic/**/*maps*.pk3'), recursive=True), reverse=True)
    text = ''
    for pk3 in cands:
        r = subprocess.run(['unzip', '-p', pk3, 'maps/%s.waypoints.cache' % mapname], capture_output=True)
        if r.returncode == 0 and r.stdout.strip():
            text = r.stdout.decode('latin-1')
            break
    if not text:
        loose = os.path.join(os.path.dirname(bsp) or '.', mapname + '.waypoints.cache')
        if os.path.exists(loose):
            text = open(loose, 'r', encoding='latin-1').read()
    stripped = text.strip()
    if depth < 4 and stripped.endswith('.waypoints.cache') and '*' not in stripped and '\n' not in stripped:
        return load_cache(stripped[:-len('.waypoints.cache')], bsp, pk3arg, depth + 1)
    return text


def parse_cache(text):
    idx, nodes, adj = {}, [], []

    def nid(v):
        k = tuple(round(float(x), 1) for x in v.split())
        if k not in idx:
            idx[k] = len(nodes)
            nodes.append(k)
            adj.append({})
        return idx[k]

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        parts = line.split('*')
        if len(parts) != 2:
            continue
        a = nid(parts[0].strip().strip("'"))
        b = nid(parts[1].strip().strip("'"))
        w = math.dist(nodes[a], nodes[b])
        adj[a][b] = min(adj[a].get(b, 1e18), w)
        adj[b][a] = min(adj[b].get(a, 1e18), w)
    return nodes, adj


def dijkstra(adj, src):
    D = [math.inf] * len(adj)
    prev = [-1] * len(adj)
    D[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > D[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < D[v]:
                D[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return D, prev


def largest_component(adj):
    n = len(adj)
    seen = [False] * n
    best = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    return best


def kcenter(adj, k):
    comp = largest_component(adj)
    d0, _ = dijkstra(adj, comp[0])
    a = max(comp, key=lambda i: d0[i])
    da, _ = dijkstra(adj, a)
    b = max(comp, key=lambda i: da[i])
    chosen = [a, b]
    dmap = {a: da, b: dijkstra(adj, b)[0]}
    while len(chosen) < k:
        best, bestd = comp[0], -1.0
        for i in comp:
            m = min(dmap[c][i] for c in chosen)
            if m > bestd:
                bestd, best = m, i
        chosen.append(best)
        dmap[best] = dijkstra(adj, best)[0]
    return chosen, dmap


def snap5(nodes, pathnodes):
    if len(pathnodes) < 2:
        p = list(nodes[pathnodes[0]])
        return [p[:] for _ in range(5)]
    cum = [0.0]
    for i in range(1, len(pathnodes)):
        cum.append(cum[-1] + math.dist(nodes[pathnodes[i - 1]], nodes[pathnodes[i]]))
    total = cum[-1]
    out = []
    for kk in range(5):
        t = total * kk / 4
        j = min(range(len(cum)), key=lambda x: abs(cum[x] - t))
        out.append(list(nodes[pathnodes[j]]))
    return out


def nav_tracks(nodes, adj, dmap, origins, kcarts):
    used = set()
    tracks = []
    for c in range(kcarts):
        src = origins[c]
        goal, gd = None, -1.0
        for o in origins:
            if o == src or o in used:
                continue
            if dmap[o][src] > gd:
                gd, goal = dmap[o][src], o
        if goal is None:
            goal = max(range(len(nodes)), key=lambda i: dmap[src][i] if dmap[src][i] < math.inf else -1)
        used.add(goal)
        _, prev = dijkstra(adj, src)
        p, u = [], goal
        while u != -1:
            p.append(u)
            u = prev[u]
        p = p[::-1] if len(p) > 1 else [src]
        pts = snap5(nodes, p)
        tracks.append([[q[0], q[1], q[2] + 16] for q in pts])
    return tracks


def spawn_tracks(pts, kcarts):
    spread = [max(p[a] for p in pts) - min(p[a] for p in pts) for a in (0, 1)]
    split_axis = 0 if spread[0] >= spread[1] else 1
    walk_axis = 1 - split_axis
    pts = sorted(pts, key=lambda p: p[split_axis])
    tracks = []
    for c in range(kcarts):
        lo = (len(pts) * c) // kcarts
        hi = (len(pts) * (c + 1)) // kcarts
        part = sorted(pts[lo:hi], key=lambda p: (p[walk_axis], p[split_axis]))
        idx = sorted({0, len(part) // 4, len(part) // 2, (3 * len(part)) // 4, len(part) - 1})
        while len(idx) < 5:
            idx.append(idx[-1])
        tracks.append([[part[i][0], part[i][1], part[i][2] + 16] for i in idx[:5]])
    return tracks


def build_tracks(bsp, mapname, pts, kteams, kcarts, pk3arg=''):
    text = load_cache(mapname, bsp, pk3arg)
    if not text:
        print('nav: no waypoints for %s, FALLBACK to spawn-origin method' % mapname)
        return spawn_tracks(pts, kcarts), None
    nodes, adj = parse_cache(text)
    if not nodes or not largest_component(adj):
        print('nav: degenerate graph for %s (%d nodes), FALLBACK to spawn-origin method' % (mapname, len(nodes)))
        return spawn_tracks(pts, kcarts), None
    cand_k = max(3, kcarts)
    origins, dmap = kcenter(adj, cand_k)
    print('nav: %s waypoints=%d links=%d candidate_origins=%d' %
          (mapname, len(nodes), sum(len(a) for a in adj) // 2, len(origins)))
    ws, es = [], []
    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            wd = dmap[origins[i]][origins[j]]
            ed = math.dist(nodes[origins[i]], nodes[origins[j]])
            ws.append(wd)
            es.append(ed)
            print('nav: origin %d %s <-> origin %d %s  walk=%.0f  euclid=%.0f  ratio=%.2f' %
                  (i, tuple(round(x) for x in nodes[origins[i]]), j,
                   tuple(round(x) for x in nodes[origins[j]]), wd, ed, wd / ed if ed else 0))
    print('nav: pairwise walk min=%.0f mean=%.0f max=%.0f balance_ratio=%.2f' %
          (min(ws), sum(ws) / len(ws), max(ws), max(ws) / min(ws) if min(ws) else 0))
    return nav_tracks(nodes, adj, dmap, origins, kcarts), (nodes, adj, dmap, origins)


def emit(bsp, out, kteams, kcarts, pk3arg=''):
    mapname = os.path.splitext(os.path.basename(bsp))[0]
    d = open(bsp, 'rb').read()
    assert d[:4] == b'IBSP', d[:4]
    off, ln = struct.unpack_from('<ii', d, 8)
    ents = d[off:off + ln].split(b'\0')[0].decode('latin-1')
    blocks = re.findall(r'\{[^{}]*\}', ents)
    mclass = {}
    for b in blocks:
        m = re.search(r'"model"\s+"(\*\d+)"', b)
        if m:
            mclass[m.group(1)] = re.search(r'"classname"\s+"([^"]+)"', b).group(1)
    models = sorted(mclass, key=lambda s: int(s[1:]))
    visible = [m for m in models if not mclass[m].startswith('trigger_')] or models
    spawns = [b for b in blocks if 'info_player_team1' in b or 'info_player_team2' in b]

    def origin(b):
        m = re.search(r'"origin"\s+"([-\d. ]+)"', b)
        return [float(x) for x in m.group(1).split()] if m else None

    pts = [origin(b) for b in spawns if origin(b)]
    print('inline models:', models[:6], 'visible:', visible[:kcarts], 'team spawns:', len(pts))

    tracks, _ = build_tracks(bsp, mapname, pts, kteams, kcarts, pk3arg)

    extra = []
    named = []
    for c in range(kcarts):
        track = tracks[c]
        names = ['plc%dn%d' % (c, i) for i in range(5)]
        named.append((names, track))
        for i, (name, p) in enumerate(zip(names, track)):
            e = ['{', '"classname" "plc_path"', '"targetname" "%s"' % name,
                 '"origin" "%.0f %.0f %.0f"' % tuple(p)]
            if i + 1 < 5:
                e.append('"target" "%s"' % names[i + 1])
            if i == 2:
                e.append('"spawnflags" "1"')
            e.append('}')
            extra.append('\n'.join(e))
        extra.append('\n'.join(['{', '"classname" "func_plc_cart"',
                                '"model" "%s"' % visible[c % len(visible)],
                                '"target" "%s"' % names[0], '"speed" "40"', '}']))

    goal_cnts = [4, 13, 12, 9, 3][:kteams]
    for t, cnt in enumerate(goal_cnts):
        names, track = named[t % kcarts]
        extra.append('\n'.join(['{', '"classname" "plc_goal"',
                                '"cnt" "%d"' % cnt, '"target" "%s"' % names[4],
                                '"radius" "64"',
                                '"origin" "%.0f %.0f %.0f"' % tuple(track[4]),
                                '}']))

    for c, (names, track) in enumerate(named):
        L = sum(math.dist(track[i], track[i + 1]) for i in range(4))
        print('cart %d: %s -> %s length %.0f' % (c, track[0], track[4], L))
    sep = min(math.dist(pa, pb) for _, ta in named[:1] for pa in ta
              for _, tb in named[1:] for pb in tb) if kcarts > 1 else -1
    print('teams', kteams, 'carts', kcarts, 'min inter-track node distance %.0f' % sep)

    open(out, 'w').write(ents.rstrip('\0') + '\n' + '\n'.join(extra) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    bsp, out = sys.argv[1], sys.argv[2]
    kteams = max(2, min(5, int(sys.argv[3]))) if len(sys.argv) > 3 else 2
    kcarts = max(1, min(4, int(sys.argv[4]))) if len(sys.argv) > 4 else 2
    pk3arg = sys.argv[5] if len(sys.argv) > 5 else ''
    emit(bsp, out, kteams, kcarts, pk3arg)
