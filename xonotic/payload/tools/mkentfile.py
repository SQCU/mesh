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


WB, WXY, WZ, WHEEL = 1.0, 0.3, 1.0, 42.0


NN = 12

def make_clip(d):
    try:
        def lump(i):
            off, ln = struct.unpack_from('<ii', d, 8 + i * 8)
            return off, ln
        to, tl = lump(1); po, pl = lump(2); bo, bl = lump(8); so, sl = lump(9)
        ntex = tl // 72
        solid = [struct.unpack_from('<i', d, to + i * 72 + 68)[0] & 1 for i in range(ntex)]
        planes = [struct.unpack_from('<4f', d, po + i * 16) for i in range(pl // 16)]
        brushes = [struct.unpack_from('<3i', d, bo + i * 12) for i in range(bl // 12)]
        sides = [struct.unpack_from('<2i', d, so + i * 8) for i in range(sl // 8)]
    except Exception:
        return lambda p: False
    def inside(p):
        for fs, ns, tx in brushes:
            if tx < 0 or tx >= ntex or not solid[tx]:
                continue
            ok = True
            for k in range(fs, fs + ns):
                pli, _ = sides[k]
                nx, ny, nz, dd = planes[pli]
                if p[0] * nx + p[1] * ny + p[2] * nz - dd > 0.25:
                    ok = False
                    break
            if ok:
                return True
        return False
    return inside

def clear_project(Q, A, inside):
    n = 0
    for i in range(len(Q)):
        if not inside(tuple(Q[i])):
            continue
        a = list(A[i])
        g = 0
        while inside(tuple(a)) and g < 8:
            a[2] += 12
            g += 1
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = (lo + hi) / 2
            m = tuple(Q[i][k] + (a[k] - Q[i][k]) * mid for k in range(3))
            if inside(m):
                lo = mid
            else:
                hi = mid
        f = min(1.0, hi + 0.05)
        Q[i] = [Q[i][k] + (a[k] - Q[i][k]) * f for k in range(3)]
        n += 1
    return n

def chord_ok(Q, i, j, inside):
    a, b = Q[i], Q[j]
    for k in range(i + 1, j):
        f = (k - i) / (j - i)
        m = tuple(a[t] + f * (b[t] - a[t]) for t in range(3))
        if inside(m) or math.dist(m, Q[k]) > 40:
            return False
    return True

def adaptive_nodes(Q, inside):
    idx = [0]
    i = 0
    while i < len(Q) - 1:
        j = len(Q) - 1
        while j > i + 1 and not chord_ok(Q, i, j, inside):
            j -= 1
        idx.append(j)
        i = j
    return [Q[k] for k in idx]

def make_floor_tracer(d):
    def lump(i):
        return struct.unpack_from('<ii', d, 8 + i * 8)
    to, tl = lump(1)
    po, pl = lump(2)
    bo, bl = lump(8)
    so, sl = lump(9)
    ntex = tl // 72
    solid = [bool(struct.unpack_from('<i', d, to + i * 72 + 68)[0] & 1) for i in range(ntex)]
    planes = [struct.unpack_from('<4f', d, po + i * 16) for i in range(pl // 16)]
    brushes = []
    for i in range(bl // 12):
        fs, ns, tex = struct.unpack_from('<iii', d, bo + i * 12)
        if 0 <= tex < ntex and solid[tex]:
            brushes.append([planes[struct.unpack_from('<ii', d, so + (fs + k) * 8)[0]] for k in range(ns)])
    if not brushes:
        return None

    def trace(x, y, zh):
        oz = zh + 64
        best = None
        for sides in brushes:
            tmin, tmax, ok = -1e18, 1e18, True
            for nx, ny, nz, pdst in sides:
                denom = -nz
                num = pdst - (nx * x + ny * y + nz * oz)
                if denom > 1e-9:
                    tmax = min(tmax, num / denom)
                elif denom < -1e-9:
                    tmin = max(tmin, num / denom)
                elif num < 0:
                    ok = False
                    break
            if not ok or tmin > tmax or tmax < 0:
                continue
            fz = oz - (tmin if tmin > 0 else 0)
            if fz <= zh + 8 and (best is None or fz > best):
                best = fz
        return best
    return trace


def resample_path(poly, sp):
    cum = [0.0]
    for i in range(1, len(poly)):
        cum.append(cum[-1] + math.dist(poly[i - 1], poly[i]))
    total = cum[-1]
    m = max(2, int(round(total / sp)))
    out = []
    for k in range(m + 1):
        t = total * k / m
        j = 0
        while j < len(cum) - 1 and cum[j + 1] < t:
            j += 1
        if j >= len(poly) - 1:
            out.append(poly[-1][:])
            continue
        segl = cum[j + 1] - cum[j]
        f = (t - cum[j]) / segl if segl > 0 else 0.0
        out.append([poly[j][a] + (poly[j + 1][a] - poly[j][a]) * f for a in range(3)])
    return out


def bending_energy(P):
    return sum(sum((P[i - 1][a] - 2 * P[i][a] + P[i + 1][a]) ** 2 for a in range(3))
               for i in range(1, len(P) - 1))


def smooth_curve(P, xy0, z0, iters=80):
    m = len(P) - 1
    Q = [r[:] for r in P]
    c = lambda i: 0 if i < 0 else (m if i > m else i)
    for _ in range(iters):
        for i in range(1, m):
            for a in range(2):
                nb = 4 * (Q[c(i - 1)][a] + Q[c(i + 1)][a]) - Q[c(i - 2)][a] - Q[c(i + 2)][a]
                Q[i][a] = (WB * nb + WXY * xy0[i][a]) / (6 * WB + WXY)
            nb = 4 * (Q[c(i - 1)][2] + Q[c(i + 1)][2]) - Q[c(i - 2)][2] - Q[c(i + 2)][2]
            Q[i][2] = (WB * nb + WZ * z0[i]) / (6 * WB + WZ)
    return Q


def snap_curve(P, n):
    if len(P) < 2:
        return [P[0][:] for _ in range(n)]
    cum = [0.0]
    for i in range(1, len(P)):
        cum.append(cum[-1] + math.dist(P[i - 1], P[i]))
    total = cum[-1]
    out = []
    for k in range(n):
        t = total * k / (n - 1)
        j = min(range(len(cum)), key=lambda x: abs(cum[x] - t))
        out.append(P[j][:])
    return out


def nav_tracks(nodes, adj, dmap, origins, kcarts, tracer, clip):
    used = set()
    tracks = []
    st = {'e0': 0.0, 'e1': 0.0, 'mxy': 0.0, 'mz': 0.0, 'bsp': 0, 'nav': 0}
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
        poly = [list(nodes[i]) for i in p]
        R = resample_path(poly, 64.0)
        xy0 = [(r[0], r[1]) for r in R]
        z0 = []
        for r in R:
            fz = tracer(r[0], r[1], r[2]) if tracer else None
            if fz is not None and r[2] - 80 <= fz <= r[2] + 16:
                z0.append(fz)
                st['bsp'] += 1
            else:
                z0.append(r[2] - 26)
                st['nav'] += 1
        z0[0] = R[0][2] - 26
        z0[-1] = R[-1][2] - 26
        P0 = [[xy0[i][0], xy0[i][1], z0[i]] for i in range(len(R))]
        Q = smooth_curve(P0, xy0, z0)
        st['e0'] += bending_energy(P0)
        st['e1'] += bending_energy(Q)
        st['mxy'] = max(st['mxy'], max(math.dist((Q[i][0], Q[i][1]), xy0[i]) for i in range(len(Q))))
        st['mz'] = max(st['mz'], max(abs(Q[i][2] - z0[i]) for i in range(len(Q))))
        track = [[q[0], q[1], q[2] - 26 + WHEEL] for q in poly]
        emb = sum(1 for q in track if clip(tuple(q)))
        if emb:
            print('nav: %d waypoint nodes read embedded, nudging up' % emb)
            for q in track:
                g = 0
                while clip(tuple(q)) and g < 6:
                    q[2] += 12
                    g += 1
        tracks.append(track)
    return tracks, st


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


def build_tracks(bsp, mapname, pts, kteams, kcarts, pk3arg='', d=None):
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
    try:
        tracer = make_floor_tracer(d if d is not None else open(bsp, 'rb').read())
    except Exception:
        tracer = None
    tracks, st = nav_tracks(nodes, adj, dmap, origins, kcarts, tracer, make_clip(d) if d else (lambda p: False))
    print('nav: terrain %s BSP-floors=%d nav-z-interp=%d wheel_offset=%.0f weights bend=%.1f xy=%.2f z=%.2f' %
          ('BSP-trace' if tracer else 'nav-z-interp(no BSP)', st['bsp'], st['nav'], WHEEL, WB, WXY, WZ))
    print('nav: curvature %.0f->%.0f (%.1f%%) maxXYdev=%.1f maxZdev=%.1f' %
          (st['e0'], st['e1'], 100 * st['e1'] / st['e0'] if st['e0'] else 0, st['mxy'], st['mz']))
    return tracks, (nodes, adj, dmap, origins)


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

    tracks, _ = build_tracks(bsp, mapname, pts, kteams, kcarts, pk3arg, d)

    extra = []
    named = []
    for c in range(kcarts):
        track = tracks[c]
        NC = len(track)
        names = ['plc%dn%d' % (c, i) for i in range(NC)]
        named.append((names, track))
        for i, (name, p) in enumerate(zip(names, track)):
            e = ['{', '"classname" "plc_path"', '"targetname" "%s"' % name,
                 '"origin" "%.0f %.0f %.0f"' % tuple(p)]
            if i + 1 < NC:
                e.append('"target" "%s"' % names[i + 1])
            if 0 < i < NC - 1:
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
                                '"cnt" "%d"' % cnt, '"target" "%s"' % names[-1],
                                '"radius" "64"',
                                '"origin" "%.0f %.0f %.0f"' % tuple(track[-1]),
                                '}']))

    for c, (names, track) in enumerate(named):
        L = sum(math.dist(track[i], track[i + 1]) for i in range(len(track) - 1))
        print('cart %d: %s -> %s length %.0f nodes %d' % (c, track[0], track[-1], L, len(track)))
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
