import struct, sys, re, math, os, glob, subprocess, heapq

BADMULT, CLEAR_TARGET, CLEAR_CAP, WB, WMED, WFLOOR, WHEEL = 8.0, 64.0, 256.0, 1.0, 0.3, 1.0, 42.0
OUTER, INNER, FLOAT_LIM, EPS, CELL = 3, 30, 96.0, 0.25, 512.0


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


def kcenter(adj, k, comp=None):
    if comp is None:
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


class Bsp:
    def __init__(self, d):
        def lump(i):
            return struct.unpack_from('<ii', d, 8 + i * 8)
        to, tl = lump(1)
        po, pl = lump(2)
        bo, bl = lump(8)
        so, sl = lump(9)
        ntex = tl // 72
        solid = [struct.unpack_from('<i', d, to + i * 72 + 68)[0] & 1 for i in range(ntex)]
        planes = [struct.unpack_from('<4f', d, po + i * 16) for i in range(pl // 16)]
        sides = [struct.unpack_from('<2i', d, so + i * 8) for i in range(sl // 8)]
        self.brushes, self.aabbs = [], []
        for i in range(bl // 12):
            fs, ns, tx = struct.unpack_from('<iii', d, bo + i * 12)
            if tx < 0 or tx >= ntex or not solid[tx]:
                continue
            bp = [planes[sides[k][0]] for k in range(fs, fs + ns)]
            lo = [-1e18] * 3
            hi = [1e18] * 3
            for nx, ny, nz, dd in bp:
                for a, c in enumerate((nx, ny, nz)):
                    if c > 0.999:
                        hi[a] = min(hi[a], dd)
                    elif c < -0.999:
                        lo[a] = max(lo[a], -dd)
            self.brushes.append(bp)
            self.aabbs.append((lo, hi))
        self.grid = {}
        for bi, (lo, hi) in enumerate(self.aabbs):
            x0, x1 = int(math.floor(lo[0] / CELL)), int(math.floor(hi[0] / CELL))
            y0, y1 = int(math.floor(lo[1] / CELL)), int(math.floor(hi[1] / CELL))
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    self.grid.setdefault((cx, cy), []).append(bi)

    def cell(self, x, y):
        return self.grid.get((int(math.floor(x / CELL)), int(math.floor(y / CELL))), ())

    def inside(self, p):
        x, y, z = p
        for bi in self.cell(x, y):
            lo, hi = self.aabbs[bi]
            if not (lo[0] - EPS <= x <= hi[0] + EPS and lo[1] - EPS <= y <= hi[1] + EPS
                    and lo[2] - EPS <= z <= hi[2] + EPS):
                continue
            ok = True
            for nx, ny, nz, dd in self.brushes[bi]:
                if x * nx + y * ny + z * nz - dd > EPS:
                    ok = False
                    break
            if ok:
                return True
        return False

    def floor(self, x, y, zh):
        oz = zh + 64
        best = None
        for bi in self.cell(x, y):
            lo, hi = self.aabbs[bi]
            if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]) or lo[2] > oz:
                continue
            tmin, tmax, ok = -1e18, 1e18, True
            for nx, ny, nz, dd in self.brushes[bi]:
                denom = -nz
                num = dd - (nx * x + ny * y + nz * oz)
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

    def clearance(self, x, y, z):
        dmin, away = CLEAR_CAP, None
        for k in range(8):
            ang = k * math.pi / 4
            dx, dy = math.cos(ang), math.sin(ang)
            t = 24.0
            hitd = None
            while t <= CLEAR_CAP:
                if self.inside((x + dx * t, y + dy * t, z)):
                    lo2, hi2 = t - 24.0, t
                    for _ in range(6):
                        mid = (lo2 + hi2) / 2
                        if self.inside((x + dx * mid, y + dy * mid, z)):
                            hi2 = mid
                        else:
                            lo2 = mid
                    hitd = lo2
                    break
                t += 24.0
            if hitd is not None and hitd < dmin:
                dmin, away = hitd, (-dx, -dy)
        return dmin, away


def seg_bad(bsp, pa, pb):
    L = math.dist(pa, pb)
    n = max(2, int(math.ceil(L / 32.0)))
    consec = 0
    sawfloor = False
    for k in range(n + 1):
        f = k / n
        p = tuple(pa[t] + f * (pb[t] - pa[t]) for t in range(3))
        if bsp.inside(p):
            return 'solid'
        fz = bsp.floor(p[0], p[1], p[2])
        if fz is None:
            consec += 1
        else:
            sawfloor = True
            if p[2] - fz > FLOAT_LIM:
                consec += 1
            else:
                consec = 0
        if consec >= 2:
            return 'fly'
    if not sawfloor:
        return 'nofloor'
    return None


def classify_edges(nodes, adj, bsp):
    bad = {}
    for u in range(len(adj)):
        for v in adj[u]:
            if v <= u:
                continue
            pa = (nodes[u][0], nodes[u][1], nodes[u][2] + 16)
            pb = (nodes[v][0], nodes[v][1], nodes[v][2] + 16)
            r = seg_bad(bsp, pa, pb)
            if r:
                bad[(u, v)] = r
    return bad


def route(adj, adj_ok, bad, src, goal):
    D, prev = dijkstra(adj_ok, src)
    admitted = []
    if D[goal] == math.inf:
        adj_adm = [dict(a) for a in adj]
        for (u, v) in bad:
            adj_adm[u][v] *= BADMULT
            adj_adm[v][u] *= BADMULT
        D, prev = dijkstra(adj_adm, src)
    p, u = [], goal
    while u != -1:
        p.append(u)
        u = prev[u]
    p = p[::-1] if len(p) > 1 else [src]
    segbad = []
    for i in range(1, len(p)):
        e = (min(p[i - 1], p[i]), max(p[i - 1], p[i]))
        segbad.append(e in bad)
        if e in bad:
            admitted.append((e[0], e[1], bad[e]))
    return p, segbad, admitted


def resample_marked(poly, segbad, sp):
    cum = [0.0]
    for i in range(1, len(poly)):
        cum.append(cum[-1] + math.dist(poly[i - 1], poly[i]))
    total = cum[-1]
    m = max(2, int(round(total / sp)))
    pts, mark = [], []
    for k in range(m + 1):
        t = total * k / m
        j = 0
        while j < len(cum) - 1 and cum[j + 1] < t:
            j += 1
        j = min(j, len(poly) - 2)
        segl = cum[j + 1] - cum[j]
        f = (t - cum[j]) / segl if segl > 0 else 0.0
        pts.append([poly[j][a] + (poly[j + 1][a] - poly[j][a]) * f for a in range(3)])
        mark.append(segbad[j] if segbad else False)
    return pts, mark


def bending_energy(P):
    return sum(sum((P[i - 1][a] - 2 * P[i][a] + P[i + 1][a]) ** 2 for a in range(3))
               for i in range(1, len(P) - 1))


def medial_smooth(bsp, R, mark):
    m = len(R) - 1
    zf = []
    for r in R:
        fz = bsp.floor(r[0], r[1], r[2]) if bsp else None
        zf.append(fz if fz is not None and r[2] - 80 <= fz <= r[2] + 16 else r[2] - 26)
    Q = [[R[i][0], R[i][1], zf[i] + WHEEL] for i in range(m + 1)]
    Q[0][2] = R[0][2] + 16
    Q[-1][2] = R[-1][2] + 16
    A = [[q[0], q[1]] for q in Q]
    c = lambda i: 0 if i < 0 else (m if i > m else i)
    for _ in range(OUTER):
        for i in range(1, m):
            dmin, away = bsp.clearance(Q[i][0], Q[i][1], Q[i][2]) if bsp else (CLEAR_CAP, None)
            if away and dmin < CLEAR_TARGET:
                push = min(CLEAR_TARGET - dmin, 24.0)
                A[i] = [Q[i][0] + away[0] * push, Q[i][1] + away[1] * push]
            else:
                A[i] = [Q[i][0], Q[i][1]]
            fz = bsp.floor(Q[i][0], Q[i][1], Q[i][2]) if bsp else None
            if fz is not None and Q[i][2] - 80 <= fz + WHEEL <= Q[i][2] + 80:
                zf[i] = fz
        for _ in range(INNER):
            for i in range(1, m):
                for a in range(2):
                    nb = 4 * (Q[c(i - 1)][a] + Q[c(i + 1)][a]) - Q[c(i - 2)][a] - Q[c(i + 2)][a]
                    Q[i][a] = (WB * nb + WMED * A[i][a]) / (6 * WB + WMED)
                nb = 4 * (Q[c(i - 1)][2] + Q[c(i + 1)][2]) - Q[c(i - 2)][2] - Q[c(i + 2)][2]
                Q[i][2] = (WB * nb + WFLOOR * (zf[i] + WHEEL)) / (6 * WB + WFLOOR)
    return Q, zf


def chord_clean(bsp, a, b, exempt):
    L = math.dist(a, b)
    n = max(1, int(math.ceil(L / 32.0)))
    for k in range(n + 1):
        f = k / n
        p = tuple(a[t] + f * (b[t] - a[t]) for t in range(3))
        if bsp.inside(p):
            return False
        if not exempt:
            fz = bsp.floor(p[0], p[1], p[2])
            if fz is None or p[2] - fz > FLOAT_LIM:
                return False
    return True


def adaptive_nodes(bsp, Q, mark):
    idx = [0]
    i = 0
    while i < len(Q) - 1:
        j = len(Q) - 1
        while j > i + 1:
            ex = any(mark[i:j + 1])
            dev = all(math.dist(Q[k], [Q[i][t] + (k - i) / (j - i) * (Q[j][t] - Q[i][t])
                                       for t in range(3)]) <= 40 for k in range(i + 1, j))
            if dev and chord_clean(bsp, Q[i], Q[j], ex):
                break
            j -= 1
        idx.append(j)
        i = j
    track = [Q[k][:] for k in idx]
    exempt = [any(mark[idx[s]:idx[s + 1] + 1]) for s in range(len(idx) - 1)]
    return track, exempt


def validate_chain(bsp, track, exempt):
    va, vb, ns = 0, 0, 0
    for s in range(len(track) - 1):
        a, b = track[s], track[s + 1]
        L = math.dist(a, b)
        n = max(1, int(math.ceil(L / 48.0)))
        for k in range(n + 1):
            f = k / n
            p = tuple(a[t] + f * (b[t] - a[t]) for t in range(3))
            ns += 1
            if bsp.inside(p):
                va += 1
            elif not exempt[s]:
                fz = bsp.floor(p[0], p[1], p[2])
                if fz is None or p[2] - fz > FLOAT_LIM:
                    vb += 1
    return va, vb, ns


def repair_chain(bsp, track, exempt):
    for _ in range(6):
        fixed = False
        s = 0
        while s < len(track) - 1:
            a, b = track[s], track[s + 1]
            L = math.dist(a, b)
            n = max(1, int(math.ceil(L / 48.0)))
            ins = None
            for k in range(1, n):
                f = k / n
                p = [a[t] + f * (b[t] - a[t]) for t in range(3)]
                if bsp.inside(tuple(p)):
                    g = 0
                    while bsp.inside(tuple(p)) and g < 12:
                        p[2] += 12
                        g += 1
                    ins = p
                    break
                if not exempt[s]:
                    fz = bsp.floor(p[0], p[1], p[2])
                    if fz is not None and p[2] - fz > FLOAT_LIM:
                        ins = [p[0], p[1], fz + WHEEL]
                        break
            if ins is not None:
                track.insert(s + 1, ins)
                exempt.insert(s, exempt[s])
                fixed = True
            s += 1
        if not fixed:
            break
    for q in track:
        g = 0
        while bsp.inside(tuple(q)) and g < 12:
            q[2] += 12
            g += 1
    return track, exempt


def nav_tracks(nodes, adj, dmap, origins, kcarts, bsp, bad):
    adj_ok = [{v: w for v, w in a.items()
               if (min(u, v), max(u, v)) not in bad} for u, a in enumerate(adj)]
    used = set()
    tracks, exempts, alladm = [], [], []
    st = {'e0': 0.0, 'e1': 0.0}
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
        p, segbad, admitted = route(adj, adj_ok, bad, src, goal)
        alladm.append(admitted)
        poly = [list(nodes[i]) for i in p]
        if bsp is None or len(poly) < 2:
            track = [[q[0], q[1], q[2] + 16] for q in poly] or [[nodes[src][0], nodes[src][1], nodes[src][2] + 16]]
            while len(track) < 2:
                track.append(track[0][:])
            tracks.append(track)
            exempts.append([True] * (len(track) - 1))
            continue
        R, mark = resample_marked(poly, segbad, 64.0)
        st['e0'] += bending_energy([[r[0], r[1], r[2] + 16] for r in R])
        Q, zf = medial_smooth(bsp, R, mark)
        st['e1'] += bending_energy(Q)
        track, exempt = adaptive_nodes(bsp, Q, mark)
        track, exempt = repair_chain(bsp, track, exempt)
        tracks.append(track)
        exempts.append(exempt)
    return tracks, exempts, alladm, st


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
    try:
        B = Bsp(d if d is not None else open(bsp, 'rb').read())
    except Exception:
        B = None
    bad = classify_edges(nodes, adj, B) if B else {}
    ne = sum(len(a) for a in adj) // 2
    rs = {'solid': 0, 'fly': 0, 'nofloor': 0}
    for r in bad.values():
        rs[r] += 1
    print('nav: %s waypoints=%d links=%d bad=%d (solid=%d fly=%d nofloor=%d)' %
          (mapname, len(nodes), ne, len(bad), rs['solid'], rs['fly'], rs['nofloor']))
    adj_ok = [{v: w for v, w in a.items()
               if (min(u, v), max(u, v)) not in bad} for u, a in enumerate(adj)]
    comp_ok = largest_component(adj_ok)
    cand_k = max(3, kcarts)
    if len(comp_ok) >= cand_k:
        origins, dmap = kcenter(adj_ok, cand_k, comp_ok)
    else:
        origins, dmap = kcenter(adj, cand_k)
    print('nav: candidate_origins=%d on %s subgraph (ok_component=%d/%d)' %
          (len(origins), 'CART_OK' if len(comp_ok) >= cand_k else 'FULL', len(comp_ok), len(nodes)))
    ws = []
    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            wd = dmap[origins[i]][origins[j]]
            ed = math.dist(nodes[origins[i]], nodes[origins[j]])
            ws.append(wd)
            print('nav: origin %d %s <-> origin %d %s  walk=%.0f  euclid=%.0f  ratio=%.2f' %
                  (i, tuple(round(x) for x in nodes[origins[i]]), j,
                   tuple(round(x) for x in nodes[origins[j]]), wd, ed, wd / ed if ed else 0))
    print('nav: pairwise walk min=%.0f mean=%.0f max=%.0f balance_ratio=%.2f' %
          (min(ws), sum(ws) / len(ws), max(ws), max(ws) / min(ws) if min(ws) else 0))
    tracks, exempts, alladm, st = nav_tracks(nodes, adj, dmap, origins, kcarts, B, bad)
    nadm = sum(len(a) for a in alladm)
    for c, a in enumerate(alladm):
        if a:
            print('nav: cart %d admitted %d bad edges: %s' % (c, len(a), a))
    print('nav: admitted_total=%d weights bend=%.1f w_med=%.2f w_floor=%.2f target=%.0f cap=%.0f outer=%d inner=%d badmult=%.0f' %
          (nadm, WB, WMED, WFLOOR, CLEAR_TARGET, CLEAR_CAP, OUTER, INNER, BADMULT))
    print('nav: curvature %.0f->%.0f (%.1f%%)' %
          (st['e0'], st['e1'], 100 * st['e1'] / st['e0'] if st['e0'] else 0))
    if B:
        va, vb, ns = 0, 0, 0
        for track, exempt in zip(tracks, exempts):
            a, b2, n2 = validate_chain(B, track, exempt)
            va += a
            vb += b2
            ns += n2
        exs = sum(sum(1 for e in ex if e) for ex in exempts)
        print('nav: validation samples=%d solid_viol=%d float_viol=%d exempt_segs=%d %s' %
              (ns, va, vb, exs, 'PASS' if va == 0 and vb == 0 else 'FAILED'))
    else:
        print('nav: validation skipped (no BSP geometry)')
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
    visible = [m for m in models if not mclass[m].startswith('trigger_')] or models or ['*1']
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
