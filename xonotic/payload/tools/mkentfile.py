import struct, sys, re, math, os, glob, subprocess, heapq

BADMULT, CLEAR_TARGET, CLEAR_CAP, WB, WMED, WFLOOR, WHEEL = 8.0, 64.0, 256.0, 1.0, 0.3, 1.0, 42.0
OUTER, INNER, FLOAT_LIM, EPS, CELL = 3, 30, 96.0, 0.25, 512.0
WPF_BAD = (1 << 21) | (1 << 15) | (1 << 14) | (1 << 13)


def push_cvars():
    r, h = 160.0, 96.0
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfg', 'gamemodes-payload.cfg')
    try:
        for line in open(cfg):
            m = re.match(r'\s*set\s+g_payload_push_(radius|height)\s+(\S+)', line)
            if m:
                if m.group(1) == 'radius':
                    r = float(m.group(2))
                else:
                    h = float(m.group(2))
    except Exception:
        pass
    return r, h


PUSH_R, PUSH_H = push_cvars()


def pk3_read(fname, bsp, pk3arg):
    cands = []
    if pk3arg:
        cands.append(pk3arg)
    if os.environ.get('XON_MAPS_PK3'):
        cands.append(os.environ['XON_MAPS_PK3'])
    cands += sorted(glob.glob(os.path.expanduser('~/dox/xonotic/Xonotic/data/*maps*.pk3')), reverse=True)
    cands += sorted(glob.glob(os.path.expanduser('~/dox/xonotic/**/*maps*.pk3'), recursive=True), reverse=True)
    for pk3 in cands:
        r = subprocess.run(['unzip', '-p', pk3, 'maps/' + fname], capture_output=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.decode('latin-1')
    loose = os.path.join(os.path.dirname(bsp) or '.', fname)
    if os.path.exists(loose):
        return open(loose, 'r', encoding='latin-1').read()
    return ''


def load_cache(mapname, bsp, pk3arg):
    name = mapname
    for _ in range(4):
        text = pk3_read(name + '.waypoints.cache', bsp, pk3arg)
        stripped = text.strip()
        if stripped.endswith('.waypoints.cache') and '*' not in stripped and '\n' not in stripped:
            name = stripped[:-len('.waypoints.cache')]
            continue
        return text, name
    return '', name


def parse_waypoints(text):
    flags = {}
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith('//')]
    for i in range(0, len(lines) - 2, 3):
        try:
            a = [float(x) for x in lines[i].strip().strip("'").split()]
            b = [float(x) for x in lines[i + 1].strip().strip("'").split()]
            fl = int(float(lines[i + 2].strip()))
        except ValueError:
            continue
        if len(a) == 3 and len(b) == 3:
            flags[tuple(round((a[k] + b[k]) / 2, 1) for k in range(3))] = fl
    return flags


def load_flags(mapname, resolved, bsp, pk3arg, nodes):
    best = {}
    bestn = -1
    for name in dict.fromkeys([mapname, resolved, mapname[:-5] if mapname.endswith('.race') else mapname + '.race']):
        fl = parse_waypoints(pk3_read(name + '.waypoints', bsp, pk3arg))
        n = sum(1 for nd in nodes if nd in fl)
        if n > bestn:
            bestn, best = n, fl
    return best


def trigger_boxes(d):
    try:
        off, ln = struct.unpack_from('<ii', d, 8)
        ents = d[off:off + ln].split(b'\0')[0].decode('latin-1')
        mo, ml = struct.unpack_from('<ii', d, 8 + 7 * 8)
        boxes = []
        for b in re.findall(r'\{[^{}]*\}', ents):
            if not re.search(r'"classname"\s+"(trigger_push|trigger_teleport)"', b):
                continue
            m = re.search(r'"model"\s+"\*(\d+)"', b)
            if not m:
                continue
            mi = int(m.group(1))
            if mo + mi * 40 + 24 > mo + ml:
                continue
            v = struct.unpack_from('<6f', d, mo + mi * 40)
            boxes.append(((v[0], v[1], v[2]), (v[3], v[4], v[5])))
        return boxes
    except Exception:
        return []


def seg_hits_box(pa, pb, lo, hi):
    t0, t1 = 0.0, 1.0
    for a in range(3):
        dd = pb[a] - pa[a]
        if abs(dd) < 1e-9:
            if pa[a] < lo[a] or pa[a] > hi[a]:
                return False
            continue
        u0 = (lo[a] - pa[a]) / dd
        u1 = (hi[a] - pa[a]) / dd
        if u0 > u1:
            u0, u1 = u1, u0
        t0 = max(t0, u0)
        t1 = min(t1, u1)
        if t0 > t1:
            return False
    return True


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


class Corridor:
    def __init__(self, pts):
        self.pts = pts
        self.grid = {}
        for i, p in enumerate(pts):
            self.grid.setdefault((int(math.floor(p[0] / PUSH_R)), int(math.floor(p[1] / PUSH_R))), []).append(i)

    def near(self, x, y, rings):
        cx, cy = int(math.floor(x / PUSH_R)), int(math.floor(y / PUSH_R))
        out = []
        for dx in range(-rings, rings + 1):
            for dy in range(-rings, rings + 1):
                out.extend(self.grid.get((cx + dx, cy + dy), ()))
        return out

    def contains(self, p):
        for i in self.near(p[0], p[1], 1):
            q = self.pts[i]
            if abs(p[2] - q[2]) <= PUSH_H and (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= PUSH_R * PUSH_R:
                return True
        return False

    def nearest(self, p):
        cand = []
        for rings in (1, 2, 4, 6):
            cand = self.near(p[0], p[1], rings)
            if cand:
                break
        if not cand:
            cand = range(len(self.pts))
        return self.pts[min(cand, key=lambda i: math.dist(p, self.pts[i]))]

    def project(self, p):
        if self.contains(p):
            return list(p)
        cand = []
        for rings in (2, 4, 6):
            cand = self.near(p[0], p[1], rings)
            if cand:
                break
        if not cand:
            cand = range(len(self.pts))
        best, bd = None, 1e18
        for i in cand:
            q = self.pts[i]
            z = min(max(p[2], q[2] - PUSH_H + 1), q[2] + PUSH_H - 1)
            dx, dy = p[0] - q[0], p[1] - q[1]
            L = math.hypot(dx, dy)
            if L > PUSH_R - 1:
                f = (PUSH_R - 1) / L
                x, y = q[0] + dx * f, q[1] + dy * f
            else:
                x, y = p[0], p[1]
            d = (x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2
            if d < bd:
                bd, best = d, [x, y, z]
        return best if best is not None else list(p)


def seg_bad(bsp, pa, pb):
    L = math.dist(pa, pb)
    n = max(2, int(math.ceil(L / 24.0)))
    sawfloor = False
    for k in range(n + 1):
        f = k / n
        p = tuple(pa[t] + f * (pb[t] - pa[t]) for t in range(3))
        if bsp.inside(p):
            return 'solid'
        fz = bsp.floor(p[0], p[1], p[2])
        if fz is not None:
            sawfloor = True
            if p[2] - fz > FLOAT_LIM:
                return 'fly'
        elif k in (0, n):
            return 'fly'
    if not sawfloor:
        return 'nofloor'
    return None


def classify_edges(nodes, adj, bsp, gen, flags, tboxes):
    bad = {}
    for u in range(len(adj)):
        for v in adj[u]:
            if v <= u:
                continue
            if u in gen or v in gen:
                bad[(u, v)] = 'gen'
                continue
            if (flags.get(nodes[u], 0) | flags.get(nodes[v], 0)) & WPF_BAD:
                bad[(u, v)] = 'flag'
                continue
            pa = (nodes[u][0], nodes[u][1], nodes[u][2] + 16)
            pb = (nodes[v][0], nodes[v][1], nodes[v][2] + 16)
            if any(seg_hits_box(pa, pb, lo, hi) for lo, hi in tboxes):
                bad[(u, v)] = 'trig'
                continue
            r = seg_bad(bsp, pa, pb)
            if r:
                bad[(u, v)] = r
    return bad


def standable_set(nodes, adj, bad, gen):
    pts = [list(n) for i, n in enumerate(nodes) if i not in gen]
    for u in range(len(adj)):
        for v in adj[u]:
            if v <= u or (u, v) in bad:
                continue
            a, b = nodes[u], nodes[v]
            L = math.dist(a, b)
            n = max(1, int(math.ceil(L / 48.0)))
            for k in range(1, n):
                f = k / n
                pts.append([a[t] + f * (b[t] - a[t]) for t in range(3)])
    return pts


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


def lipschitz_env(zf, spacing, slope=0.8):
    n = len(zf)
    out = list(zf)
    for i in range(n):
        for j in range(max(0, i - 4), min(n, i + 5)):
            out[i] = max(out[i], zf[j] - slope * abs(i - j) * spacing)
    return out


def medial_smooth(bsp, cor, R, mark):
    m = len(R) - 1
    zf = []
    for r in R:
        fz = bsp.floor(r[0], r[1], r[2]) if bsp else None
        zf.append(fz if fz is not None and r[2] - 80 <= fz <= r[2] + 16 else r[2] - 26)
    zf = lipschitz_env(zf, 64.0)
    Q = [[R[i][0], R[i][1], zf[i] + WHEEL] for i in range(m + 1)]
    Q[0][2] = R[0][2] + 16
    Q[-1][2] = R[-1][2] + 16
    A = [[q[0], q[1]] for q in Q]
    c = lambda i: 0 if i < 0 else (m if i > m else i)
    for _ in range(OUTER):
        for i in range(1, m):
            fz = bsp.floor(Q[i][0], Q[i][1], Q[i][2]) if bsp else None
            if fz is None or Q[i][2] - fz > FLOAT_LIM + WHEEL:
                A[i] = [R[i][0], R[i][1]]
                continue
            dmin, away = bsp.clearance(Q[i][0], Q[i][1], Q[i][2]) if bsp else (CLEAR_CAP, None)
            if away and dmin < CLEAR_TARGET:
                push = min(CLEAR_TARGET - dmin, 24.0)
                A[i] = [Q[i][0] + away[0] * push, Q[i][1] + away[1] * push]
            else:
                A[i] = [Q[i][0], Q[i][1]]
            if Q[i][2] - 80 <= fz + WHEEL <= Q[i][2] + 80:
                zf[i] = fz
        zs = lipschitz_env(zf, 64.0)
        for _ in range(INNER):
            for i in range(1, m):
                for a in range(2):
                    nb = 4 * (Q[c(i - 1)][a] + Q[c(i + 1)][a]) - Q[c(i - 2)][a] - Q[c(i + 2)][a]
                    Q[i][a] = (WB * nb + WMED * A[i][a]) / (6 * WB + WMED)
                nb = 4 * (Q[c(i - 1)][2] + Q[c(i + 1)][2]) - Q[c(i - 2)][2] - Q[c(i + 2)][2]
                Q[i][2] = (WB * nb + WFLOOR * (zs[i] + WHEEL)) / (6 * WB + WFLOOR)
        for i in range(1, m):
            if not cor.contains(Q[i]):
                Q[i] = cor.project(Q[i])
    return Q, zf


def chord_clean(bsp, cor, a, b, exempt):
    L = math.dist(a, b)
    n = 3 * max(1, int(math.ceil(L / 48.0)))
    for k in range(n + 1):
        f = k / n
        p = tuple(a[t] + f * (b[t] - a[t]) for t in range(3))
        if bsp.inside(p):
            return False
        if not exempt:
            if not cor.contains(p):
                return False
            fz = bsp.floor(p[0], p[1], p[2])
            if fz is None or p[2] - fz > FLOAT_LIM:
                return False
    return True


def feasible(bsp, cor, p, exempt):
    q = list(p)
    for _ in range(2):
        if not exempt:
            fz = bsp.floor(q[0], q[1], q[2])
            if fz is not None and q[2] - fz > FLOAT_LIM:
                if q[2] - fz <= FLOAT_LIM + 64:
                    q[2] = fz + WHEEL
                else:
                    n = cor.nearest(q)
                    f2 = bsp.floor(n[0], n[1], n[2])
                    q = [n[0], n[1], f2 + WHEEL if f2 is not None else n[2] + 16]
            elif fz is None:
                n = cor.nearest(q)
                f2 = bsp.floor(n[0], n[1], n[2])
                q = [n[0], n[1], f2 + WHEEL if f2 is not None else n[2] + 16]
            if not cor.contains(q):
                q = cor.project(q)
        g = 0
        while bsp.inside(tuple(q)) and g < 16:
            q[2] += 12
            g += 1
        if not bsp.inside(tuple(q)):
            if exempt or cor.contains(q):
                return q
    g = 0
    while bsp.inside(tuple(q)) and g < 16:
        q[2] += 12
        g += 1
    return q


def refine(bsp, cor, a, b, exempt, depth):
    if chord_clean(bsp, cor, a, b, exempt) or depth == 0 or math.dist(a, b) < 8:
        return [b]
    gm = [(a[t] + b[t]) / 2 for t in range(3)]
    mid = feasible(bsp, cor, gm, exempt)
    if math.dist(mid, a) < 4 or math.dist(mid, b) < 4:
        mid = list(gm)
        g = 0
        while bsp.inside(tuple(mid)) and g < 16:
            mid[2] += 12
            g += 1
    if math.dist(mid, a) < 4 or math.dist(mid, b) < 4:
        return [b]
    return refine(bsp, cor, a, mid, exempt, depth - 1) + refine(bsp, cor, mid, b, exempt, depth - 1)


def adaptive_nodes(bsp, cor, Q, R, mark):
    track = [Q[0][:]]
    exempt = []
    i = 0
    while i < len(Q) - 1:
        j = len(Q) - 1
        while j > i + 1:
            ex = any(mark[i:j + 1])
            dev = all(math.dist(Q[k], [Q[i][t] + (k - i) / (j - i) * (Q[j][t] - Q[i][t])
                                       for t in range(3)]) <= 40 for k in range(i + 1, j))
            if dev and chord_clean(bsp, cor, Q[i], Q[j], ex):
                break
            j -= 1
        ex = any(mark[i:j + 1])
        segs = refine(bsp, cor, track[-1], Q[j][:], ex, 6)
        pts = [track[-1]] + segs
        if any(not chord_clean(bsp, cor, pts[t], pts[t + 1], ex) for t in range(len(pts) - 1)):
            segs = []
            for k in range(i + 1, j + 1):
                q = [R[k][0], R[k][1], R[k][2] + 16]
                g = 0
                while bsp.inside(tuple(q)) and g < 16:
                    q[2] += 12
                    g += 1
                segs.append(q)
        for np in segs:
            track.append(np)
            exempt.append(ex)
        i = j
    return track, exempt


def seg_valid(bsp, cor, a, b, exempt):
    L = math.dist(a, b)
    n = max(1, int(math.ceil(L / 48.0)))
    for k in range(n + 1):
        f = k / n
        p = tuple(a[t] + f * (b[t] - a[t]) for t in range(3))
        if bsp.inside(p):
            return False
        if not exempt:
            if not cor.contains(p):
                return False
            fz = bsp.floor(p[0], p[1], p[2])
            if fz is None or p[2] - fz > FLOAT_LIM:
                return False
    return True


def polish_chain(bsp, cor, track, exempt):
    out = [track[0]]
    oex = []
    for seg in range(len(track) - 1):
        a, b = track[seg], track[seg + 1]
        ex = exempt[seg]
        if not seg_valid(bsp, cor, a, b, ex):
            L = math.dist(a, b)
            n = max(2, int(math.ceil(L / 24.0)))
            for k in range(1, n):
                f = k / n
                q = feasible(bsp, cor, [a[t] + f * (b[t] - a[t]) for t in range(3)], ex)
                if math.dist(q, out[-1]) >= 4 and math.dist(q, b) >= 4:
                    out.append(q)
                    oex.append(ex)
        out.append(b[:])
        oex.append(ex)
    return out, oex


def validate_chain(bsp, cor, track, exempt):
    va, vb, vc, vx, ns = 0, 0, 0, 0, 0
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
                continue
            inc = cor.contains(p)
            if exempt[s]:
                if not inc:
                    vx += 1
                continue
            if not inc:
                vc += 1
            fz = bsp.floor(p[0], p[1], p[2])
            if fz is None or p[2] - fz > FLOAT_LIM:
                vb += 1
    return va, vb, vc, vx, ns


DANGLE_MIN, OVERLAP_MAX, PEN0 = 320.0, 0.3, 8.0
REGION_LINK, HOST_MIN, HOST_FRACTION = 700.0, 8, 0.4

# Spatial regions of a fused megamap. Same measurement as design/AGENDA.md R20:
# union-find over origins with a REGION_LINK-unit XY link. Cart tracks placed in
# different regions make choosing cart A over cart B a traversal decision, not a
# free one.
def region_labels(points, thr=REGION_LINK):
    n = len(points)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    buckets = {}
    for i, p in enumerate(points):
        buckets.setdefault((int(math.floor(p[0] / thr)), int(math.floor(p[1] / thr))), []).append(i)
    for (cx, cy), idxs in buckets.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx + dx, cy + dy), ()):
                    for i in idxs:
                        if i >= j:
                            continue
                        if abs(points[i][0] - points[j][0]) <= thr and abs(points[i][1] - points[j][1]) <= thr:
                            a, b = find(i), find(j)
                            if a != b:
                                parent[a] = b
    roots, out = {}, []
    for i in range(n):
        r = find(i)
        out.append(roots.setdefault(r, len(roots)))
    return out


def components(adj):
    n = len(adj)
    seen = [False] * n
    out = []
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
        out.append(comp)
    out.sort(key=len, reverse=True)
    return out


def host_components(adj_ok, nodes, kcarts):
    """Pick one bot-reachable host component per cart, spread over regions.

    Every host is a genuine flood-fill component of the bad-edge-free navmesh,
    so each cart track stays bot-reachable end to end; the hosts sit in
    different regions, so the trip from cart i to cart j is a real cross-region
    traversal on the stock navmesh (which does connect the regions, through the
    fusion joins the cart-track filter drops).

    Regions are discovered at runtime by union-find over the node origins -- N
    of them, whatever the fused world happens to contain -- and never assumed.
    A region's secondary component is only used when it is a comparable place
    to put a cart (>= HOST_FRACTION of that region's largest); otherwise the
    carts share the region's largest component and the existing overlap
    planner separates them, which is the single-region behavior stock maps
    want.
    """
    comps = [c for c in components(adj_ok) if len(c) >= HOST_MIN]
    if not comps:
        comps = components(adj_ok)[:1]
    labels = region_labels(nodes)
    by_region = {}
    for comp in comps:
        counts = {}
        for u in comp:
            counts[labels[u]] = counts.get(labels[u], 0) + 1
        by_region.setdefault(max(counts, key=lambda r: counts[r]), []).append(comp)
    candidates = {}
    for region, group in by_region.items():
        head = group[0]
        candidates[region] = [head] + [c for c in group[1:] if len(c) >= HOST_FRACTION * len(head)]
    order = sorted(candidates, key=lambda r: -len(candidates[r][0]))
    hosts, cursor = [], {r: 0 for r in order}
    while len(hosts) < kcarts:
        progressed = False
        for region in order:
            if len(hosts) == kcarts:
                break
            index = cursor[region]
            if index < len(candidates[region]):
                hosts.append((region, candidates[region][index]))
                cursor[region] = index + 1
                progressed = True
        if not progressed:
            # Every distinct host is taken. Surplus carts go back onto the
            # region-largest components in the same round-robin order and share
            # them, which is exactly the pre-R20 single-component behavior --
            # for those carts only.
            index = 0
            while len(hosts) < kcarts:
                region = order[index % len(order)]
                hosts.append((region, candidates[region][0]))
                index += 1
            break
    return hosts[:kcarts]


def prune_component(adj_ok, comp):
    keep = set(comp)

    def deg(u):
        return sum(1 for v in adj_ok[u] if v in keep)

    pruned = 0
    changed = True
    while changed:
        changed = False
        for u in list(keep):
            if u not in keep or deg(u) != 1:
                continue
            chain = [u]
            L = 0.0
            prev, cur = -1, u
            while True:
                nxt = [v for v in adj_ok[cur] if v in keep and v != prev]
                if not nxt:
                    break
                L += adj_ok[cur][nxt[0]]
                prev, cur = cur, nxt[0]
                if deg(cur) != 2:
                    break
                chain.append(cur)
            if L < DANGLE_MIN and deg(cur) > 2:
                keep -= set(chain)
                pruned += len(chain)
                changed = True
    interior = [u for u in keep if deg(u) >= 2]
    bf = (sum(deg(u) for u in interior) / len(interior) - 1.0) if interior else 0.0
    return keep, bf, pruned, len(comp)


def prune_network(adj_ok):
    return prune_component(adj_ok, largest_component(adj_ok))


def subgraph(adj_ok, keep):
    return [{v: w for v, w in a.items() if u in keep and v in keep} for u, a in enumerate(adj_ok)]


def extract_path(prev, src, goal):
    p, u = [], goal
    while u != -1:
        p.append(u)
        u = prev[u]
    return p[::-1] if len(p) > 1 and p[-1] != p[0] else [src]


def plan_spans(nodes, adj, adj_ok, bad, kcarts, keep):
    adjn = subgraph(adj_ok, keep)
    comp = sorted(keep)
    ncand = min(max(3, 2 * kcarts), len(comp))
    if ncand < 2:
        only = comp[0] if comp else 0
        return [[only]] * kcarts, [[]] * kcarts, [[]] * kcarts, [[0.0] * kcarts for _ in range(kcarts)], [only] * kcarts
    cands, dmapc = kcenter(adjn, ncand, comp)
    order = sorted(((dmapc[a][b], a, b) for i, a in enumerate(cands) for b in cands[i + 1:]
                    if dmapc[a][b] < math.inf), reverse=True)

    def greedy(rows):
        out, used = [], set()
        for d, a, b in rows:
            if a not in used and b not in used:
                out.append((a, b))
                used |= {a, b}
            if len(out) == kcarts:
                return out
        for d, a, b in rows:
            if len(out) == kcarts:
                break
            if (a, b) not in out:
                out.append((a, b))
        i = 0
        while len(out) < kcarts and rows:
            out.append(rows[i % len(rows)][1:])
            i += 1
        return out

    pairings = [greedy(order)]
    if len(order) > 1:
        pairings.append(greedy(order[1:] + order[:1]))
    p0 = pairings[0]
    if len(p0) >= 2:
        pairings.append([(p0[i][0], p0[(i + 1) % len(p0)][1]) for i in range(len(p0))])
    seen = set()
    pairings = [pr for pr in pairings if not (tuple(pr) in seen or seen.add(tuple(pr)))]
    best = None
    for pairs in pairings:
        P = PEN0
        for _ in range(3):
            npen = [0] * len(nodes)
            paths, segbads, adms = [], [], []
            for a, b in pairs:
                adjp = [{v: w * (1 + P * (npen[u] + npen[v]) / 2) for v, w in adjn[u].items()}
                        for u in range(len(nodes))]
                D, prev = dijkstra(adjp, a)
                admitted = []
                if D[b] == math.inf:
                    adjf = [dict(x) for x in adj]
                    for (uu, vv) in bad:
                        adjf[uu][vv] *= BADMULT
                        adjf[vv][uu] *= BADMULT
                    for u in range(len(nodes)):
                        for v in adjf[u]:
                            adjf[u][v] *= (1 + P * (npen[u] + npen[v]) / 2)
                    D, prev = dijkstra(adjf, a)
                p = extract_path(prev, a, b)
                sb = []
                for i in range(1, len(p)):
                    e = (min(p[i - 1], p[i]), max(p[i - 1], p[i]))
                    sb.append(e in bad)
                    if e in bad:
                        admitted.append((e[0], e[1], bad[e]))
                for n in p:
                    npen[n] += 1
                paths.append(p)
                segbads.append(sb)
                adms.append(admitted)
            ov = [[0.0] * kcarts for _ in range(kcarts)]
            mx = 0.0
            for i in range(kcarts):
                for j in range(i + 1, kcarts):
                    si, sj = set(paths[i]), set(paths[j])
                    f = len(si & sj) / max(1, min(len(si), len(sj)))
                    ov[i][j] = ov[j][i] = f
                    mx = max(mx, f)
            if best is None or mx < best[0]:
                best = (mx, paths, segbads, adms, ov)
            if mx <= OVERLAP_MAX:
                break
            P *= 8
        if best[0] <= OVERLAP_MAX:
            break
    mx, paths, segbads, adms, ov = best

    def omat(ps):
        m = [[0.0] * kcarts for _ in range(kcarts)]
        top = 0.0
        for i in range(kcarts):
            for j in range(i + 1, kcarts):
                si, sj = set(ps[i]), set(ps[j])
                f = len(si & sj) / max(1, min(len(si), len(sj)))
                m[i][j] = m[j][i] = f
                top = max(top, f)
        return m, top

    for _ in range(3 * kcarts):
        if mx <= OVERLAP_MAX:
            break
        wi, wj = max(((i, j) for i in range(kcarts) for j in range(i + 1, kcarts)),
                     key=lambda t: ov[t[0]][t[1]])
        improved = False
        for ri, oth in ((wi, wj), (wj, wi)):
            for mode in (0, 1):
                if mode == 0:
                    blocked = set(paths[oth][1:-1])
                else:
                    blocked = set(paths[ri]) & set(paths[oth])
                blocked -= {paths[ri][0], paths[ri][-1]}
                adjb = [{v: w for v, w in adjn[u].items() if u not in blocked and v not in blocked}
                        for u in range(len(nodes))]
                D, prev = dijkstra(adjb, paths[ri][0])
                if D[paths[ri][-1]] == math.inf:
                    continue
                np_ = extract_path(prev, paths[ri][0], paths[ri][-1])
                cand = paths[:ri] + [np_] + paths[ri + 1:]
                nov, nmx = omat(cand)
                if nmx < mx:
                    paths, ov, mx = cand, nov, nmx
                    segbads[ri] = [
                        (min(np_[i - 1], np_[i]), max(np_[i - 1], np_[i])) in bad
                        for i in range(1, len(np_))]
                    adms[ri] = []
                    improved = True
                    break
            if improved:
                break
        if not improved:
            ri, oth = (wi, wj) if len(paths[wi]) <= len(paths[wj]) else (wj, wi)
            pen = [0] * len(nodes)
            for k2 in range(kcarts):
                if k2 != ri:
                    for n in paths[k2]:
                        pen[n] += 1
            adjb = [{v: w * (1 + 32 * (pen[u] + pen[v]) / 2) for v, w in adjn[u].items()}
                    for u in range(len(nodes))]
            bestalt = None
            dcache = {}
            for d, a2, b2 in order:
                if a2 not in dcache:
                    dcache[a2] = dijkstra(adjb, a2)
                D, prev = dcache[a2]
                if D[b2] == math.inf:
                    continue
                np_ = extract_path(prev, a2, b2)
                cand = paths[:ri] + [np_] + paths[ri + 1:]
                nov, nmx = omat(cand)
                if nmx < mx and (bestalt is None or nmx < bestalt[0]):
                    bestalt = (nmx, np_, nov)
            if bestalt is None:
                break
            nmx, np_, nov = bestalt
            paths = paths[:ri] + [np_] + paths[ri + 1:]
            ov, mx = nov, nmx
            segbads[ri] = [(min(np_[i - 1], np_[i]), max(np_[i - 1], np_[i])) in bad
                           for i in range(1, len(np_))]
            adms[ri] = []
    return paths, segbads, adms, ov, None


def flow_assign(nodes, paths, segbads, adj_ok):
    k = len(paths)
    segsets = []
    for p in paths:
        segs = []
        for i in range(1, len(p)):
            a, b = nodes[p[i - 1]], nodes[p[i]]
            L = math.dist(a, b)
            if L < 1:
                continue
            segs.append((tuple((a[t] + b[t]) / 2 for t in range(3)),
                         tuple((b[t] - a[t]) / L for t in range(3))))
        segsets.append(segs)
    rad = 2 * PUSH_R

    def score(flips):
        num = den = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                fi = -1.0 if flips >> i & 1 else 1.0
                fj = -1.0 if flips >> j & 1 else 1.0
                for mi, ti in segsets[i]:
                    for mj, tj in segsets[j]:
                        d = math.dist(mi, mj)
                        if d <= rad:
                            w = 1 - d / rad
                            num += w * fi * fj * sum(ti[t] * tj[t] for t in range(3))
                            den += w
        return num / den if den else 0.0

    dm = {}

    def spread(flips):
        orgs = [paths[i][-1] if flips >> i & 1 else paths[i][0] for i in range(k)]
        for o in set(orgs):
            if o not in dm:
                dm[o] = dijkstra(adj_ok, o)[0]
        ds = [dm[orgs[i]][orgs[j]] for i in range(k) for j in range(i + 1, k) if orgs[i] != orgs[j]]
        ds = [d for d in ds if d < math.inf]
        return min(ds) if ds else 0.0

    scored = sorted((round(score(f), 3), -spread(f), f) for f in range(1 << k))
    chosen = scored[0]
    worst = scored[-1][0]
    flips = chosen[2]
    out_paths, out_segbads = [], []
    for i in range(k):
        if flips >> i & 1:
            out_paths.append(paths[i][::-1])
            out_segbads.append(segbads[i][::-1])
        else:
            out_paths.append(paths[i])
            out_segbads.append(segbads[i])
    return out_paths, out_segbads, chosen[0], worst, -chosen[1]


def botwalk_chain(bsp, poly, segbad):
    track = [[poly[0][0], poly[0][1], poly[0][2] + 16]]
    exempt = []
    for i in range(1, len(poly)):
        a, b = poly[i - 1], poly[i]
        L = math.dist(a, b)
        n = max(2, int(math.ceil(L / 24.0)))
        ex = segbad[i - 1] if segbad else False
        for k in range(1, n + 1):
            f = k / n
            q = [a[t] + f * (b[t] - a[t]) for t in range(3)]
            q[2] += 16
            g = 0
            while bsp.inside(tuple(q)) and g < 16:
                q[2] += 12
                g += 1
            if math.dist(q, track[-1]) < 2 and k < n:
                continue
            track.append(q)
            exempt.append(ex)
    return track, exempt


def nav_tracks(nodes, adj, kcarts, bsp, cor, bad):
    adj_ok = [{v: w for v, w in a.items()
               if (min(u, v), max(u, v)) not in bad} for u, a in enumerate(adj)]
    hosts = host_components(adj_ok, nodes, kcarts)
    slots = {}
    for c, (region, comp) in enumerate(hosts):
        slots.setdefault((region, id(comp)), [region, comp, []])[2].append(c)
    st = {'e0': 0.0, 'e1': 0.0, 'bf': 0.0, 'pruned': 0, 'net': 0, 'tot': 0,
          'regions': [], 'hosts': len(slots)}
    paths = [None] * kcarts
    segbads = [None] * kcarts
    adms = [[]] * kcarts
    ov = [[0.0] * kcarts for _ in range(kcarts)]
    interior_deg, interior_n = 0.0, 0
    for region, comp, cart_ids in slots.values():
        keep, bf, pruned, total = prune_component(adj_ok, comp)
        st['pruned'] += pruned
        st['net'] += len(keep)
        st['tot'] += total
        interior_deg += bf * len(keep)
        interior_n += len(keep)
        st['regions'].append({'region': region, 'carts': list(cart_ids),
                              'net': len(keep), 'tot': total, 'bf': bf})
        lp, lsb, ladm, lov, _ = plan_spans(nodes, adj, adj_ok, bad, len(cart_ids), keep)
        for local, c in enumerate(cart_ids):
            paths[c] = lp[local]
            segbads[c] = lsb[local]
            adms[c] = ladm[local]
        for a, ca in enumerate(cart_ids):
            for b, cb in enumerate(cart_ids):
                ov[ca][cb] = lov[a][b]
    st['bf'] = interior_deg / interior_n if interior_n else 0.0
    paths, segbads, align, worst, spr = flow_assign(nodes, paths, segbads, adj_ok)
    st['ov'] = ov
    st['align'] = align
    st['worst'] = worst
    tracks, exempts = [], []
    for c in range(kcarts):
        poly = [list(nodes[i]) for i in paths[c]]
        segbad = segbads[c]
        if bsp is None or cor is None or len(poly) < 2:
            track = [[q[0], q[1], q[2] + 16] for q in poly] or [[0.0, 0.0, 0.0]]
            while len(track) < 2:
                track.append(track[0][:])
            tracks.append(track)
            exempts.append([True] * (len(track) - 1))
            continue
        R, mark = resample_marked(poly, segbad, 64.0)
        st['e0'] += bending_energy([[r[0], r[1], r[2] + 16] for r in R])
        Q, zf = medial_smooth(bsp, cor, R, mark)
        for i in range(1, len(Q) - 1):
            Q[i] = feasible(bsp, cor, Q[i], mark[i])
        st['e1'] += bending_energy(Q)
        track, exempt = adaptive_nodes(bsp, cor, Q, R, mark)
        track, exempt = polish_chain(bsp, cor, track, exempt)
        if any(not seg_valid(bsp, cor, track[t], track[t + 1], exempt[t]) for t in range(len(track) - 1)):
            track, exempt = botwalk_chain(bsp, poly, segbad)
            st['bw'] = st.get('bw', 0) + 1
        tracks.append(track)
        exempts.append(exempt)
    return tracks, exempts, adms, paths, st


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
    text, resolved = load_cache(mapname, bsp, pk3arg)
    if not text:
        print('nav: no waypoints for %s, FALLBACK to spawn-origin method' % mapname)
        return spawn_tracks(pts, kcarts), None
    nodes, adj = parse_cache(text)
    if not nodes or not largest_component(adj):
        print('nav: degenerate graph for %s (%d nodes), FALLBACK to spawn-origin method' % (mapname, len(nodes)))
        return spawn_tracks(pts, kcarts), None
    db = d if d is not None else open(bsp, 'rb').read()
    try:
        B = Bsp(db)
    except Exception:
        B = None
    flags = load_flags(mapname, resolved, bsp, pk3arg, nodes)
    gen = {i for i, nd in enumerate(nodes) if nd not in flags} if flags else set()
    tboxes = trigger_boxes(db)
    bad = classify_edges(nodes, adj, B, gen, flags, tboxes) if B else {}
    ne = sum(len(a) for a in adj) // 2
    rs = {'gen': 0, 'flag': 0, 'trig': 0, 'solid': 0, 'fly': 0, 'nofloor': 0}
    for r in bad.values():
        rs[r] += 1
    print('nav: %s waypoints=%d saved=%d gen=%d trigboxes=%d links=%d bad=%d (gen=%d flag=%d trig=%d solid=%d fly=%d nofloor=%d)' %
          (mapname, len(nodes), len(flags), len(gen), len(tboxes), ne, len(bad),
           rs['gen'], rs['flag'], rs['trig'], rs['solid'], rs['fly'], rs['nofloor']))
    cor = Corridor(standable_set(nodes, adj, bad, gen)) if B else None
    if cor:
        print('nav: corridor standable=%d push_radius=%.0f push_height=%.0f' % (len(cor.pts), PUSH_R, PUSH_H))
    tracks, exempts, alladm, paths, st = nav_tracks(nodes, adj, kcarts, B, cor, bad)
    print('nav: network kept=%d/%d pruned_dangles=%d branching=%.2f target=1.5 %s' %
          (st['net'], st['tot'], st['pruned'], st['bf'], 'OK' if st['bf'] >= 1.5 else 'BELOW'))
    print('nav: host_components=%d over %d spatial region(s) (link=%.0f)' %
          (st['hosts'], len({r['region'] for r in st['regions']}), REGION_LINK))
    for r in st['regions']:
        print('nav:   region %d carts %s kept=%d/%d branching=%.2f' %
              (r['region'], r['carts'], r['net'], r['tot'], r['bf']))
    ov = st['ov']
    mx = max((ov[i][j] for i in range(kcarts) for j in range(i + 1, kcarts)), default=0.0)
    for i in range(kcarts):
        for j in range(i + 1, kcarts):
            print('nav: overlap %d-%d %.2f' % (i, j, ov[i][j]))
    print('nav: overlap_max=%.2f bound=%.2f %s' % (mx, OVERLAP_MAX, 'HELD' if mx <= OVERLAP_MAX else 'EXCEEDED'))
    print('nav: flow_alignment chosen=%.3f worst=%.3f' % (st['align'], st['worst']))
    origins = [p[0] for p in paths]
    adj_ok2 = [{v: w for v, w in a.items()
                if (min(u, v), max(u, v)) not in bad} for u, a in enumerate(adj)]
    dmo = {o: dijkstra(adj_ok2, o)[0] for o in set(origins)}
    # Bots reach a cart over the STOCK navmesh, which keeps the fusion joins
    # (teleporters, pads) that the cart-track filter drops; cart->cart traversal
    # cost is therefore measured on `adj`, the full waypoint graph.
    dmn = {o: dijkstra(adj, o)[0] for o in set(origins)}
    ws, ns = [], []
    labels = region_labels(nodes)
    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            if origins[i] == origins[j]:
                continue
            wd = dmo[origins[i]][origins[j]]
            nd = dmn[origins[i]][origins[j]]
            ed = math.dist(nodes[origins[i]], nodes[origins[j]])
            if wd < math.inf:
                ws.append(wd)
            if nd < math.inf:
                ns.append(nd)
            print('nav: origin %d r%d %s <-> origin %d r%d %s  navmesh=%.0f  track_walk=%.0f  euclid=%.0f  ratio=%.2f' %
                  (i, labels[origins[i]], tuple(round(x) for x in nodes[origins[i]]),
                   j, labels[origins[j]], tuple(round(x) for x in nodes[origins[j]]),
                   nd if nd < math.inf else -1,
                   wd if wd < math.inf else -1, ed, nd / ed if ed and nd < math.inf else 0))
    if ns:
        print('nav: cart-to-cart navmesh min=%.0f mean=%.0f max=%.0f balance_ratio=%.2f' %
              (min(ns), sum(ns) / len(ns), max(ns), max(ns) / min(ns) if min(ns) else 0))
    if ws:
        print('nav: same-component walk min=%.0f mean=%.0f max=%.0f' %
              (min(ws), sum(ws) / len(ws), max(ws)))
    nadm = sum(len(a) for a in alladm)
    for c, a in enumerate(alladm):
        if a:
            print('nav: cart %d admitted %d bad edges: %s' % (c, len(a), a))
    print('nav: admitted_total=%d weights bend=%.1f w_med=%.2f w_floor=%.2f target=%.0f cap=%.0f outer=%d inner=%d badmult=%.0f' %
          (nadm, WB, WMED, WFLOOR, CLEAR_TARGET, CLEAR_CAP, OUTER, INNER, BADMULT))
    print('nav: curvature %.0f->%.0f (%.1f%%) botwalk_fallback_carts=%d' %
          (st['e0'], st['e1'], 100 * st['e1'] / st['e0'] if st['e0'] else 0, st.get('bw', 0)))
    if B and cor:
        va, vb, vc, vx, ns = 0, 0, 0, 0, 0
        for track, exempt in zip(tracks, exempts):
            r = validate_chain(B, cor, track, exempt)
            va, vb, vc, vx, ns = va + r[0], vb + r[1], vc + r[2], vx + r[3], ns + r[4]
        exs = sum(sum(1 for e in ex if e) for ex in exempts)
        print('nav: validation samples=%d solid_viol=%d float_viol=%d corridor_viol=%d exempt_segs=%d exempt_corridor_viol=%d %s' %
              (ns, va, vb, vc, exs, vx, 'PASS' if va == 0 and vb == 0 and vc == 0 else 'FAILED'))
    else:
        print('nav: validation skipped (no BSP geometry)')
    return tracks, (nodes, adj, paths)


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
