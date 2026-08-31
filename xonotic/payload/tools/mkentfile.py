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


def classify_edges(nodes, adj, ns, gen, flags, tboxes):
    """Which navmesh links a CART may follow.

    DELETED here: `seg_bad`, which walked every link in 24-unit steps asking
    `Bsp.inside` and `Bsp.floor` -- a point sampler over an AABB-from-plane-distance
    grid that had already produced both a 75 GB unguarded indexing loop and phantom
    "no floor" reports along clear corridors.  The geometric half of the question is
    now answered in closed form by `navmesh.Navmesh.classify_edges` against the
    world's COMPUTED free volume; the semantic half (NAV-SPEC §4: a link that encodes
    a jump-pad or teleport trajectory is not a cart segment) is kept and merged."""
    NMOD = _navmesh()
    nm = NMOD.Navmesh(nodes, adj, flags={tuple(round(x, 1) for x in k): v
                                         for k, v in (flags or {}).items()},
                      triggerboxes=tboxes)
    bad = dict(nm.classify_edges(ns)) if ns is not None else {}
    for u in range(len(adj)):
        for v in adj[u]:
            if v <= u:
                continue
            if u in gen or v in gen:
                bad[(u, v)] = 'gen'
            elif (flags.get(nodes[u], 0) | flags.get(nodes[v], 0)) & WPF_BAD:
                bad[(u, v)] = 'flag'
    return bad


def _navmesh():
    import navmesh
    return navmesh


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


DANGLE_MIN, OVERLAP_MAX, PEN0 = 320.0, 0.3, 8.0
REGION_LINK, HOST_MIN, HOST_FRACTION = 700.0, 8, 0.4


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


def nav_tracks(nodes, adj, kcarts, ns, bad, solver=None):
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
    # ---- THE PATH PLACER (NAV-SPEC §2, §3)
    # DELETED here: `medial_smooth` / `feasible` / `refine` / `adaptive_nodes` /
    # `polish_chain` / `seg_valid` / `validate_chain` / `botwalk_chain` -- a stack of
    # point-sampled repair passes over `Bsp.inside`, plus a "bot walk" fallback for
    # when they failed.  They were the shape of the problem: emit a curve, poke at
    # it, patch what the pokes noticed.  What replaces them is a tangent-energy
    # curve optimizer whose ITERATES ARE PROJECTED INTO THE COMPUTED FREE VOLUME,
    # so every intermediate curve -- and therefore the final one -- is a motion plan
    # inside negative space.  A path that satisfies the constraint by construction
    # cannot burrow, which is why there is no unstick and nothing left to validate.
    NMOD = _navmesh()
    if solver is None and ns is not None:
        solver = NMOD.PathSolver(ns)
    tracks, exempts = [], []
    st['infeasible'] = 0
    st['airborne'] = 0
    st['maxdev'] = 0.0
    for c in range(kcarts):
        poly = [list(nodes[i]) for i in paths[c]]
        segbad = segbads[c]
        if solver is None or len(poly) < 2:
            track = [[q[0], q[1], q[2] + 16] for q in poly] or [[0.0, 0.0, 0.0]]
            while len(track) < 2:
                track.append(track[0][:])
            tracks.append(track)
            exempts.append([True] * (len(track) - 1))
            continue
        seed = [[q[0], q[1], q[2] + 16] for q in poly]
        st['e0'] += NMOD.tangent_energy(seed)
        track, ts = solver.solve(seed, pin=(0, len(seed) - 1))
        st['e1'] += ts['e1']
        st['infeasible'] += ts['infeasible']
        st['unplaceable'] = st.get('unplaceable', 0) + ts.get('unplaceable', 0)
        st['airborne'] += ts['airborne']
        st['maxdev'] = max(st['maxdev'], ts['max_activation_distance'])
        if len(track) < 2:
            track = seed
        tracks.append(track)
        exempts.append([bool(segbad and any(segbad))] * (len(track) - 1))
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


def build_tracks(bsp, mapname, pts, kteams, kcarts, pk3arg='', d=None, ns=None):
    text, resolved = load_cache(mapname, bsp, pk3arg)
    if not text:
        print('nav: no waypoints for %s, FALLBACK to spawn-origin method' % mapname)
        return spawn_tracks(pts, kcarts), None
    nodes, adj = parse_cache(text)
    if not nodes or not largest_component(adj):
        print('nav: degenerate graph for %s (%d nodes), FALLBACK to spawn-origin method' % (mapname, len(nodes)))
        return spawn_tracks(pts, kcarts), None
    db = d if d is not None else open(bsp, 'rb').read()
    if ns is None:
        # standalone use on a single map: its own BSP tree covers the whole world,
        # so the free volume can be computed straight from it
        import negspace as _NS
        ns = _NS.NegSpace(db, mask=_NS.MASK_PLAYERSOLID)
    flags = load_flags(mapname, resolved, bsp, pk3arg, nodes)
    gen = {i for i, nd in enumerate(nodes) if nd not in flags} if flags else set()
    tboxes = trigger_boxes(db)
    bad = classify_edges(nodes, adj, ns, gen, flags, tboxes)
    ne = sum(len(a) for a in adj) // 2
    rs = {'gen': 0, 'flag': 0, 'semantic': 0, 'burrow': 0, 'airborne': 0}
    for r in bad.values():
        rs[r] = rs.get(r, 0) + 1
    print('nav: %s waypoints=%d saved=%d gen=%d trigboxes=%d links=%d not-cart-segments=%d '
          '(unsaved=%d flagged=%d semantic_jump/teleport=%d burrows_through_solid=%d '
          'floats_over_non-walkable=%d)' %
          (mapname, len(nodes), len(flags), len(gen), len(tboxes), ne, len(bad),
           rs['gen'], rs['flag'], rs['semantic'], rs['burrow'], rs['airborne']))
    tracks, exempts, alladm, paths, st = nav_tracks(nodes, adj, kcarts, ns, bad)
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
    ws, nsd = [], []
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
                nsd.append(nd)
            print('nav: origin %d r%d %s <-> origin %d r%d %s  navmesh=%.0f  track_walk=%.0f  euclid=%.0f  ratio=%.2f' %
                  (i, labels[origins[i]], tuple(round(x) for x in nodes[origins[i]]),
                   j, labels[origins[j]], tuple(round(x) for x in nodes[origins[j]]),
                   nd if nd < math.inf else -1,
                   wd if wd < math.inf else -1, ed, nd / ed if ed and nd < math.inf else 0))
    if nsd:
        print('nav: cart-to-cart navmesh min=%.0f mean=%.0f max=%.0f balance_ratio=%.2f' %
              (min(nsd), sum(nsd) / len(nsd), max(nsd), max(nsd) / min(nsd) if min(nsd) else 0))
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
    # ---- VORONOI OVER THE NAVMESH (NAV-SPEC §2, §8) and the equidistance of the
    # cart origins in navmesh WALKING distance (NAV-SPEC §1).  Both are computed
    # over the stock waypoint graph -- the one navigation definition (§5) -- and
    # the free volume, so the numbers below are properties of the structure the
    # path placer actually consumed.
    # OFF BY DEFAULT, and the reason is a measurement, not a preference.
    # The Voronoi decomposition needs `negspace.build_portals`, which is exact but
    # quadratic in the number of cells sharing one plane.  The soundness fix that
    # stopped the complex calling brush interiors free multiplied cell counts about
    # 2.5x (dance 3 794 -> 9 384), and that pushed portal construction from seconds
    # to longer than an entity emission has any business taking -- it is what left a
    # 2-tile world's emit stage running for an hour.  Nothing that SHIPS depends on
    # it: spawn placement, the cart-path solver, the doorway and connector checks
    # and fusecheck all use cell membership, coverage and segment intervals, none of
    # which touch portals.  So it is opt-in (FUSE_VORONOI=1) until the pairing has a
    # real 2-D index inside each plane.
    if not os.environ.get('FUSE_VORONOI'):
        print('nav: Voronoi-over-navmesh + k-center equidistance pass SKIPPED '
              '(set FUSE_VORONOI=1 to run it; it needs portal construction, which '
              'is quadratic in cells-per-plane and is the slowest thing in this '
              'module -- nothing that ships depends on it)')
    else:
      try:
          NMOD = _navmesh()
          nmv = NMOD.Navmesh(nodes, adj,
                             flags={tuple(round(x, 1) for x in k): v for k, v in (flags or {}).items()},
                             triggerboxes=tboxes)
          nmv.bad = bad
          origins0 = [p[0] for p in paths]
          owner = nmv.voronoi(ns, sites=origins0, verbose=True)
          import numpy as _np
          vol = _np.maximum(ns.hi - ns.lo, 0.0)
          vol = vol[:, 0] * vol[:, 1] * vol[:, 2]
          for si, o in enumerate(origins0):
              m = owner == si
              print('nav: voronoi cell of cart %d (origin wp %d): %d free cells, '
                    '%.4g u^3 of navigable free volume' % (si, o, int(m.sum()), float(vol[m].sum())))
          pool = [i for i in range(len(nodes)) if any(True for _ in adj[i])]
          opt, ost = nmv.equidistant_origins(max(3, kcarts), pool=pool, verbose=True)
          D0 = {o: nmv.walk_dist(o)[0] for o in set(origins0)}
          vals = [D0[a][b] for i, a in enumerate(origins0) for b in origins0[i + 1:]
                  if D0[a][b] < math.inf and a != b]
          if vals:
              print('nav: cart origins as PLACED: pairwise navmesh-walking min=%.0f max=%.0f '
                    'spread_ratio=%.2f; the k-center optimum over this navmesh is %.2f '
                    '(1.00 = exactly equidistant, NAV-SPEC §1)'
                    % (min(vals), max(vals), max(vals) / min(vals) if min(vals) else 0.0,
                       ost.get('ratio', 0.0)))
      except Exception as e:
        print('nav: voronoi/equidistance pass unavailable: %r' % (e,))

    # The path placer's constraint is the free volume itself, so what is worth
    # printing is whether the constraint HELD, not the result of a second sampler.
    print('nav: cart paths %d over %d nodes: %d points off the computed free volume '
          '(0 = the plan is inside negative space everywhere), of which %d have no '
          'legal cart placement within reach at all; %d points with no walkable floor '
          'beneath; max activation distance to negative space %.2fu -- %s'
          % (len(tracks), sum(len(t) for t in tracks), st.get('infeasible', 0),
             st.get('unplaceable', 0), st.get('airborne', 0), st.get('maxdev', 0.0),
             'CONSTRAINT HELD' if st.get('infeasible', 0) == 0 else 'CONSTRAINT VIOLATED'))
    return tracks, (nodes, adj, paths)


def emit(bsp, out, kteams, kcarts, pk3arg='', ns=None):
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

    tracks, _ = build_tracks(bsp, mapname, pts, kteams, kcarts, pk3arg, d, ns=ns)

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
            if 0 < i < NC - 1 and i % max(1, NC // 4) == 0:  # every Nth interior node only: an unconditional checkpoint spawns a WaypointSprite per node (sv_payload.qc:475) -> unreadable label stack
                e.append('"spawnflags" "1"')
            e.append('}')
            extra.append('\n'.join(e))
        extra.append('\n'.join(['{', '"classname" "func_plc_cart"',
                                '"model" "%s"' % visible[c % len(visible)],
                                '"target" "%s"' % names[0], '"speed" "40"', '}']))

    # `plc_goal "cnt"` is a TEAM INDEX MINUS ONE, not a team id. sv_payload.qc:821
    #     this.team = Team_IndexToTeam(this.cnt + 1);
    # and teams.qh:185 maps index 1..5 -> NUM_TEAM_1..5 = 5, 14, 13, 10, 4
    # (red, blue, yellow, pink, GREEN). So `cnt = t` for t in 0..kteams-1 is the
    # encoding, it reaches green at t=4, and it imposes no ceiling here.
    #
    # I got this wrong three times and each time by not reading this consumer.
    # (1) deleted the old `[4, 13, 12, 9, 3]` table as "magic", emitting values
    # that resolved to no team; (2) "restored" it as a FOUR-entry list plus an
    # error above four teams, claiming stock Xonotic has no fifth team -- made
    # up: NUM_TEAMS is 5 and a green cart lane is visible in the running game;
    # (3) tried to restore the five-entry table, which would have SCRAMBLED the
    # lanes -- [4,13,12,9,3] is dom's raw-team-id convention, and this gamemode
    # does its own index->id mapping. The table was never right for this caller.
    #
    # The real limit is Team_IndexToTeam returning -1 above index 5, i.e. in
    # teams.qh, extensible by adding NUM_TEAM_6.. and cases -- which is what the
    # earlier 8-team prototypes did. It is not a barrier and not enforced here.
    for t in range(kteams):
        names, track = named[t % kcarts]
        extra.append('\n'.join(['{', '"classname" "plc_goal"',
                                '"cnt" "%d"' % t, '"target" "%s"' % names[-1],
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
    # No upper clamp: the design target is ~256 playerbots across 5+ teams, and a
    # hardcoded ceiling of 5 silently made that target unproducible.
    kteams = max(2, int(sys.argv[3])) if len(sys.argv) > 3 else 2
    kcarts = max(1, int(sys.argv[4])) if len(sys.argv) > 4 else 2
    pk3arg = sys.argv[5] if len(sys.argv) > 5 else ''
    emit(bsp, out, kteams, kcarts, pk3arg)
