import struct, sys, re, math, os, glob, subprocess, heapq, time, json, zlib

import numpy as np
import negspace as _NS
from strategy_io_schema import MAP_MEASUREMENT_SCHEMA
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from rdma.workload import WorkloadMeter

CLEAR_TARGET, CLEAR_CAP, WB, WMED, WFLOOR, WHEEL = 64.0, 256.0, 1.0, 0.3, 1.0, 42.0
FLOAT_LIM, EPS, CELL = 96.0, 0.25, 512.0
WPF_BAD = (1 << 21) | (1 << 15) | (1 << 14) | (1 << 13)

def push_cvars():
    r, h, s, capture = 160.0, 96.0, 30.0, 64.0
    error = None
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfg', 'gamemodes-payload.cfg')
    try:
        for line in open(cfg):
            m = re.match(r'\s*set\s+g_payload_(push_radius|push_height|speed|capture_radius)\s+(\S+)', line)
            if m:
                if m.group(1) == 'push_radius':
                    r = float(m.group(2))
                elif m.group(1) == 'push_height':
                    h = float(m.group(2))
                elif m.group(1) == 'speed':
                    s = float(m.group(2))
                else:
                    capture = float(m.group(2))
    except Exception as exc:
        error = '%s: %s' % (type(exc).__name__, exc)
    return r, h, s, capture, error

PUSH_R, PUSH_H, PAYLOAD_SPEED, CAPTURE_RADIUS, PUSH_CONFIGURATION_ERROR = push_cvars()
CART_RIDE = -_NS.CART_MIN[2]
SPAWN_APPROACH_SPEED = 320.0
PATH_MIN = max(PUSH_R, CAPTURE_RADIUS) + EPS
CART_ORIGIN_SEP = max(
    _NS.CART_MAX[0] - _NS.CART_MIN[0],
    _NS.CART_MAX[1] - _NS.CART_MIN[1],
)
PLAYER_SPAWN_CLASSES = {
    'info_player_deathmatch', 'info_player_start', 'info_player_survivor',
    'info_player_race', 'info_player_attacker', 'info_player_defender',
    'team_CTF_redplayer', 'team_CTF_redspawn', 'team_CTF_blueplayer',
    'team_CTF_bluespawn', 'team_redplayer', 'team_blueplayer',
}
NEUTRAL_SPAWN_CLASSES = {
    'info_player_deathmatch', 'info_player_start', 'info_player_survivor',
    'info_player_race',
}

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
    seen = []
    while name not in seen:
        seen.append(name)
        text = pk3_read(name + '.waypoints.cache', bsp, pk3arg)
        stripped = text.strip()
        if stripped.endswith('.waypoints.cache') and '*' not in stripped and '\n' not in stripped:
            name = stripped[:-len('.waypoints.cache')]
            continue
        return text, name
    print('nav: cache_alias_cycle_mass=1 aliases=%s' % ','.join(seen), file=sys.stderr)
    return text, name

def parse_waypoints(text):
    flags = {}
    parse_error_mass = 0
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith('//')]
    for i in range(0, len(lines) - 2, 3):
        try:
            a = [float(x) for x in lines[i].strip().strip("'").split()]
            b = [float(x) for x in lines[i + 1].strip().strip("'").split()]
            fl = int(float(lines[i + 2].strip()))
        except ValueError:
            parse_error_mass += 1
            continue
        if len(a) == 3 and len(b) == 3:
            flags[tuple(round((a[k] + b[k]) / 2, 1) for k in range(3))] = fl
        else:
            parse_error_mass += 1
    parse_error_mass += len(lines) % 3
    if parse_error_mass:
        print('nav: waypoint_parse_error_mass=%d source_line_mass=%d' %
              (parse_error_mass, len(lines)), file=sys.stderr)
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
    except Exception as exc:
        print('nav: trigger_box_parse_error_mass=1 error=%s:%s' %
              (type(exc).__name__, exc), file=sys.stderr)
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
    parse_error_mass = 0

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
            parse_error_mass += 1
            continue
        try:
            a = nid(parts[0].strip().strip("'"))
            b = nid(parts[1].strip().strip("'"))
        except ValueError:
            parse_error_mass += 1
            continue
        w = math.dist(nodes[a], nodes[b])
        adj[a][b] = min(adj[a].get(b, 1e18), w)
        adj[b][a] = min(adj[b].get(a, 1e18), w)
    if parse_error_mass:
        print('nav: cache_parse_error_mass=%d source_line_mass=%d' %
              (parse_error_mass, len(text.splitlines())), file=sys.stderr)
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

def classify_edges(nodes, adj, ns, flags, tboxes):
    NMOD = _navmesh()
    nm = NMOD.Navmesh(nodes, adj, flags={tuple(round(x, 1) for x in k): v
                                         for k, v in (flags or {}).items()},
                      triggerboxes=tboxes)
    cart_incompatible = dict(nm.classify_edges(ns)) if ns is not None else {}
    for u in range(len(adj)):
        for v in adj[u]:
            if v <= u:
                continue
            if (flags.get(nodes[u], 0) | flags.get(nodes[v], 0)) & WPF_BAD:
                cart_incompatible[(u, v)] = 'flag'
    return cart_incompatible

def _navmesh():
    import navmesh
    return navmesh

OVERLAP_MAX, PEN0 = 0.3, 8.0
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

def component_path_horizon(adj_ok, comp):
    horizon = 0.0
    for source in comp:
        distance, _ = dijkstra(adj_ok, source)
        horizon = max(horizon, max(distance[node] for node in comp))
        if horizon >= PATH_MIN:
            break
    return horizon

def cart_path_components(adj_ok):
    candidates = []
    for comp in components(adj_ok):
        if len(comp) < 2:
            continue
        if component_path_horizon(adj_ok, comp) >= PATH_MIN:
            candidates.append(comp)
    return candidates

def host_components(adj_ok, metric_adj, kcarts):
    candidates = cart_path_components(adj_ok)
    if not candidates:
        candidates = components(adj_ok)[:1]
    owner = {node: index for index, comp in enumerate(candidates) for node in comp}
    pool = set(owner)
    metric_groups = [sorted(pool.intersection(comp)) for comp in components(metric_adj)]
    metric_groups = [group for group in metric_groups if group]
    metric_pool = max(metric_groups, key=lambda group: (len(group), -min(group)))
    distance = {}
    distance[metric_pool[0]] = dijkstra(metric_adj, metric_pool[0])[0]
    first = max(metric_pool, key=lambda node: distance[metric_pool[0]][node])
    picks = [first]
    distance[first] = dijkstra(metric_adj, first)[0]
    while len(picks) < min(kcarts, len(metric_pool)):
        node = max(
            (value for value in metric_pool if value not in picks),
            key=lambda value: min(distance[pick][value] for pick in picks),
        )
        picks.append(node)
        distance[node] = dijkstra(metric_adj, node)[0]
    order = list(dict.fromkeys(owner[node] for node in picks))
    order.extend(index for index in range(len(candidates)) if index not in order)
    hosts = [(index, candidates[index]) for index in order]
    while hosts and len(hosts) < kcarts:
        hosts.extend(hosts[:kcarts - len(hosts)])
    return hosts[:kcarts], {
        'candidate_component_mass': len(candidates),
        'candidate_node_mass': len(pool),
        'metric_component_mass': len(metric_groups),
        'metric_pool_mass': len(metric_pool),
        'metric_disconnected_candidate_mass': len(pool) - len(metric_pool),
        'selected_component_mass': len(set(index for index, _ in hosts[:kcarts])),
    }

def component_branching(adj_ok, comp):
    keep = set(comp)

    def deg(u):
        return sum(1 for v in adj_ok[u] if v in keep)

    interior = [u for u in keep if deg(u) >= 2]
    bf = (sum(deg(u) for u in interior) / len(interior) - 1.0) if interior else 0.0
    return keep, bf, len(comp)

def subgraph(adj_ok, keep):
    return [{v: w for v, w in a.items() if u in keep and v in keep} for u, a in enumerate(adj_ok)]

def extract_path(prev, src, goal):
    p, u = [], goal
    while u != -1:
        p.append(u)
        u = prev[u]
    return p[::-1] if len(p) > 1 and p[-1] != p[0] else [src]

def partition_spans(nodes, adjn, kcarts, comp):
    if len(comp) < 2 * kcarts:
        return None
    seeds, _ = kcenter(adjn, kcarts, comp)
    dist = [math.inf] * len(nodes)
    owner = [-1] * len(nodes)
    pq = []
    for c, seed in enumerate(seeds):
        dist[seed] = 0.0
        owner[seed] = c
        heapq.heappush(pq, (0.0, c, seed))
    while pq:
        d, c, u = heapq.heappop(pq)
        if d > dist[u] or c != owner[u]:
            continue
        for v, w in adjn[u].items():
            nd = d + w
            if nd < dist[v] - 1e-9 or (abs(nd - dist[v]) <= 1e-9 and c < owner[v]):
                dist[v] = nd
                owner[v] = c
                heapq.heappush(pq, (nd, c, v))
    paths = []
    for c, seed in enumerate(seeds):
        D = [math.inf] * len(nodes)
        prev = [-1] * len(nodes)
        D[seed] = 0.0
        pq = [(0.0, seed)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > D[u]:
                continue
            for v, w in adjn[u].items():
                if owner[v] != c:
                    continue
                nd = d + w
                if nd < D[v]:
                    D[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        goal = max((u for u in comp if owner[u] == c),
                   key=lambda u: math.dist(nodes[seed], nodes[u]))
        if math.dist(nodes[seed], nodes[goal]) < PATH_MIN:
            return None
        paths.append(extract_path(prev, seed, goal))
    overlap = [[0.0] * kcarts for _ in range(kcarts)]
    return paths, overlap, seeds

def plan_spans(nodes, adj_ok, kcarts, keep):
    adjn = subgraph(adj_ok, keep)
    comp = sorted(keep)
    partitioned = partition_spans(nodes, adjn, kcarts, comp)
    if partitioned is not None:
        return partitioned
    ncand = min(max(3, 2 * kcarts), len(comp))
    if ncand < 2:
        only = comp[0] if comp else 0
        return [[only]] * kcarts, [[0.0] * kcarts for _ in range(kcarts)], [only] * kcarts
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

    pairings = [greedy(sorted(order)), greedy(order)]
    if len(order) > 1:
        pairings.append(greedy(order[1:] + order[:1]))
    p0 = greedy(order)
    if len(p0) >= 2:
        pairings.append([(p0[i][0], p0[(i + 1) % len(p0)][1]) for i in range(len(p0))])
    seen = set()
    pairings = [pr for pr in pairings if not (tuple(pr) in seen or seen.add(tuple(pr)))]
    best = None
    for pairs in pairings:
        P = PEN0
        seen_paths = set()
        while True:
            npen = [0] * len(nodes)
            paths = []
            for a, b in pairs:
                adjp = [{v: w * (1 + P * (npen[u] + npen[v]) / 2) for v, w in adjn[u].items()}
                        for u in range(len(nodes))]
                _, prev = dijkstra(adjp, a)
                p = extract_path(prev, a, b)
                for n in p:
                    npen[n] += 1
                paths.append(p)
            ov = [[0.0] * kcarts for _ in range(kcarts)]
            mx = 0.0
            for i in range(kcarts):
                for j in range(i + 1, kcarts):
                    si, sj = set(paths[i]), set(paths[j])
                    f = len(si & sj) / max(1, min(len(si), len(sj)))
                    ov[i][j] = ov[j][i] = f
                    mx = max(mx, f)
            if best is None or mx < best[0]:
                best = (mx, paths, ov)
            if mx <= OVERLAP_MAX:
                break
            signature = tuple(tuple(path) for path in paths)
            if signature in seen_paths:
                break
            seen_paths.add(signature)
            P *= 8
        if best[0] <= OVERLAP_MAX:
            break
    mx, paths, ov = best

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

    while mx > OVERLAP_MAX:
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
            fixed = max((ov[i][j] for i in range(kcarts) for j in range(i + 1, kcarts)
                         if i != ri and j != ri), default=0.0)
            others = [(j, set(paths[j])) for j in range(kcarts) if j != ri]
            reserved = {paths[j][q] for j in range(kcarts) if j != ri for q in (0, -1)}
            for d, a2, b2 in order:
                if a2 in reserved or b2 in reserved:
                    continue
                if a2 not in dcache:
                    dcache[a2] = dijkstra(adjb, a2)
                D, prev = dcache[a2]
                if D[b2] == math.inf:
                    continue
                np_ = extract_path(prev, a2, b2)
                sn = set(np_)
                row = [(j, len(sn & sj) / max(1, min(len(sn), len(sj))))
                       for j, sj in others]
                nmx = max([fixed] + [f for j, f in row])
                if nmx < mx and (bestalt is None or nmx < bestalt[0]):
                    bestalt = (nmx, np_, row)
            if bestalt is None:
                break
            nmx, np_, row = bestalt
            paths = paths[:ri] + [np_] + paths[ri + 1:]
            for j, f in row:
                ov[ri][j] = ov[j][ri] = f
            mx = nmx
    return paths, ov, None

def flow_assign(nodes, paths, adj_ok, metric_adj=None):
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
    coupling = [[0.0] * k for _ in range(k)]
    den = 0.0
    buckets = {}
    for i, segs in enumerate(segsets):
        for mi, ti in segs:
            cell = tuple(int(math.floor(x / rad)) for x in mi)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j, mj, tj in buckets.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                            if i == j:
                                continue
                            d = math.dist(mi, mj)
                            if d <= rad:
                                w = 1 - d / rad
                                a, b = sorted((i, j))
                                coupling[a][b] += w * sum(ti[t] * tj[t] for t in range(3))
                                den += w
            buckets.setdefault(cell, []).append((i, mi, ti))

    def score(flips):
        num = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                fi = -1.0 if flips >> i & 1 else 1.0
                fj = -1.0 if flips >> j & 1 else 1.0
                num += coupling[i][j] * fi * fj
        return num / den if den else 0.0

    def headon(flips):
        num = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                fi = -1.0 if flips >> i & 1 else 1.0
                fj = -1.0 if flips >> j & 1 else 1.0
                num += max(0.0, -coupling[i][j] * fi * fj)
        return num / den if den else 0.0

    def descend(flips):
        while True:
            signs = [-1.0 if flips >> i & 1 else 1.0 for i in range(k)]
            best_delta, best_i = -1e-9, -1
            for i in range(k):
                field = sum(coupling[min(i, j)][max(i, j)] * signs[j]
                            for j in range(k) if j != i)
                delta = -2 * signs[i] * field
                if delta < best_delta:
                    best_delta, best_i = delta, i
            if best_i < 0:
                return flips
            flips ^= 1 << best_i

    dm = {}
    walking_graph = metric_adj if metric_adj is not None else adj_ok

    def spread(flips):
        orgs = [paths[i][-1] if flips >> i & 1 else paths[i][0] for i in range(k)]
        for o in set(orgs):
            if o not in dm:
                dm[o] = dijkstra(walking_graph, o)[0]
        ds = [dm[orgs[i]][orgs[j]] for i in range(k) for j in range(i + 1, k) if orgs[i] != orgs[j]]
        ds = [d for d in ds if d < math.inf]
        return min(ds) if ds else 0.0

    whole = (1 << k) - 1
    starts = {0, whole}
    starts.update(1 << i for i in range(k))
    starts.update(whole ^ (1 << i) for i in range(k))
    minima = {descend(f) for f in starts}
    candidates = starts | minima | {f ^ whole for f in minima}
    scored = sorted((round(headon(f), 3), -round(score(f), 3), -spread(f), f)
                    for f in candidates)
    chosen = scored[0]
    worst = min(-s[1] for s in scored)
    flips = chosen[3]
    out_paths = []
    for i in range(k):
        if flips >> i & 1:
            out_paths.append(paths[i][::-1])
        else:
            out_paths.append(paths[i])
    directed = [{(path[i - 1], path[i]) for i in range(1, len(path))} for path in out_paths]
    reverse = sum(len(directed[i] & {(b, a) for a, b in directed[j]})
                  for i in range(k) for j in range(i + 1, k))
    exact_headon = reverse / max(1, sum(len(row) for row in directed))
    return out_paths, {
        'flow_alignment': -chosen[1],
        'candidate_worst_alignment': worst,
        'origin_walking_separation': -chosen[2],
        'headon': exact_headon,
        'push_zone_counterflow': chosen[0],
    }

def track_segment_ok(a, b, solver):
    return solver.segment_feasible(a, b)

def compress_track(track, solver):
    if solver is None or len(track) < 3:
        return track
    out = [track[0]]
    i = 0
    while i + 1 < len(track):
        best = i + 1
        for j in range(i + 2, len(track)):
            if track_segment_ok(track[i], track[j], solver):
                best = j
        out.append(track[best])
        i = best
    return out

def track_length(track):
    return sum(math.dist(track[i], track[i + 1]) for i in range(len(track) - 1))

def cart_speed_scales(tracks, approach_distances=None):
    lengths = [track_length(track) for track in tracks]
    approaches = list(approach_distances) if approach_distances is not None else [0.0] * len(lengths)
    approach_times = [None if distance is None or not math.isfinite(distance)
                      else distance / SPAWN_APPROACH_SPEED for distance in approaches]
    raw = [length / PAYLOAD_SPEED + approach for length, approach in zip(lengths, approach_times)
           if approach is not None]
    finite_approaches = [value for value in approach_times if value is not None]
    target = max(sum(raw) / max(len(raw), 1),
                 max(finite_approaches, default=0.0) + min(lengths, default=0.0) / PAYLOAD_SPEED)
    available = [None if approach is None else max(EPS, target - approach)
                 for approach in approach_times]
    scales = [1.0 if seconds is None else length / (PAYLOAD_SPEED * seconds)
              for length, seconds in zip(lengths, available)]
    traversal_times = [length / max(PAYLOAD_SPEED * scale, EPS) for length, scale in zip(lengths, scales)]
    end_to_end = [None if approach is None else approach + traversal
                  for approach, traversal in zip(approach_times, traversal_times)]
    finite_end_to_end = [value for value in end_to_end if value is not None]
    ratio = (max(finite_end_to_end) / max(min(finite_end_to_end), EPS)
             if finite_end_to_end else None)
    return lengths, scales, approach_times, traversal_times, end_to_end, ratio

def trim_track(track, distance):
    for i in range(len(track) - 1):
        length = math.dist(track[i], track[i + 1])
        if distance <= length:
            fraction = distance / length if length else 0.0
            point = [track[i][axis] + fraction * (track[i + 1][axis] - track[i][axis]) for axis in range(3)]
            return [point] + [list(value) for value in track[i + 1:]]
        distance -= length
    return [list(track[-1])]

def swizzle_track_origins(tracks, solver=None):
    out, origins = [], []
    for track in tracks:
        choices = []
        length = track_length(track)
        steps = int(max(0.0, length - PATH_MIN) // CART_ORIGIN_SEP)
        for step in range(steps + 1):
            trimmed = trim_track(track, step * CART_ORIGIN_SEP)
            separation = min((math.dist(trimmed[0], point) for point in origins), default=math.inf)
            if separation >= CART_ORIGIN_SEP and track_length(trimmed) >= PATH_MIN:
                choices.append((track_length(trimmed), separation, -step, trimmed))
        if not choices and solver is not None and len(track) > 1:
            for radius in (CART_ORIGIN_SEP, 2 * CART_ORIGIN_SEP, 3 * CART_ORIGIN_SEP):
                for slot in range(8):
                    angle = slot * math.pi / 4
                    point = [track[0][0] + radius * math.cos(angle),
                             track[0][1] + radius * math.sin(angle), track[0][2]]
                    floor = solver.ns.floor_under(point, 96.0, footprint=(32.0, 32.0))
                    if floor is None:
                        continue
                    point[2] = floor + solver.ride
                    shifted = [point] + [list(value) for value in track[1:]]
                    separation = min((math.dist(point, value) for value in origins), default=math.inf)
                    if separation >= CART_ORIGIN_SEP and track_length(shifted) >= PATH_MIN and track_segment_ok(shifted[0], shifted[1], solver):
                        choices.append((track_length(shifted), separation, -slot, shifted))
                        break
                if choices:
                    break
        if choices:
            chosen = max(choices)[-1]
            out.append(chosen)
            origins.append(chosen[0])
    return out

def nav_tracks(nodes, adj, kcarts, ns, cart_incompatible, solver=None):
    started = time.monotonic()
    adj_ok = [{v: w for v, w in a.items()
               if (min(u, v), max(u, v)) not in cart_incompatible}
              for u, a in enumerate(adj)]
    if len(largest_component(adj_ok)) < 2:
        print('nav: cart-traversable connected edge mass=0; using negative-space construction domain')
    hosts, host_measures = host_components(adj_ok, adj, kcarts)
    slots = {}
    for c, (component_index, comp) in enumerate(hosts):
        slots.setdefault((component_index, id(comp)), [component_index, comp, []])[2].append(c)
    st = {'e0': 0.0, 'e1': 0.0, 'bf': 0.0, 'net': 0, 'tot': 0,
          'path_components': [], 'hosts': len(slots), 'host_measures': host_measures}
    paths = [None] * kcarts
    ov = [[0.0] * kcarts for _ in range(kcarts)]
    interior_deg, interior_n = 0.0, 0
    for component_index, comp, cart_ids in slots.values():
        keep, bf, total = component_branching(adj_ok, comp)
        st['net'] += len(keep)
        st['tot'] += total
        interior_deg += bf * len(keep)
        interior_n += len(keep)
        st['path_components'].append({'component': component_index, 'carts': list(cart_ids),
                                      'net': len(keep), 'tot': total, 'bf': bf})
        lp, lov, _ = plan_spans(nodes, adj_ok, len(cart_ids), keep)
        for local, c in enumerate(cart_ids):
            paths[c] = lp[local]
        for a, ca in enumerate(cart_ids):
            for b, cb in enumerate(cart_ids):
                ov[ca][cb] = lov[a][b]
    print('nav: span planning %.2fs' % (time.monotonic() - started))
    st['bf'] = interior_deg / interior_n if interior_n else 0.0
    paths, flow_measures = flow_assign(nodes, paths, adj_ok, metric_adj=adj)
    print('nav: flow assignment %.2fs' % (time.monotonic() - started))
    st['ov'] = ov
    st['align'] = flow_measures['flow_alignment']
    st['worst'] = flow_measures['candidate_worst_alignment']
    st['origin_walking_separation'] = flow_measures['origin_walking_separation']
    st['headon'] = flow_measures['headon']
    st['push_counterflow'] = flow_measures['push_zone_counterflow']

    NMOD = _navmesh()
    if solver is None and ns is not None:
        solver = NMOD.PathSolver(ns)
    st['solver'] = solver
    tracks = []
    st['infeasible'] = 0
    st['airborne'] = 0
    st['maxdev'] = 0.0
    for c in range(kcarts):
        poly = [list(nodes[i]) for i in paths[c]]
        if solver is None or len(poly) < 2:
            track = [[q[0], q[1], q[2]] for q in poly] or [[0.0, 0.0, 0.0]]
            while len(track) < 2:
                track.append(track[0][:])
            tracks.append(track)
            continue
        seed = [[q[0], q[1], q[2]] for q in poly]
        st['e0'] += NMOD.tangent_energy(seed)
        track, ts = solver.solve(seed, pin=(0, len(seed) - 1))
        st['e1'] += ts['e1']
        st['infeasible'] += ts['infeasible']
        st['unplaceable'] = st.get('unplaceable', 0) + ts.get('unplaceable', 0)
        st['airborne'] += ts['airborne']
        st['maxdev'] = max(st['maxdev'], ts['max_activation_distance'])
        if len(track) < 2:
            continue
        st['raw_nodes'] = st.get('raw_nodes', 0) + len(track)
        track = compress_track(track, solver)
        tracks.append(track)
        if (c + 1) % 8 == 0 or c + 1 == kcarts:
            print('nav: solved %d/%d carts %.2fs' % (c + 1, kcarts, time.monotonic() - started))
    return tracks, paths, st

def spawn_tracks(pts, kcarts, ns):
    NMOD = _navmesh()
    solver = NMOD.PathSolver(ns)
    points = [list(point) for point in sorted(set(tuple(point) for point in pts))]
    cell_extent = np.maximum(ns.hi - ns.lo, 0.0)
    cell_volume = cell_extent[:, 0] * cell_extent[:, 1] * cell_extent[:, 2]
    cell_order = np.argsort(cell_volume)[::-1]
    required_points = max(2, 2 * kcarts)
    for index in cell_order:
        point = ns.standing_point((ns.lo[index] + ns.hi[index]) * 0.5)
        if point is not None:
            points.append(list(point))
        if len(set(tuple(value) for value in points)) >= required_points:
            break
    points = [list(point) for point in spread_points(points, max(required_points, len(pts)))]
    grounded = {}

    def ground(index):
        if index not in grounded:
            point = points[index]
            floor = ns.floor_under(point, 512.0, footprint=(32.0, 32.0))
            grounded[index] = None if floor is None else [point[0], point[1], floor + CART_RIDE]
        return grounded[index]

    coords = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    firsts, seconds = np.triu_indices(len(coords), 1)
    distances = np.linalg.norm(coords[firsts] - coords[seconds], axis=1)
    distance_domain = np.nonzero(distances >= PATH_MIN)[0]
    order = distance_domain[np.argsort(np.abs(distances[distance_domain] - 8 * PATH_MIN))]
    tracks, origins = [], []
    candidate_segment_mass = 0
    for index in order:
        first, second = int(firsts[index]), int(seconds[index])
        a = ground(first)
        b = ground(second)
        if a is None or b is None:
            continue
        track = [a, b]
        if min((math.dist(track[0], point) for point in origins), default=math.inf) < CART_ORIGIN_SEP:
            continue
        candidate_segment_mass += 1
        if not track_segment_ok(track[0], track[1], solver):
            continue
        tracks.append(track)
        origins.append(track[0])
        if len(tracks) == kcarts:
            break
    if len(tracks) < kcarts and len(points) > 1:
        if ns.adj is None:
            ns.build_portals()
        portal_radius = max(abs(solver.mins[0]), abs(solver.mins[1]),
                            abs(solver.maxs[0]), abs(solver.maxs[1]))

        def portal_polyline(a, b):
            source = ns.cell_at(a)
            target = ns.cell_at(b)
            if source < 0 or target < 0:
                return None
            queue = [source]
            previous = {source: (None, None)}
            cursor = 0
            while cursor < len(queue) and target not in previous:
                cell = queue[cursor]
                cursor += 1
                for neighbor, portal_index in ns.adj[cell]:
                    if neighbor in previous or ns.portals[portal_index].radius < portal_radius:
                        continue
                    previous[neighbor] = (cell, portal_index)
                    queue.append(neighbor)
            if target not in previous:
                return None
            portals = []
            cell = target
            while previous[cell][0] is not None:
                cell, portal_index = previous[cell]
                portals.append(portal_index)
            return [a] + [ns.portals[index].centre for index in reversed(portals)] + [b]

        for index in order:
            if len(tracks) == kcarts:
                break
            first, second = int(firsts[index]), int(seconds[index])
            a = ground(first)
            b = ground(second)
            if a is None or b is None:
                continue
            if min((math.dist(a, point) for point in origins), default=math.inf) < CART_ORIGIN_SEP:
                continue
            polyline = portal_polyline(a, b)
            if polyline is None:
                continue
            track, _ = solver.solve(polyline, pin=(0, len(polyline) - 1))
            if track_length(track) < PATH_MIN:
                continue
            if any(not track_segment_ok(track[i], track[i + 1], solver)
                   for i in range(len(track) - 1)):
                continue
            tracks.append(track)
            origins.append(track[0])
    for index in range(len(points)):
        if len(tracks) == kcarts:
            break
        start = ground(index)
        if start is None:
            continue
        if min((math.dist(start, point) for point in origins), default=math.inf) < CART_ORIGIN_SEP:
            continue
        found = None
        for radius in (1024.0, 768.0, 512.0, 384.0, 256.0, 192.0, PATH_MIN):
            for slot in range(16):
                angle = slot * math.pi / 8
                point = [start[0] + radius * math.cos(angle),
                         start[1] + radius * math.sin(angle), start[2]]
                floor = ns.floor_under(point, 96.0, footprint=(32.0, 32.0))
                if floor is None:
                    continue
                point[2] = floor + CART_RIDE
                if track_segment_ok(start, point, solver):
                    found = [start, point]
                    break
            if found is not None:
                break
        if found is not None:
            tracks.append(found)
            origins.append(start)
    cell_segment_mass = 0
    if not tracks:
        horizontal_radius = max(abs(solver.mins[0]), abs(solver.mins[1]),
                                abs(solver.maxs[0]), abs(solver.maxs[1]))
        for index in cell_order:
            center = ns.standing_point((ns.lo[index] + ns.hi[index]) * 0.5)
            if center is None:
                continue
            for axis in np.argsort(cell_extent[index, :2])[::-1]:
                half = max(0.0, cell_extent[index, axis] / 2.0 - horizontal_radius)
                if 2.0 * half < PATH_MIN:
                    continue
                first = list(center)
                second = list(center)
                first[axis] -= half
                second[axis] += half
                first = solver.settle(first)
                second = solver.settle(second)
                cell_segment_mass += 1
                if (first is not None and second is not None
                        and track_length([first, second]) >= PATH_MIN
                        and track_segment_ok(first, second, solver)):
                    tracks.append([first, second])
                    origins.append(first)
                    break
            if tracks:
                break
    if tracks:
        tracks = swizzle_track_origins([tracks[index % len(tracks)] for index in range(kcarts)], solver)
    print('nav: spawn-origin candidates=%d candidate_segment_mass=%d cell_segment_mass=%d tracks=%d' %
          (len(order), candidate_segment_mass, cell_segment_mass, len(tracks)))
    return tracks

def build_tracks(bsp, mapname, pts, kteams, kcarts, pk3arg='', d=None, ns=None):
    db = d if d is not None else open(bsp, 'rb').read()
    if ns is None:
        import negspace as _NS
        ns = _NS.from_bsp(db, mask=_NS.MASK_PLAYERSOLID)
    text, resolved = load_cache(mapname, bsp, pk3arg)
    if not text:
        print('nav: %s waypoint_mass=0 construction_source=spawn_origin' % mapname)
        return [], None, {'overlap_max': None, 'headon': None, 'flow_alignment': None}
    nodes, adj = parse_cache(text)
    if not nodes or not largest_component(adj):
        print('nav: %s waypoint_mass=%d connected_waypoint_mass=0 construction_source=spawn_origin' %
              (mapname, len(nodes)))
        return [], None, {'overlap_max': None, 'headon': None, 'flow_alignment': None}
    flags = load_flags(mapname, resolved, bsp, pk3arg, nodes)
    gen = {i for i, nd in enumerate(nodes) if nd not in flags} if flags else set()
    tboxes = trigger_boxes(db)
    cart_incompatible = classify_edges(nodes, adj, ns, flags, tboxes)
    ne = sum(len(a) for a in adj) // 2
    rs = {'flag': 0, 'semantic': 0, 'burrow': 0, 'airborne': 0}
    for r in cart_incompatible.values():
        rs[r] = rs.get(r, 0) + 1
    print('nav: %s waypoints=%d saved=%d gen=%d trigboxes=%d links=%d not-cart-segments=%d '
          '(unsaved=%d flagged=%d semantic_jump/teleport=%d burrows_through_solid=%d '
          'floats_over_non-walkable=%d)' %
          (mapname, len(nodes), len(flags), len(gen), len(tboxes), ne,
           len(cart_incompatible),
           len(gen), rs['flag'], rs['semantic'], rs['burrow'], rs['airborne']))
    tracks, paths, st = nav_tracks(nodes, adj, kcarts, ns, cart_incompatible)
    tracks = swizzle_track_origins(tracks, st.get('solver'))
    overlap_max = max((st['ov'][i][j] for i in range(len(tracks))
                       for j in range(i + 1, len(tracks))), default=0.0)
    print('nav: cart-domain=%d/%d branching=%.2f branching_target_delta=%.2f' %
          (st['net'], st['tot'], st['bf'], st['bf'] - 1.5))
    print('nav: selected_cart_path_components=%d candidate_components=%d candidate_nodes=%d' %
          (st['hosts'], st['host_measures']['candidate_component_mass'],
           st['host_measures']['candidate_node_mass']))
    for row in st['path_components']:
        print('nav:   component %d carts %s kept=%d/%d branching=%.2f' %
              (row['component'], row['carts'], row['net'], row['tot'], row['bf']))
    ov = st['ov']
    mx = max((ov[i][j] for i in range(kcarts) for j in range(i + 1, kcarts)), default=0.0)
    for i in range(kcarts):
        for j in range(i + 1, kcarts):
            if ov[i][j] > 0:
                print('nav: overlap %d-%d %.2f' % (i, j, ov[i][j]))
    print('nav: overlap_max=%.2f bound=%.2f bound_minus_overlap=%.2f' %
          (mx, OVERLAP_MAX, OVERLAP_MAX - mx))
    print('nav: flow_alignment chosen=%.3f candidate_worst=%.3f headon=%.3f '
          'push_counterflow=%.3f origin_walking_separation=%.0f' %
          (st['align'], st['worst'], st['headon'], st['push_counterflow'],
           st['origin_walking_separation']))
    origin_points = [track[0] for track in tracks]
    origin_nodes = [min(range(len(nodes)), key=lambda node: math.dist(point, nodes[node]))
                    for point in origin_points]
    origin_attachment = [math.dist(point, nodes[node])
                         for point, node in zip(origin_points, origin_nodes)]
    adj_ok2 = [{v: w for v, w in a.items()
                if (min(u, v), max(u, v)) not in cart_incompatible}
               for u, a in enumerate(adj)]
    dmo = {node: dijkstra(adj_ok2, node)[0] for node in set(origin_nodes)}

    dmn = {node: dijkstra(adj, node)[0] for node in set(origin_nodes)}
    ws, nsd = [], []
    for i in range(len(origin_nodes)):
        for j in range(i + 1, len(origin_nodes)):
            wd = dmo[origin_nodes[i]][origin_nodes[j]]
            nd = dmn[origin_nodes[i]][origin_nodes[j]]
            if wd < math.inf:
                ws.append(origin_attachment[i] + wd + origin_attachment[j])
            if nd < math.inf:
                nsd.append(origin_attachment[i] + nd + origin_attachment[j])
    if nsd:
        print('nav: cart-to-cart navmesh min=%.0f mean=%.0f max=%.0f balance_ratio=%.2f' %
              (min(nsd), sum(nsd) / len(nsd), max(nsd), max(nsd) / min(nsd) if min(nsd) else 0))
    if ws:
        print('nav: same-component walk min=%.0f mean=%.0f max=%.0f' %
              (min(ws), sum(ws) / len(ws), max(ws)))
    print('nav: weights bend=%.1f w_med=%.2f w_floor=%.2f target=%.0f cap=%.0f' %
          (WB, WMED, WFLOOR, CLEAR_TARGET, CLEAR_CAP))
    print('nav: curvature %.0f->%.0f (%.1f%%) botwalk_fallback_carts=%d' %
          (st['e0'], st['e1'], 100 * st['e1'] / st['e0'] if st['e0'] else 0, st.get('bw', 0)))

    NMOD = _navmesh()
    nmv = NMOD.Navmesh(nodes, adj,
                       flags={tuple(round(x, 1) for x in k): v for k, v in (flags or {}).items()},
                       triggerboxes=tboxes)
    origins0 = origin_nodes
    pool = sorted({node for comp in cart_path_components(adj_ok2) for node in comp})
    opt, ost = nmv.equidistant_origins(max(3, kcarts), pool=pool, verbose=True)
    canonical_sites = list(dict.fromkeys(origins0 + opt + nmv.component_representatives()))
    owner = nmv.voronoi(sites=canonical_sites, verbose=True)
    navigation_realization = nmv.realization()
    node_measure = np.asarray(navigation_realization['node_measure'], dtype=np.float64)
    for si, o in enumerate(origins0):
        m = owner == canonical_sites.index(o)
        print('nav: voronoi cell of cart %d (origin wp %d): %d support nodes, '
              '%.4g u of stock navigation measure' %
              (si, o, int(m.sum()), float(node_measure[m].sum())))
    D0 = {node: nmv.walk_dist(node)[0] for node in set(origins0)}
    vals = [origin_attachment[i] + D0[left][right] + origin_attachment[j]
            for i, left in enumerate(origins0)
            for j, right in enumerate(origins0[i + 1:], i + 1)
            if D0[left][right] < math.inf]
    origin_pairs = len(origins0) * (len(origins0) - 1) // 2
    origin_ratio = max(vals) / min(vals) if vals and min(vals) else math.inf
    optimum_ratio = ost.get('ratio')
    approximation_ratio = (origin_ratio / optimum_ratio
                           if origin_ratio is not None and math.isfinite(origin_ratio)
                           and optimum_ratio and math.isfinite(optimum_ratio) else None)
    if vals:
        print('nav: cart origins as PLACED: pairwise navmesh-walking min=%.0f max=%.0f '
              'spread_ratio=%.2f; the k-center optimum over this navmesh is %.2f '
              '(1.00 = exactly equidistant, NAV-SPEC §1)'
              % (min(vals), max(vals), origin_ratio,
                 ost.get('ratio', 0.0)))

    shipped_nodes = sum(len(t) for t in tracks)
    print('nav: cart paths %d over %d shipped nodes from %d solved nodes: free-volume residual=%d '
          'unplaceable=%d airborne=%d max_activation_distance=%.2fu'
          % (len(tracks), shipped_nodes, st.get('raw_nodes', shipped_nodes), st.get('infeasible', 0),
             st.get('unplaceable', 0), st.get('airborne', 0), st.get('maxdev', 0.0)))
    return tracks, (nodes, adj, paths), {
        'overlap_max': overlap_max,
        'headon': st['headon'],
        'push_counterflow': st['push_counterflow'],
        'flow_alignment': st['align'],
        'origin_orientation_walking_separation': st['origin_walking_separation'],
        'cart_origin_attachment_distance': origin_attachment,
        'cart_origin_attachment_distance_max': max(origin_attachment, default=None),
        'voronoi_support_node_mass': len(owner),
        'voronoi_assigned_support_node_mass': int((owner >= 0).sum()),
        'voronoi_unassigned_support_node_mass': int((owner < 0).sum()),
        'voronoi_site_mass': len(canonical_sites),
        'navigation_realization_id': navigation_realization['realization_id'],
        'navigation_realization': navigation_realization,
        'cart_origin_pool_method': 'largest_connected_component_farthest_first_kcenter',
        'cart_origin_pool_count': len(opt),
        'cart_origin_candidate_component_mass': ost.get('component_mass'),
        'cart_origin_metric_pool_count': ost.get('metric_pool_mass'),
        'cart_origin_disconnected_pool_count': ost.get('disconnected_pool_mass'),
        'cart_origin_pool_min': ost.get('min'),
        'cart_origin_pool_max': ost.get('max'),
        'cart_origin_pool_spread_ratio': ost.get('ratio'),
        'cart_origin_selection_method': 'counterflow_constrained_maximum_minimum_navmesh_distance',
        'cart_path_candidate_component_mass': st['host_measures']['candidate_component_mass'],
        'cart_path_candidate_node_mass': st['host_measures']['candidate_node_mass'],
        'cart_path_metric_component_mass': st['host_measures']['metric_component_mass'],
        'cart_path_metric_pool_mass': st['host_measures']['metric_pool_mass'],
        'cart_path_metric_disconnected_candidate_mass': st['host_measures']['metric_disconnected_candidate_mass'],
        'cart_path_selected_component_mass': st['host_measures']['selected_component_mass'],
        'cart_origin_navmesh_pairs': len(vals),
        'cart_origin_navmesh_pairs_expected': origin_pairs,
        'cart_origin_navmesh_min': min(vals) if vals else None,
        'cart_origin_navmesh_max': max(vals) if vals else None,
        'cart_origin_navmesh_spread_ratio': origin_ratio if vals else None,
        'cart_origin_kcenter_approximation_ratio': approximation_ratio,
    }

def entity_value(block, key):
    match = re.search(r'"%s"\s+"([^"]*)"' % re.escape(key), block)
    return match.group(1) if match else ''

def entity_origin(block):
    value = entity_value(block, 'origin')
    return tuple(float(x) for x in value.split()) if value else None

def is_player_spawn(block):
    classname = entity_value(block, 'classname')
    return classname in PLAYER_SPAWN_CLASSES or re.fullmatch(r'info_player_team\d+', classname) is not None

def point_segment_distance(point, a, b):
    delta = tuple(b[i] - a[i] for i in range(3))
    denom = sum(value * value for value in delta)
    t = 0.0 if denom == 0 else max(0.0, min(1.0,
        sum((point[i] - a[i]) * delta[i] for i in range(3)) / denom))
    return math.dist(point, tuple(a[i] + t * delta[i] for i in range(3)))

def cart_clearance(point, tracks):
    return min((point_segment_distance(point, track[i], track[i + 1])
                for track in tracks for i in range(len(track) - 1)), default=math.inf)

def cart_origin_clearance(point, tracks):
    return min((
        math.hypot(point[0] - track[0][0], point[1] - track[0][1])
        if track and abs(point[2] - (track[0][2] + CART_RIDE)) <= PUSH_H else math.inf
        for track in tracks
    ), default=math.inf)

def spawn_occupies_cart_origin(point, track):
    if not track:
        return False
    origin = track[0]
    return (math.hypot(point[0] - origin[0], point[1] - origin[1]) <= PUSH_R
            and abs(point[2] - (origin[2] + CART_RIDE)) <= PUSH_H)

def rider_gap_counts(tracks, ns):
    return [
        sum(bool(ns.support_gaps(
            track[i], track[i + 1], _NS.CART_RIDER_MIN, _NS.CART_RIDER_MAX,
        )) for i in range(len(track) - 1))
        for track in tracks
    ]

def track_construction_measures(tracks, count, ns):
    gaps = rider_gap_counts(tracks, ns)
    lengths = [track_length(track) for track in tracks]
    origins = [track[0] for track in tracks if track]
    collisions = sum(
        math.dist(origins[i], origins[j]) < CART_ORIGIN_SEP
        for i in range(len(origins)) for j in range(i + 1, len(origins))
    )
    missing = max(0, count - len(tracks))
    surplus = max(0, len(tracks) - count)
    short_nodes = sum(len(track) < 2 for track in tracks)
    short_paths = sum(length < PATH_MIN for length in lengths)
    distinct = len(set(tuple(round(float(x), 1) for x in point) for point in origins))
    origin_residual = max(0, count - distinct)
    residual = missing + surplus + short_nodes + short_paths + sum(gaps) + collisions + origin_residual
    separation = min((math.dist(origins[i], origins[j])
                      for i in range(len(origins)) for j in range(i + 1, len(origins))),
                     default=math.inf)
    return {
        'cart_mass': len(tracks),
        'requested_cart_mass': count,
        'missing_cart_mass': missing,
        'surplus_cart_mass': surplus,
        'path_segment_mass': sum(max(0, len(track) - 1) for track in tracks),
        'short_node_path_mass': short_nodes,
        'short_path_mass': short_paths,
        'rider_gap_segment_mass': sum(gaps),
        'distinct_origin_mass': distinct,
        'origin_identity_residual_mass': origin_residual,
        'origin_collision_pair_mass': collisions,
        'origin_separation': separation if len(origins) > 1 else None,
        'construction_residual_mass': residual,
        'rider_gap_by_cart': gaps,
    }

def realize_cart_tracks(tracks, count, points, graph, ns):
    primary = track_construction_measures(tracks, count, ns)
    if not primary['construction_residual_mass']:
        return tracks, ['navigation'] * len(tracks)
    nodes = graph[0] if graph is not None else []
    fallback_points = list(points) + list(spread_points(nodes, max(2, 2 * count)))
    fallback = spawn_tracks(fallback_points, count, ns)
    pool = [(track, 'navigation') for track in tracks]
    pool.extend((track, 'negative_space_spawn_origin') for track in fallback)
    solver = _navmesh().PathSolver(ns)
    pool = [
        (track, source) for track, source in pool
        if len(track) >= 2
        and track_length(track) >= PATH_MIN
        and all(solver.segment_feasible(track[i], track[i + 1]) for i in range(len(track) - 1))
    ]
    pool.sort(key=lambda row: track_length(row[0]), reverse=True)
    realized, sources, origins = [], [], []
    while pool and len(realized) < count:
        track, source = pool.pop(0)
        if min((math.dist(track[0], origin) for origin in origins), default=math.inf) < CART_ORIGIN_SEP:
            continue
        realized.append(track)
        sources.append(source)
        origins.append(track[0])
    swizzled = swizzle_track_origins(realized, solver)
    return swizzled, sources[:len(swizzled)]

def entity_tracks(entities):
    blocks = [block for block in re.findall(r'\{[^{}]*\}', entities)
              if entity_value(block, 'classname') == 'plc_path']
    points = {entity_value(block, 'targetname'): entity_origin(block) for block in blocks}
    targets = {entity_value(block, 'targetname'): entity_value(block, 'target') for block in blocks}
    incoming = {target for target in targets.values() if target in points}
    starts = sorted(set(points) - incoming)
    starts.extend(name for name in sorted(points) if name not in starts)
    tracks, visited = [], set()
    for start in starts:
        if start in visited or points[start] is None:
            continue
        track, name, local = [], start, set()
        while name in points and name not in local and points[name] is not None:
            track.append(points[name])
            local.add(name)
            visited.add(name)
            name = targets.get(name, '')
        if track:
            tracks.append(track)
    return tracks

def spread_points(points, limit):
    points = sorted(set(tuple(float(x) for x in point) for point in points))
    if len(points) <= limit:
        return points
    chosen = [points[0]]
    distance = [math.dist(point, chosen[0]) for point in points]
    while len(chosen) < limit:
        index = max(range(len(points)), key=lambda i: distance[i])
        chosen.append(points[index])
        distance = [min(distance[i], math.dist(points[i], points[index])) for i in range(len(points))]
    return chosen

def localized_points(points):
    remaining = sorted(set(tuple(point) for point in points))
    if not remaining:
        return []
    ordered = [remaining.pop(0)]
    while remaining:
        index = min(range(len(remaining)), key=lambda i: math.dist(ordered[-1], remaining[i]))
        ordered.append(remaining.pop(index))
    return ordered

class SpawnAccessRelation(object):
    def __init__(self, tracks, graph):
        self.origins = [track[0] for track in tracks if track]
        self.nodes, self.adj, _ = graph if graph is not None else ([], [], [])
        self.node_array = np.asarray(self.nodes, dtype=np.float64)
        self.nearest_cache = {}
        self.origin_nodes = [self.nearest(origin)[0] for origin in self.origins]
        self.origin_attachment = np.asarray(
            [self.nearest(origin)[1] for origin in self.origins], dtype=np.float64,
        )
        self.distance = {
            node: np.asarray(dijkstra(self.adj, node)[0], dtype=np.float64)
            for node in set(self.origin_nodes)
        } if self.nodes and self.adj else {}

    def nearest(self, point):
        key = tuple(float(value) for value in point)
        if key not in self.nearest_cache:
            if len(self.node_array):
                delta = self.node_array - np.asarray(key, dtype=np.float64)
                index = int(np.argmin(np.einsum('ij,ij->i', delta, delta)))
                self.nearest_cache[key] = (index, float(np.linalg.norm(delta[index])))
            else:
                self.nearest_cache[key] = (-1, 0.0)
        return self.nearest_cache[key]

    def matrix(self, points):
        points = list(points)
        if not points or not self.origins:
            return np.empty((len(points), len(self.origins)), dtype=np.float64)
        if self.distance:
            attached = [self.nearest(point) for point in points]
            nodes = [row[0] for row in attached]
            offsets = np.asarray([row[1] for row in attached], dtype=np.float64)
            matrix = np.asarray([
                [self.distance[origin][node] for origin in self.origin_nodes]
                for node in nodes
            ], dtype=np.float64)
            return matrix + offsets[:, None] + self.origin_attachment[None, :]
        point_array = np.asarray(points, dtype=np.float64)
        origin_array = np.asarray(self.origins, dtype=np.float64)
        delta = point_array[:, None, :] - origin_array[None, :, :]
        return np.sqrt(np.einsum('ijk,ijk->ij', delta, delta))

def spawn_access_matrix(points, tracks, graph, access=None):
    relation = access if access is not None else SpawnAccessRelation(tracks, graph)
    return relation.matrix(points)

def balanced_spawn_points(points, tracks, graph, limit, access=None):
    matrix = spawn_access_matrix(points, tracks, graph, access)
    finite_mask = np.isfinite(matrix)
    reachable = [i for i in range(len(points)) if finite_mask[i].any()]
    if not reachable or not matrix.shape[1]:
        return []
    size = min(len(reachable), limit)
    finite = [i for i in reachable if finite_mask[i].all()]
    if not finite:
        group = []
        uncovered = set(range(matrix.shape[1]))
        available = set(reachable)
        while available and uncovered and len(group) < size:
            index = min(available, key=lambda i: (
                -sum(finite_mask[i, column] for column in uncovered),
                sum(float(matrix[i, column]) for column in uncovered if finite_mask[i, column]),
                i,
            ))
            group.append(index)
            available.remove(index)
            uncovered -= {column for column in uncovered if finite_mask[index, column]}
        while available and len(group) < size:
            index = max(available, key=lambda i: min(
                (math.dist(points[i], points[chosen]) for chosen in group), default=math.inf,
            ))
            group.append(index)
            available.remove(index)
        return localized_points([points[i] for i in group])
    finite_rows = matrix[finite]
    row_ratio = np.max(finite_rows, axis=1) / np.maximum(np.min(finite_rows, axis=1), 1.0)
    row_sum = np.sum(finite_rows, axis=1)
    coordinates = np.asarray([points[i] for i in finite], dtype=np.float64)
    centroid = np.median(coordinates, axis=0)
    center_distance = np.sqrt(np.sum((coordinates - centroid) ** 2, axis=1))
    order = np.lexsort((np.asarray(finite), center_distance, row_sum, row_ratio))
    chosen = [finite[int(index)] for index in order[:size]]
    return localized_points([points[i] for i in chosen])

def spawn_access_metrics(points, tracks, graph, access=None):
    matrix = spawn_access_matrix(points, tracks, graph, access)
    nodes, adj, _ = graph if graph is not None else ([], [], [])
    relation = 'stock_playerbot_navigation' if nodes and adj else 'straight_line_geometric'
    if not len(points) or not matrix.shape[1]:
        return {
            'relation': relation,
            'nonfinite_count': int(matrix.size),
            'cart_median_ratio': None,
            'cart_first_claim_ratio': None,
            'per_spawn_ratio_p90': None,
            'per_spawn_ratio_max': None,
            'cart_distance': [],
        }
    safe = np.where(np.isfinite(matrix), matrix, np.nan)
    cart_distance = []
    for column in range(matrix.shape[1]):
        values = matrix[:, column][np.isfinite(matrix[:, column])]
        cart_distance.append({
            'finite_spawns': len(values),
            'spawns': len(points),
            'minimum': round(float(values.min()), 3) if len(values) else None,
            'median': round(float(np.median(values)), 3) if len(values) else None,
            'p90': round(float(np.percentile(values, 90)), 3) if len(values) else None,
            'maximum': round(float(values.max()), 3) if len(values) else None,
        })
    medians = [row['median'] for row in cart_distance if row['median'] is not None]
    minimums = [row['minimum'] for row in cart_distance if row['minimum'] is not None]
    row_min = np.nanmin(safe, axis=1)
    row_max = np.nanmax(safe, axis=1)
    return {
        'relation': relation,
        'nonfinite_count': int(matrix.size - np.isfinite(matrix).sum()),
        'cart_median_ratio': round(max(medians) / max(min(medians), 1.0), 6) if medians else None,
        'cart_first_claim_ratio': round(max(minimums) / max(min(minimums), 1.0), 6) if minimums else None,
        'per_spawn_ratio_p90': round(float(np.nanpercentile(row_max / np.maximum(row_min, 1.0), 90)), 6),
        'per_spawn_ratio_max': round(float(np.nanmax(row_max / np.maximum(row_min, 1.0))), 6),
        'cart_distance': cart_distance,
    }

def spawn_overlay(entities, tracks, graph, limit=None, clearance=None, ns=None, access=None):
    clearance = PUSH_R + EPS if clearance is None else float(clearance)
    blocks = re.findall(r'\{[^{}]*\}', entities)
    originals = [entity_origin(block) for block in blocks
                 if is_player_spawn(block)]
    original_spawn_classes = [entity_value(block, 'classname') for block in blocks
                              if is_player_spawn(block)]
    nodes, adj, _ = graph if graph is not None else ([], [], [])
    candidates = [point for point in originals if point]
    candidates.extend(point for index, point in enumerate(nodes) if adj[index])
    if ns is not None:
        body_width = _NS.PL_MAX[0] - _NS.PL_MIN[0]
        spawn_mins = (_NS.PL_MIN[0] - body_width, _NS.PL_MIN[1] - body_width,
                      _NS.PL_MIN[2])
        spawn_maxs = (_NS.PL_MAX[0] + body_width, _NS.PL_MAX[1] + body_width,
                      _NS.PL_MAX[2])
        spawn_lift = (abs(_NS.PL_MIN[2]),)
        standing, present, standing_measures = ns.standing_points(
            candidates, mins=spawn_mins, maxs=spawn_maxs, lift=spawn_lift,
        )
        candidates = standing[present].tolist()
        candidates = spread_points(candidates, len(candidates))
    else:
        standing_measures = None
    clear = [point for point in candidates if cart_origin_clearance(point, tracks) >= clearance]
    covered = np.isfinite(spawn_access_matrix(clear, tracks, graph, access)).any(axis=0) if clear else np.zeros(len(tracks), dtype=bool)
    all_access = spawn_access_matrix(candidates, tracks, graph, access)
    selected = list(dict.fromkeys(tuple(point) for point in clear))
    selected_set = set(selected)
    for cart in np.flatnonzero(~covered):
        reachable = [
            index for index in range(len(candidates))
            if all_access.shape[1] > cart and np.isfinite(all_access[index, cart])
            and not any(spawn_occupies_cart_origin(candidates[index], track) for track in tracks)
        ]
        if reachable:
            index = max(reachable, key=lambda value: cart_origin_clearance(candidates[value], tracks))
            point = tuple(candidates[index])
            if point not in selected_set:
                selected.append(point)
                selected_set.add(point)
    clear = [list(point) for point in selected]
    limit = max(1, len(originals), len(tracks)) if limit is None else max(1, int(limit))
    chosen = balanced_spawn_points(clear, tracks, graph, limit, access) if tracks else localized_points(
        spread_points(clear, min(len(clear), limit))
    )
    recovery_mass = 0
    if not chosen:
        recovery = list(candidates) or [point for point in originals if point]
        if not recovery and ns is not None:
            for index in np.argsort(np.prod(np.maximum(ns.hi - ns.lo, 0.0), axis=1))[::-1]:
                point = ns.standing_point((ns.lo[index] + ns.hi[index]) * 0.5)
                if point is not None:
                    recovery.append(list(point))
                    break
        chosen = localized_points(spread_points(recovery, min(len(recovery), limit)))
        recovery_mass = len(chosen)

    def keep(match):
        block = match.group(0)
        return '' if is_player_spawn(block) else block

    stripped = re.sub(r'\{[^{}]*\}', keep, entities).rstrip()
    spawned = ['\n'.join(('{', '"classname" "info_player_deathmatch"',
                            '"origin" "%.9g %.9g %.9g"' % point, '}'))
               for point in chosen]
    minimum = min((cart_origin_clearance(point, tracks) for point in chosen), default=None)
    measures = spawn_access_metrics(chosen, tracks, graph, access)
    measures['standing_point_measures'] = standing_measures
    return (stripped + '\n' + '\n'.join(spawned) + '\n', chosen, minimum, len(clear),
            measures, original_spawn_classes, recovery_mass, clearance)

def emit(bsp, out, kteams, kcarts, pk3arg='', ns=None):
    mapname = os.path.splitext(os.path.basename(bsp))[0]
    meter = WorkloadMeter('xonotic-map-entity-builder', {
        'map': mapname, 'environment': 'map-entity-builder',
    })
    d = open(bsp, 'rb').read()
    off, ln = struct.unpack_from('<ii', d, 8)
    ents = d[off:off + ln].split(b'\0')[0].decode('latin-1')
    blocks = re.findall(r'\{[^{}]*\}', ents)
    spawns = [b for b in blocks if is_player_spawn(b)]

    def origin(b):
        m = re.search(r'"origin"\s+"([-\d. ]+)"', b)
        return [float(x) for x in m.group(1).split()] if m else None

    pts = [origin(b) for b in spawns if origin(b)]
    print('team spawns:', len(pts))

    if ns is None:
        cache = bsp + '.negspace.npz'
        cache_id = cache + '.bsp.json'
        identity = {'schema': _NS.NEGSPACE_SCHEMA, 'bytes': len(d), 'crc32': zlib.crc32(d)}
        try:
            if os.path.exists(cache) and json.load(open(cache_id)) == identity:
                ns = _NS.load_saved(cache)
                print('nav: loaded negative-space cache', cache)
        except Exception as exc:
            print('nav: negative-space cache unavailable:', repr(exc))
        if ns is not None and not hasattr(ns, 'blk_H'):
            print('nav: upgrading negative-space cache with swept-hull brush index')
            ns = None
        if ns is None:
            ns = _NS.from_bsp(d, mask=_NS.MASK_PLAYERSOLID)
            _NS.save(ns, cache)
            with open(cache_id, 'w') as handle:
                json.dump(identity, handle, sort_keys=True)
                handle.write('\n')
            print('nav: wrote negative-space cache', cache)

    with meter.span('cart-construction', rows=kcarts, operations={
        'teams': kteams, 'requested_carts': kcarts,
    }):
        tracks, graph, track_stats = build_tracks(
            bsp, mapname, pts, kteams, kcarts, pk3arg, d, ns=ns,
        )
        tracks, track_sources = realize_cart_tracks(tracks, kcarts, pts, graph, ns)
        construction = track_construction_measures(tracks, kcarts, ns)
    construction_source = '+'.join(sorted(set(track_sources))) if track_sources else 'none'
    construction_source_mass = {
        source: track_sources.count(source) for source in sorted(set(track_sources))
    }

    placed_tracks = [[[p[0], p[1], p[2] - CART_RIDE] for p in track] for track in tracks]
    spawn_access_relation = SpawnAccessRelation(placed_tracks, graph)
    (entities, spawn_points, spawn_minimum, spawn_candidates, spawn_access,
     original_spawn_classes, spawn_recovery_mass, spawn_origin_clearance) = spawn_overlay(
        ents.rstrip('\0'), placed_tracks, graph, ns=ns, access=spawn_access_relation,
    )
    approach_distances = [row['median'] for row in spawn_access['cart_distance']]
    lengths, speed_scales, approach_times, traversal_times, end_to_end_times, end_to_end_ratio = cart_speed_scales(placed_tracks, approach_distances)
    extra = []
    named = []
    for c, track in enumerate(placed_tracks):
        NC = len(track)
        names = ['plc%dn%d' % (c, i) for i in range(NC)]
        named.append((names, track))
        for i, (name, p) in enumerate(zip(names, track)):
            e = ['{', '"classname" "plc_path"', '"targetname" "%s"' % name,
                 '"origin" "%.0f %.0f %.0f"' % tuple(p)]
            if i + 1 < NC:
                e.append('"target" "%s"' % names[i + 1])
            if 0 < i < NC - 1 and i % max(1, NC // 4) == 0:
                e.append('"spawnflags" "1"')
            e.append('}')
            extra.append('\n'.join(e))
        extra.append('\n'.join(['{', '"classname" "func_plc_cart"',
                                '"target" "%s"' % names[0],
                                '"plc_speed_scale" "%.9g"' % speed_scales[c], '}']))

    if named:
        for t in range(kteams):
            names, track = named[t % len(named)]
            extra.append('\n'.join(['{', '"classname" "plc_goal"',
                                    '"cnt" "%d"' % t, '"target" "%s"' % names[-1],
                                    '"radius" "64"',
                                    '"origin" "%.0f %.0f %.0f"' % tuple(track[-1]),
                                    '}']))

    for c, (names, track) in enumerate(named):
        L = sum(math.dist(track[i], track[i + 1]) for i in range(len(track) - 1))
        print('cart %d: %s -> %s length %.0f nodes %d' % (c, track[0], track[-1], L, len(track)))
    sep = min(math.dist(pa, pb) for _, ta in named[:1] for pa in ta
              for _, tb in named[1:] for pb in tb) if len(named) > 1 else None
    print('teams', kteams, 'requested_carts', kcarts, 'realized_carts', len(named),
          'min_inter_track_node_distance', sep)

    rider_gap_by_cart = construction['rider_gap_by_cart']
    rider_gap_segments = sum(rider_gap_by_cart)
    open(out, 'w').write(entities + '\n'.join(extra) + '\n')
    length_ratio = max(lengths) / max(min(lengths), 1.0) if lengths else None
    traversal_time_ratio = max(traversal_times) / max(min(traversal_times), EPS) if traversal_times else None
    origin_separation = min((math.dist(placed_tracks[i][0], placed_tracks[j][0])
                             for i in range(len(placed_tracks))
                             for j in range(i + 1, len(placed_tracks))), default=None)
    spawn_cart_matrix = spawn_access_matrix(
        spawn_points, placed_tracks, graph, spawn_access_relation,
    )
    cart_path_measures = []
    for c, track in enumerate(placed_tracks):
        finite = np.isfinite(spawn_cart_matrix[:, c]) if spawn_cart_matrix.shape[1] > c else np.zeros(len(spawn_points), dtype=bool)
        distances = spawn_cart_matrix[:, c][finite] if spawn_cart_matrix.shape[1] > c else np.empty(0)
        origin_distances = [math.dist(point, track[0]) for point in spawn_points]
        path_non_degenerate = len(track) > 1 and lengths[c] > 0
        rider_continuous = rider_gap_by_cart[c] == 0
        spawn_reachable = bool(finite.sum()) and spawn_access['relation'] == 'stock_playerbot_navigation'
        row = {
            'cart': c,
            'path_nodes': len(track),
            'path_segments': max(0, len(track) - 1),
            'path_length': lengths[c],
            'rider_gap_segments': rider_gap_by_cart[c],
            'finite_spawn_routes': int(finite.sum()),
            'spawn_routes': len(spawn_points),
            'path_non_degenerate_mass': int(path_non_degenerate),
            'rider_continuous_mass': int(rider_continuous),
            'spawn_reachable_mass': int(spawn_reachable),
            'advanceable_mass': int(path_non_degenerate and rider_continuous and spawn_reachable),
            'spawn_route_min': float(distances.min()) if len(distances) else None,
            'spawn_route_median': float(np.median(distances)) if len(distances) else None,
            'spawn_route_max': float(distances.max()) if len(distances) else None,
            'origin_spawn_distance_min': min(origin_distances) if origin_distances else None,
            'origin_spawn_distance_median': float(np.median(origin_distances)) if origin_distances else None,
            'origin_spawn_distance_max': max(origin_distances) if origin_distances else None,
        }
        cart_path_measures.append(row)
    team_objective_measures = [
        {
            'team': team + 1,
            'capture_cart_ids': list(range(len(placed_tracks))),
            'capture_cart_count': len(placed_tracks),
            'advanceable_cart_ids': [
                row['cart'] for row in cart_path_measures if row['advanceable_mass']
            ],
            'advanceable_cart_count': sum(
                row['advanceable_mass'] for row in cart_path_measures
            ),
            'spawn_reachable_cart_ids': [
                row['cart'] for row in cart_path_measures if row['spawn_reachable_mass']
            ],
            'rider_continuous_cart_ids': [
                row['cart'] for row in cart_path_measures if row['rider_continuous_mass']
            ],
            'nominal_lane_cart': team % len(placed_tracks) if placed_tracks else None,
            'nominal_lane_endpoint': list(placed_tracks[team % len(placed_tracks)][-1]) if placed_tracks else None,
        }
        for team in range(kteams)
    ]
    shared_spawn_team_pairs = len(spawn_points) * kteams
    cart_route_mass = sum(row['spawn_reachable_mass'] for row in cart_path_measures)
    cart_advanceable_mass = sum(row['advanceable_mass'] for row in cart_path_measures)
    cart_rider_continuous_mass = sum(row['rider_continuous_mass'] for row in cart_path_measures)
    cart_path_non_degenerate_mass = sum(row['path_non_degenerate_mass'] for row in cart_path_measures)
    team_cart_pair_mass = kteams * len(placed_tracks)
    team_cart_advanceable_pair_mass = kteams * cart_advanceable_mass
    spawn_cart_origin_occupancy_pair_mass = sum(
        spawn_occupies_cart_origin(point, track)
        for point in spawn_points for track in placed_tracks
    )
    measurements = {
        'schema': MAP_MEASUREMENT_SCHEMA,
        'map': mapname,
        'negative_space_schema': _NS.NEGSPACE_SCHEMA,
        'compiled_collision_brush_mass': int(getattr(ns, 'compiled_brush_mass', 0)),
        'compiled_collision_patch_triangle_mass': int(
            getattr(ns, 'patch_triangle_mass', 0)
        ),
        'teams': kteams,
        'carts': kcarts,
        'realized_carts': len(placed_tracks),
        'push_configuration_error_mass': int(PUSH_CONFIGURATION_ERROR is not None),
        'push_configuration_error': PUSH_CONFIGURATION_ERROR,
        'goals': kteams if placed_tracks else 0,
        'generic_spawns': len(spawn_points),
        'original_player_spawn_mass': len(original_spawn_classes),
        'original_team_labeled_spawn_mass': sum(
            name not in NEUTRAL_SPAWN_CLASSES for name in original_spawn_classes
        ),
        'residual_team_labeled_spawn_mass': 0,
        'spawn_candidates_clear': spawn_candidates,
        'spawn_hull_clear': len(spawn_points),
        'spawn_cart_origin_clearance': spawn_origin_clearance,
        'spawn_cart_origin_clearance_mass': sum(
            cart_origin_clearance(point, placed_tracks) >= spawn_origin_clearance for point in spawn_points
        ),
        'spawn_cart_origin_clearance_residual_mass': sum(
            cart_origin_clearance(point, placed_tracks) < spawn_origin_clearance for point in spawn_points
        ),
        'spawn_recovery_mass': spawn_recovery_mass,
        'spawn_cart_origin_clearance_min': spawn_minimum,
        'spawn_path_clearance_min': min(
            (cart_clearance(point, placed_tracks) for point in spawn_points), default=None,
        ),
        'shared_spawn_team_pairs': shared_spawn_team_pairs,
        'spawn_team_pairs': shared_spawn_team_pairs,
        'spawn_team_access_ratio': shared_spawn_team_pairs / max(1, len(spawn_points) * kteams),
        'spawn_cart_origin_occupancy_pair_mass': spawn_cart_origin_occupancy_pair_mass,
        'spawn_access_relation': spawn_access['relation'],
        'spawn_standing_point_measures': spawn_access['standing_point_measures'],
        'spawn_access_origin_node_mass': len(set(spawn_access_relation.origin_nodes)),
        'spawn_access_distance_row_mass': len(spawn_access_relation.distance),
        'spawn_access_nearest_cache_mass': len(spawn_access_relation.nearest_cache),
        'spawn_access_origin_attachment_distance': spawn_access_relation.origin_attachment.tolist(),
        'spawn_cart_nonfinite_distances': spawn_access['nonfinite_count'],
        'spawn_cart_median_ratio': spawn_access['cart_median_ratio'],
        'spawn_cart_first_claim_ratio': spawn_access['cart_first_claim_ratio'],
        'spawn_cart_per_spawn_ratio_p90': spawn_access['per_spawn_ratio_p90'],
        'spawn_cart_per_spawn_ratio_max': spawn_access['per_spawn_ratio_max'],
        'spawn_cart_distance': spawn_access['cart_distance'],
        'path_lengths': lengths,
        'path_length_ratio': length_ratio,
        'cart_speed_scales': speed_scales,
        'spawn_approach_speed': SPAWN_APPROACH_SPEED,
        'spawn_cart_approach_times': approach_times,
        'spawn_cart_approach_missing_mass': sum(value is None for value in approach_times),
        'nominal_traversal_times': traversal_times,
        'nominal_traversal_time_ratio': traversal_time_ratio,
        'nominal_end_to_end_times': end_to_end_times,
        'nominal_end_to_end_time_ratio': end_to_end_ratio,
        'path_overlap_max': track_stats['overlap_max'],
        'headon_flow': track_stats['headon'],
        'push_zone_counterflow': track_stats.get('push_counterflow'),
        'flow_alignment': track_stats['flow_alignment'],
        'origin_orientation_walking_separation': track_stats.get('origin_orientation_walking_separation'),
        'cart_origin_attachment_distance': track_stats.get('cart_origin_attachment_distance'),
        'cart_origin_attachment_distance_max': track_stats.get('cart_origin_attachment_distance_max'),
        'voronoi_support_node_mass': track_stats.get('voronoi_support_node_mass'),
        'voronoi_assigned_support_node_mass': track_stats.get('voronoi_assigned_support_node_mass'),
        'voronoi_unassigned_support_node_mass': track_stats.get('voronoi_unassigned_support_node_mass'),
        'voronoi_site_mass': track_stats.get('voronoi_site_mass'),
        'navigation_realization': track_stats.get('navigation_realization'),
        'cart_origin_pool_method': track_stats.get('cart_origin_pool_method'),
        'cart_origin_pool_count': track_stats.get('cart_origin_pool_count'),
        'cart_origin_candidate_component_mass': track_stats.get('cart_origin_candidate_component_mass'),
        'cart_origin_metric_pool_count': track_stats.get('cart_origin_metric_pool_count'),
        'cart_origin_disconnected_pool_count': track_stats.get('cart_origin_disconnected_pool_count'),
        'cart_origin_pool_min': track_stats.get('cart_origin_pool_min'),
        'cart_origin_pool_max': track_stats.get('cart_origin_pool_max'),
        'cart_origin_pool_spread_ratio': track_stats.get('cart_origin_pool_spread_ratio'),
        'cart_origin_selection_method': track_stats.get('cart_origin_selection_method'),
        'cart_path_candidate_component_mass': track_stats.get('cart_path_candidate_component_mass'),
        'cart_path_candidate_node_mass': track_stats.get('cart_path_candidate_node_mass'),
        'cart_path_metric_component_mass': track_stats.get('cart_path_metric_component_mass'),
        'cart_path_metric_pool_mass': track_stats.get('cart_path_metric_pool_mass'),
        'cart_path_metric_disconnected_candidate_mass': track_stats.get('cart_path_metric_disconnected_candidate_mass'),
        'cart_path_selected_component_mass': track_stats.get('cart_path_selected_component_mass'),
        'cart_origin_navmesh_pairs': track_stats.get('cart_origin_navmesh_pairs'),
        'cart_origin_navmesh_pairs_expected': track_stats.get('cart_origin_navmesh_pairs_expected'),
        'cart_origin_navmesh_min': track_stats.get('cart_origin_navmesh_min'),
        'cart_origin_navmesh_max': track_stats.get('cart_origin_navmesh_max'),
        'cart_origin_navmesh_spread_ratio': track_stats.get('cart_origin_navmesh_spread_ratio'),
        'cart_origin_kcenter_approximation_ratio': track_stats.get('cart_origin_kcenter_approximation_ratio'),
        'distinct_origins': len(set(tuple(round(float(x), 1) for x in track[0]) for track in placed_tracks)),
        'origin_separation': origin_separation,
        'rider_gap_segments': rider_gap_segments,
        'rider_continuous_cart_mass': sum(value == 0 for value in rider_gap_by_cart),
        'cart_construction_source': construction_source,
        'cart_construction_source_mass': construction_source_mass,
        'cart_construction_measures': construction,
        'team_cart_capture_pair_mass': team_cart_pair_mass,
        'team_cart_advanceable_pair_mass': team_cart_advanceable_pair_mass,
        'team_cart_nonadvanceable_pair_mass': team_cart_pair_mass - team_cart_advanceable_pair_mass,
        'team_cart_spawn_unreachable_pair_mass': kteams * (len(placed_tracks) - cart_route_mass),
        'team_cart_rider_discontinuous_pair_mass': kteams * (len(placed_tracks) - cart_rider_continuous_mass),
        'team_cart_path_degenerate_pair_mass': kteams * (len(placed_tracks) - cart_path_non_degenerate_mass),
        'cart_path_measures': cart_path_measures,
        'team_objective_measures': team_objective_measures,
    }
    with open(out + '.measurements.json', 'w') as handle:
        json.dump(measurements, handle, indent=2, sort_keys=True)
        handle.write('\n')
    meter.close({
        'teams': kteams,
        'carts': kcarts,
        'realized_carts': len(placed_tracks),
        'cart_construction_measures': construction,
        'rider_gap_segments': rider_gap_segments,
        'spawn_team_pairs': shared_spawn_team_pairs,
        'team_cart_capture_pair_mass': team_cart_pair_mass,
        'team_cart_advanceable_pair_mass': team_cart_advanceable_pair_mass,
    })
    print('wrote', out)

if __name__ == '__main__':
    bsp, out = sys.argv[1], sys.argv[2]

    kteams = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    kcarts = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    pk3arg = sys.argv[5] if len(sys.argv) > 5 else ''
    emit(bsp, out, kteams, kcarts, pk3arg)
