import glob
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np

import negspace as NS
from strategy_io_schema import navigation_targets

CART_MIN = NS.CART_MIN
CART_MAX = NS.CART_MAX
GROUND_DROP = 96.0
OVERLAP_MAX = 0.3
PEN0 = 8.0
WPF_JUMP = 1 << 14
WPF_TELEPORT = 1 << 21
WPF_CART_TRANSITION = WPF_JUMP | WPF_TELEPORT | (1 << 15) | (1 << 13)

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
        return Path(loose).read_text(encoding='latin-1')
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
        values = tuple(float(x) for x in v.split())
        if len(values) != 3 or not all(math.isfinite(x) for x in values):
            raise ValueError('waypoint coordinate is not a finite 3-vector')
        k = tuple(round(x, 1) for x in values)
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
        if a != b:
            w = math.dist(nodes[a], nodes[b])
            adj[a][b] = min(adj[a].get(b, math.inf), w)
            adj[b][a] = min(adj[b].get(a, math.inf), w)
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

def kcenter(adj, k, comp=None):
    comp = list(comp) if comp is not None else max(components(adj), key=len, default=[])
    picks, distances = [], {}
    if comp:
        distance, _ = dijkstra(adj, comp[0])
        first = max(comp, key=lambda node: distance[node])
        picks.append(first)
        distances[first] = dijkstra(adj, first)[0]
    while len(picks) < min(k, len(comp)):
        node = max((node for node in comp if node not in distances),
                   key=lambda node: min(distances[pick][node] for pick in picks))
        picks.append(node)
        distances[node] = dijkstra(adj, node)[0]
    return picks, distances

def prune_component(adj, comp, dangle_min):
    keep = set(comp)
    changed = True
    while changed:
        changed = False
        for start in sorted(keep):
            neighbors = [node for node in adj[start] if node in keep]
            if start not in keep or len(neighbors) != 1:
                continue
            chain, length, previous, current = [start], 0.0, -1, start
            while True:
                following = [node for node in adj[current] if node in keep and node != previous]
                if not following:
                    break
                neighbor = following[0]
                length += adj[current][neighbor]
                previous, current = current, neighbor
                if sum(node in keep for node in adj[current]) != 2:
                    break
                chain.append(current)
            if length < dangle_min and sum(node in keep for node in adj[current]) > 2:
                keep.difference_update(chain)
                changed = True
    return keep

def overlap_matrix(paths):
    members = [set(path) for path in paths]
    return [[len(a & b) / max(1, min(len(a), len(b))) if i != j else 0.0
             for j, b in enumerate(members)] for i, a in enumerate(members)]

def subgraph(adj_ok, keep):
    return [{v: w for v, w in a.items() if u in keep and v in keep} for u, a in enumerate(adj_ok)]

def extract_path(prev, src, goal):
    p, u = [], goal
    while u != -1:
        p.append(u)
        u = prev[u]
    return p[::-1] if len(p) > 1 and p[-1] != p[0] else [src]

def plan_spans(nodes, adj_ok, kcarts, keep):
    adjn = subgraph(adj_ok, keep)
    comp = sorted(keep)
    ncand = min(max(3, 2 * kcarts), len(comp))
    if ncand < 2:
        only = comp[0] if comp else 0
        return [[only]] * kcarts, [[0.0] * kcarts for _ in range(kcarts)], {'endpoint_nodes': comp, 'candidate_pair_mass': 0}
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
    p0 = greedy(order)
    if len(p0) >= 2:
        pairings.append([(p0[i][0], p0[(i + 1) % len(p0)][1]) for i in range(len(p0))])
    seen = set()
    pairings = [pr for pr in pairings if not (tuple(pr) in seen or seen.add(tuple(pr)))]
    best = None
    for pairs in pairings:
        P = PEN0
        for _ in range(3):
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
    return paths, ov, {'endpoint_nodes': cands, 'candidate_pair_mass': len(order)}

def flow_assign(nodes, paths, adj_ok, push_radius, metric_adj=None):
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
    rad = 2 * push_radius
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
    candidates = range(1 << k) if k <= 12 else candidates
    scored = sorted((round(score(f), 3), -spread(f), f) for f in candidates)
    chosen = scored[0]
    worst = scored[-1][0]
    flips = chosen[2]
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
        'flow_alignment': chosen[0],
        'candidate_worst_alignment': worst,
        'origin_walking_separation': -chosen[1],
        'headon': exact_headon,
        'push_zone_counterflow': headon(flips),
        'direction_search': 'exhaustive' if k <= 12 else 'coordinate_descent',
        'direction_candidate_mass': len(scored),
    }

class Navmesh(object):

    def __init__(self, nodes, adj, flags=None, jumplinks=(), triggerboxes=()):
        self.nodes = [list(n) for n in nodes]
        self.adj = [dict(a) for a in adj]
        self.flags = flags or {}
        self.jumplinks = set(jumplinks)
        self.triggerboxes = list(triggerboxes)
        self.cart_incompatible = {}

    @classmethod
    def load(cls, bsp, mapname, pk3arg='', data=None):
        text, resolved = load_cache(mapname, bsp, pk3arg)
        nodes, adj = parse_cache(text)
        flags = load_flags(mapname, resolved, bsp, pk3arg, nodes)
        return cls(nodes, adj, flags=flags,
                   triggerboxes=trigger_boxes(data if data is not None else Path(bsp).read_bytes()))

    def classify_edges(self, ns, verbose=False):
        solver = PathSolver(ns, triggerboxes=self.triggerboxes)
        self.cart_nodes, present, _ = solver.settle_many(self.nodes)
        edges = [(u, v) for u, row in enumerate(self.adj) for v in row if u < v]
        starts = np.asarray([self.cart_nodes[u] for u, v in edges]).reshape((-1, 3))
        ends = np.asarray([self.cart_nodes[v] for u, v in edges]).reshape((-1, 3))
        free, grounded, _ = solver.segment_relations(starts, ends)
        self.cart_incompatible = {}
        self.cart_adj = [{} for _ in self.nodes]
        for index, (u, v) in enumerate(edges):
            reason = ('semantic' if self._cart_incompatible_semantics(u, v)
                      else 'burrow' if not free[index] or not (present[u] and present[v])
                      else 'airborne' if not grounded[index] else '')
            if reason:
                self.cart_incompatible[u, v] = reason
            else:
                self.cart_adj[u][v] = self.cart_adj[v][u] = math.dist(self.cart_nodes[u], self.cart_nodes[v])
        if verbose:
            counts = {reason: list(self.cart_incompatible.values()).count(reason)
                      for reason in ('semantic', 'burrow', 'airborne')}
            print('nav: stock_nodes=%d stock_edges=%d cart_edges=%d excluded=%s' %
                  (len(self.nodes), len(edges), len(edges) - len(self.cart_incompatible), counts))
        return self.cart_incompatible

    def _cart_incompatible_semantics(self, u, v):
        fu = self.flags.get(tuple(round(x, 1) for x in self.nodes[u]), 0)
        fv = self.flags.get(tuple(round(x, 1) for x in self.nodes[v]), 0)
        return (bool((fu | fv) & WPF_CART_TRANSITION)
                or (u, v) in self.jumplinks or (v, u) in self.jumplinks
                or any(seg_hits_box(self.nodes[u], self.nodes[v], lo, hi)
                       for lo, hi in self.triggerboxes))

    def voronoi(self, sites=None, verbose=False):
        import heapq
        if sites is None:
            sites = list(range(len(self.nodes)))
        owner = np.full(len(self.nodes), -1, dtype=np.int64)
        distance = np.full(len(self.nodes), np.inf, dtype=np.float64)
        queue = []
        for si, i in enumerate(sites):
            if 0 <= i < len(self.nodes) and (0.0 < distance[i] or si < owner[i]):
                owner[i] = si
                distance[i] = 0.0
                heapq.heappush(queue, (0.0, si, int(i)))
        while queue:
            travelled, source, node = heapq.heappop(queue)
            if travelled > distance[node] or source != owner[node]:
                continue
            for neighbor, length in self.adj[node].items():
                candidate = travelled + float(length)
                if (candidate < distance[neighbor] - 1e-9
                        or abs(candidate - distance[neighbor]) <= 1e-9
                        and source < owner[neighbor]):
                    distance[neighbor] = candidate
                    owner[neighbor] = source
                    heapq.heappush(queue, (candidate, source, int(neighbor)))
        self.vor_owner = owner
        self.vor_sites = [int(site) for site in sites]
        assigned = int((owner >= 0).sum())
        if verbose:
            import collections
            sz = collections.Counter(int(o) for o in owner if o >= 0)
            print('navmesh: metric-graph Voronoi over %d sites -> %d/%d nodes assigned '
                  '(%d unreachable from every site); nodes-per-site '
                  'median=%d max=%d' % (len(sites), assigned, len(self.nodes),
                                        len(self.nodes) - assigned,
                                        int(np.median(list(sz.values()))) if sz else 0,
                                        max(sz.values()) if sz else 0))
        return owner

    def realization(self):
        node_measure = np.zeros(len(self.nodes), dtype=np.float64)
        edges = []
        for left, neighbors in enumerate(self.adj):
            for right, length in neighbors.items():
                if left < right:
                    edges.append([left, int(right), float(length)])
                    node_measure[left] += 0.5 * float(length)
                    node_measure[right] += 0.5 * float(length)
        record = {
            'schema': 3,
            'relation': 'stock_playerbot_navigation_metric_graph_voronoi',
            'nodes': [[float(value) for value in node] for node in self.nodes],
            'edges': edges,
            'node_measure': node_measure.tolist(),
            'total_measure': float(node_measure.sum()),
            'measure_dimension': 1,
        }
        owner = getattr(self, 'vor_owner', None)
        sites = getattr(self, 'vor_sites', None)
        if owner is not None and sites is not None:
            cell_measure = np.zeros(len(sites), dtype=np.float64)
            assigned = owner >= 0
            np.add.at(cell_measure, owner[assigned], node_measure[assigned])
            record['voronoi'] = {
                'site_nodes': sites,
                'owner': owner.tolist(),
                'cell_measure': cell_measure.tolist(),
                'assigned_node_mass': int(assigned.sum()),
                'unassigned_node_mass': int((~assigned).sum()),
            }
        record['targets'] = navigation_targets(record)
        canonical = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
        record['realization_id'] = hashlib.sha256(canonical).hexdigest()
        return record

def resample(poly, spacing):
    if len(poly) < 2:
        return [list(p) for p in poly]
    out = [list(poly[0])]
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        L = math.dist(a, b)
        if L < 1e-6:
            continue
        t = spacing
        while t < L:
            f = t / L
            out.append([a[j] + f * (b[j] - a[j]) for j in range(3)])
            t += spacing
        out.append([float(x) for x in b])
    return out

def tangent_energy(points):
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    second = points[2:] - 2.0 * points[1:-1] + points[:-2]
    return float(np.sum(second * second))

class PathSolver:
    def __init__(self, ns, mins=CART_MIN, maxs=CART_MAX, spacing=64.0, ride=24.0, triggerboxes=()):
        self.ns, self.mins, self.maxs = ns, mins, maxs
        self.spacing, self.ride = spacing, ride
        self.triggerboxes = triggerboxes

    def segment_relations(self, starts, ends):
        starts = np.asarray(starts, dtype=np.float64).reshape((-1, 3))
        ends = np.asarray(ends, dtype=np.float64).reshape((-1, 3))
        free, _, measures = self.ns.segment_relations(starts, ends, self.mins, self.maxs)
        foot = (0.0, 0.0, self.mins[2])
        _, grounded, _ = self.ns.segment_relations(starts, ends, foot, foot, GROUND_DROP)
        for index, (a, b) in enumerate(zip(starts, ends)):
            free[index] &= not any(seg_hits_box(a, b, lo, hi) for lo, hi in self.triggerboxes)
        return free, grounded, measures

    def path_relations(self, points):
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        return self.segment_relations(points[:-1], points[1:])

    def segment_feasible(self, a, b):
        free, grounded, _ = self.segment_relations([a], [b])
        return bool(free[0] and grounded[0])

    def settle_many(self, points):
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        projected, _, measures = self.ns.project_many(points, self.mins, self.maxs)
        floors, _ = self.ns.floor_under_many(projected, GROUND_DROP * 3.0,
                                             footprint=(self.maxs[0], self.maxs[1]))
        settled = projected.copy()
        settled[:, 2] = np.where(np.isfinite(floors), floors + self.ride, projected[:, 2])
        free, grounded, _ = self.segment_relations(settled, settled)
        present = free & grounded
        settled[~present] = projected[~present]
        free, grounded, _ = self.segment_relations(settled, settled)
        return settled, free & grounded, measures

    def solve(self, poly):
        current = np.asarray(resample(poly, self.spacing), dtype=np.float64).reshape((-1, 3))
        energy = tangent_energy(current)
        initial = energy
        iterations = 0
        for _ in range(24):
            candidate = current.copy()
            for _ in range(6):
                second = candidate[2:] - 2.0 * candidate[1:-1] + candidate[:-2]
                gradient = np.zeros_like(candidate)
                gradient[:-2] += 2.0 * second
                gradient[1:-1] -= 4.0 * second
                gradient[2:] += 2.0 * second
                candidate[1:-1] -= gradient[1:-1] / 64.0
            candidate[1:-1], present, _ = self.settle_many(candidate[1:-1])
            free, grounded, _ = self.path_relations(candidate)
            next_energy = tangent_energy(candidate)
            if not (np.all(present) and np.all(free & grounded) and next_energy < energy):
                break
            current, energy = candidate, next_energy
            iterations += 1
        return current.tolist(), {'e0': initial, 'e1': energy, 'iterations': iterations}

def track_length(track):
    return sum(math.dist(a, b) for a, b in zip(track, track[1:]))

def trim_track(track, distance):
    for index, (a, b) in enumerate(zip(track, track[1:])):
        length = math.dist(a, b)
        if distance <= length:
            fraction = distance / length if length else 0.0
            point = [a[axis] + fraction * (b[axis] - a[axis]) for axis in range(3)]
            return [point] + [list(value) for value in track[index + 1:]]
        distance -= length
    return [list(track[-1])]

def place_carts(navigation, ns, count, push_radius=160.0, min_length=160.25, origin_separation=64.0):
    navigation.classify_edges(ns, verbose=True)
    nodes, adjacency = navigation.cart_nodes, navigation.cart_adj
    networks = []
    total_pruned = 0
    for component in components(adjacency):
        keep = prune_component(adjacency, component, 2.0 * push_radius)
        graph = subgraph(adjacency, keep)
        endpoints, distances = kcenter(graph, 2, sorted(keep))
        horizon = distances[endpoints[0]][endpoints[-1]] if endpoints else 0.0
        total_pruned += len(component) - len(keep)
        if horizon >= min_length:
            networks.append((horizon, keep))
    networks.sort(key=lambda row: (-row[0], min(row[1])))
    paths, component_rows = [], []
    for index, (horizon, keep) in enumerate(networks):
        slots = len(range(index, count, len(networks)))
        if slots:
            selected, _, candidates = plan_spans(nodes, adjacency, slots, keep)
            component_rows.append({'component': index, 'network_nodes': sorted(keep),
                                   'path_mass': len(selected), 'horizon': horizon, **candidates})
            paths.extend(selected)
    paths, flow = flow_assign(nodes, paths, adjacency, push_radius, metric_adj=navigation.adj)
    solver = PathSolver(ns, triggerboxes=navigation.triggerboxes)
    tracks, realized_paths, fits = [], [], []
    for path in paths:
        seed = [nodes[node].tolist() for node in path]
        track, fit = solver.solve(seed)
        length = track_length(track)
        choices = [trim_track(track, step * origin_separation)
                   for step in range(int(max(0.0, length - min_length) // origin_separation) + 1)]
        chosen = next((choice for choice in choices
                       if track_length(choice) >= min_length
                       and all(math.dist(choice[0], other[0]) >= origin_separation for other in tracks)), None)
        if chosen is None:
            print('nav: origin placement unresolved for stock path %s' % path, file=sys.stderr)
        else:
            tracks.append(chosen)
            realized_paths.append(path)
            fits.append(fit)
    origin_nodes = [min(range(len(navigation.nodes)),
                        key=lambda node: math.dist(track[0], navigation.nodes[node])) for track in tracks]
    attachment = [math.dist(track[0], navigation.nodes[node]) for track, node in zip(tracks, origin_nodes)]
    pool = sorted({node for _, keep in networks for node in keep})
    sites, _ = kcenter(navigation.adj, max(3, count), pool)
    sites = list(dict.fromkeys(origin_nodes + sites + [min(comp) for comp in components(navigation.adj)]))
    owner = navigation.voronoi(sites=sites)
    realization = navigation.realization()
    distances = {node: dijkstra(navigation.adj, node)[0] for node in set(origin_nodes)}
    pairs = [attachment[i] + distances[left][right] + attachment[j]
             for i, left in enumerate(origin_nodes)
             for j, right in enumerate(origin_nodes[i + 1:], i + 1)
             if math.isfinite(distances[left][right])]
    overlap = overlap_matrix(realized_paths)
    network_nodes = sorted({node for _, keep in networks for node in keep})
    network_set = set(network_nodes)
    degrees = [len(adjacency[node].keys() & network_set) for node in network_nodes]
    interior = [degree for degree in degrees if degree >= 2]
    plan = {
        'algorithm': 'pruned_stock_network_spans_signed_counterflow',
        'network_nodes': network_nodes,
        'cart_nodes': nodes.tolist(),
        'network_edges': [[u, v, w] for u in network_nodes for v, w in adjacency[u].items()
                          if u < v and v in network_set],
        'excluded_edges': [[u, v, reason] for (u, v), reason in sorted(navigation.cart_incompatible.items())],
        'components': component_rows,
        'pruned_node_mass': total_pruned,
        'branching_factor': sum(interior) / len(interior) - 1.0 if interior else 0.0,
        'selected_paths': realized_paths,
        'tracks': tracks,
        'curve_fits': fits,
        'requested_lane_mass': count,
        'realized_lane_mass': len(tracks),
        'unresolved_lane_mass': max(0, count - len(tracks)),
        **flow,
    }
    stats = {
        'cart_path_plan': plan,
        'overlap_max': max((value for row in overlap for value in row), default=0.0),
        'headon': flow['headon'], 'push_counterflow': flow['push_zone_counterflow'],
        'flow_alignment': flow['flow_alignment'],
        'origin_orientation_walking_separation': flow['origin_walking_separation'],
        'cart_origin_attachment_distance': attachment,
        'cart_origin_attachment_distance_max': max(attachment, default=None),
        'voronoi_support_node_mass': len(owner),
        'voronoi_assigned_support_node_mass': int((owner >= 0).sum()),
        'voronoi_unassigned_support_node_mass': int((owner < 0).sum()),
        'voronoi_site_mass': len(sites),
        'navigation_realization_id': realization['realization_id'],
        'navigation_realization': realization,
        'cart_origin_selection_method': 'signed_counterflow_then_walking_separation',
        'cart_path_candidate_component_mass': len(networks),
        'cart_path_candidate_node_mass': len(pool),
        'cart_path_selected_component_mass': len(component_rows),
        'cart_origin_navmesh_pairs': len(pairs),
        'cart_origin_navmesh_pairs_expected': len(tracks) * (len(tracks) - 1) // 2,
        'cart_origin_navmesh_min': min(pairs, default=None),
        'cart_origin_navmesh_max': max(pairs, default=None),
        'cart_origin_navmesh_spread_ratio': max(pairs) / min(pairs) if pairs and min(pairs) else None,
    }
    print('nav: network_nodes=%d pruned=%d branching=%.2f candidates=%d lanes=%d/%d overlap=%.3f alignment=%.3f' %
          (len(network_nodes), total_pruned, plan['branching_factor'],
           sum(row['candidate_pair_mass'] for row in component_rows), len(tracks), count,
           stats['overlap_max'], flow['flow_alignment']))
    return tracks, realized_paths, stats
