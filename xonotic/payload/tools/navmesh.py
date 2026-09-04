#!/usr/bin/env mesh-python
import hashlib
import json
import math

import numpy as np

import curve_reconcile as CR
import negspace as NS

CART_MIN = NS.CART_RIDER_MIN
CART_MAX = NS.CART_RIDER_MAX

ACTIVATION = 1.0

GROUND_DROP = 96.0

WPF_JUMP = 0x004
WPF_TELEPORT = 0x008

class Navmesh(object):

    def __init__(self, nodes, adj, flags=None, jumplinks=(), triggerboxes=()):
        self.nodes = [list(n) for n in nodes]
        self.adj = [dict(a) for a in adj]
        self.flags = flags or {}
        self.jumplinks = set(jumplinks)
        self.triggerboxes = list(triggerboxes)
        self.cart_incompatible = {}

    def classify_edges(self, ns, verbose=False, ride=24.0):
        cart_incompatible = {}
        stat = {'semantic': 0, 'burrow': 0, 'airborne': 0}
        geometric = []
        for u, a in enumerate(self.adj):
            for v in a:
                if u >= v:
                    continue
                k = (u, v)
                if k in cart_incompatible:
                    continue
                if self._cart_incompatible_semantics(u, v):
                    cart_incompatible[k] = 'semantic'
                    stat['semantic'] += 1
                    continue
                geometric.append((u, v))
        starts = np.asarray([self.nodes[u] for u, _ in geometric], dtype=np.float64)
        ends = np.asarray([self.nodes[v] for _, v in geometric], dtype=np.float64)
        free, supported, measures = ns.segment_relations(
            starts, ends, CART_MIN, CART_MAX, ACTIVATION,
        )
        for index, edge in enumerate(geometric):
            if not free[index]:
                cart_incompatible[edge] = 'burrow'
                stat['burrow'] += 1
            elif not supported[index]:
                cart_incompatible[edge] = 'airborne'
                stat['airborne'] += 1
        self.cart_incompatible = cart_incompatible
        self.classification_measures = measures
        if verbose:
            ne = sum(len(a) for a in self.adj) // 2
            print('navmesh: %d nodes %d links; %d links are not cart segments '
                  '(semantic jump/teleport=%d, burrows through solid=%d, '
                  'floats over non-walkable air=%d)'
                  % (len(self.nodes), ne, len(cart_incompatible), stat['semantic'], stat['burrow'],
                     stat['airborne']))
        return cart_incompatible

    def _cart_incompatible_semantics(self, u, v):
        fu = self.flags.get(tuple(round(x, 1) for x in self.nodes[u]), 0)
        fv = self.flags.get(tuple(round(x, 1) for x in self.nodes[v]), 0)
        if (fu | fv) & (WPF_JUMP | WPF_TELEPORT):
            return True
        if (u, v) in self.jumplinks or (v, u) in self.jumplinks:
            return True
        for lo, hi in self.triggerboxes:
            for p in (self.nodes[u], self.nodes[v]):
                if all(lo[a] - 8 <= p[a] <= hi[a] + 8 for a in range(3)):
                    return True
        return False

    def _grounded(self, ns, pa, pb, nsamp=None, ride=24.0):
        return ns.segment_supported(pa, pb, CART_MIN, CART_MAX, ACTIVATION)

    def walk_dist(self, src, allow_cart_incompatible=False):
        import heapq
        n = len(self.nodes)
        d = [math.inf] * n
        prev = [-1] * n
        d[src] = 0.0
        q = [(0.0, src)]
        while q:
            du, u = heapq.heappop(q)
            if du > d[u] + 1e-9:
                continue
            for v, w in self.adj[u].items():
                if (not allow_cart_incompatible
                        and (min(u, v), max(u, v)) in self.cart_incompatible):
                    continue
                nd = du + w
                if nd < d[v] - 1e-9:
                    d[v] = nd
                    prev[v] = u
                    heapq.heappush(q, (nd, v))
        return d, prev

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
                if (min(node, neighbor), max(node, neighbor)) in self.cart_incompatible:
                    continue
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
            'schema': 2,
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
        canonical = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
        record['realization_id'] = hashlib.sha256(canonical).hexdigest()
        return record

    def component_representatives(self):
        seen = set()
        representatives = []
        for start in range(len(self.nodes)):
            if start in seen:
                continue
            pending = [start]
            seen.add(start)
            component = []
            while pending:
                node = pending.pop()
                component.append(node)
                for neighbor in self.adj[node]:
                    if ((min(node, neighbor), max(node, neighbor)) in self.cart_incompatible
                            or neighbor in seen):
                        continue
                    seen.add(neighbor)
                    pending.append(neighbor)
            representatives.append(min(component))
        return representatives

    def equidistant_origins(self, k, pool=None, verbose=False):
        if pool is None:
            pool = [i for i in range(len(self.nodes))]
        if not pool:
            return [], {}
        pool = list(dict.fromkeys(pool))
        pool_set = set(pool)
        seen = set()
        components = []
        for start in range(len(self.nodes)):
            if start in seen:
                continue
            pending = [start]
            seen.add(start)
            component = []
            while pending:
                node = pending.pop()
                if node in pool_set:
                    component.append(node)
                for neighbor in self.adj[node]:
                    if (neighbor in seen
                            or (min(node, neighbor), max(node, neighbor)) in self.cart_incompatible):
                        continue
                    seen.add(neighbor)
                    pending.append(neighbor)
            if component:
                components.append(component)
        pool = max(components, key=lambda values: (len(values), -min(values)))
        D = {}
        D[pool[0]] = self.walk_dist(pool[0])[0]
        finite = [value for value in pool if D[pool[0]][value] < math.inf]
        start = max(finite, key=lambda value: D[pool[0]][value]) if finite else pool[0]
        D[start] = self.walk_dist(start)[0]
        picks = [start]
        while len(picks) < k:
            candidates = [candidate for candidate in pool
                          if candidate not in picks and min(D[p][candidate] for p in picks) < math.inf]
            best = max(candidates, key=lambda candidate: min(D[p][candidate] for p in picks), default=None)
            if best is None:
                break
            picks.append(best)
            D[best] = self.walk_dist(best)[0]

        def spread(ps):
            vals = [D[a][b] for i, a in enumerate(ps) for b in ps[i + 1:]
                    if D[a][b] < math.inf]
            if not vals:
                return 0.0, 0.0, math.inf
            return min(vals), max(vals), (max(vals) / min(vals) if min(vals) else math.inf)

        lo, hi, rat = spread(picks)
        st = {'min': lo, 'max': hi, 'ratio': rat, 'k': len(picks),
              'component_mass': len(components), 'metric_pool_mass': len(pool),
              'disconnected_pool_mass': sum(len(values) for values in components) - len(pool)}
        if verbose:
            print('navmesh: %d cart origins, pairwise navmesh-walking distance '
                  'min=%.0f max=%.0f spread_ratio=%.2f (1.00 = exactly '
                  'equidistant)' % (len(picks), lo, hi, rat))
        return picks, st

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

def tangent_energy(P):
    if len(P) < 3:
        return 0.0
    precision = math.sqrt(np.finfo(np.float64).eps)
    weights = CR.CurveWeights(
        anchor=0.0, strain=0.0, bend=1.0, cusp=0.0,
        tangent_point=0.0, thickness=0.0, length_scale=1.0,
        thickness_scale=1.0, tangent_power=2.0, thickness_power=2.0,
        cusp_epsilon=precision, spatial_epsilon=precision,
    )
    return CR.accumulate(P, P, weights)[1]['bend_energy']

def reconciliation_weights(spacing, mins, maxs):
    precision = math.sqrt(np.finfo(np.float64).eps)
    thickness = max(float(maxs[axis] - mins[axis]) for axis in range(3))
    return CR.CurveWeights(
        anchor=1.0, strain=1.0, bend=1.0, cusp=1.0,
        tangent_point=1.0, thickness=1.0, length_scale=float(spacing),
        thickness_scale=thickness, tangent_power=2.0, thickness_power=2.0,
        cusp_epsilon=precision, spatial_epsilon=float(spacing) * precision,
    )

class PathSolver(object):

    def __init__(self, ns, mins=CART_MIN, maxs=CART_MAX, spacing=128.0,
                 step=1.0, ride=24.0, weights=None):
        self.ns = ns
        self.mins = mins
        self.maxs = maxs
        self.spacing = spacing
        self.step = step
        self.ride = ride
        self.weights = weights or reconciliation_weights(spacing, mins, maxs)
        self.spatial_resolution = spacing * math.sqrt(np.finfo(np.float64).eps)
        self.energy_resolution = math.sqrt(np.finfo(np.float64).eps)

    def feasible(self, p):
        return self.ns.fits(p, self.mins, self.maxs)

    def segment_feasible(self, a, b):
        return self.ns.segment_supported(a, b, self.mins, self.maxs, ACTIVATION)

    def path_relations(self, points):
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        return self.ns.segment_relations(
            points[:-1], points[1:], self.mins, self.maxs, ACTIVATION,
        )

    def project_many(self, points):
        return self.ns.project_many(
            points, self.mins, self.maxs, tolerance=self.spatial_resolution,
        )

    def project(self, p):
        points, distances, _ = self.project_many(np.asarray(p, dtype=np.float64)[None, :])
        return points[0].tolist(), float(distances[0])

    def settle_many(self, points):
        projected, _, measures = self.project_many(points)
        footprint = (max(abs(self.mins[0]), abs(self.maxs[0])),
                     max(abs(self.mins[1]), abs(self.maxs[1])))
        floors, floor_measures = self.ns.floor_under_many(
            projected, GROUND_DROP * 3.0, footprint=footprint,
        )
        settled = projected.copy()
        present = np.isfinite(floors)
        settled[present, 2] = floors[present] + self.ride
        rows = np.flatnonzero(present)
        relation_measures = {
            'segment_mass': 0, 'working_set_mass': 0, 'candidate_pair_mass': 0,
            'plane_evaluation_mass': 0, 'support_face_mass': 0,
            'support_constraint_mass': 0,
        }
        if len(rows):
            fits = self.ns.fits_many(settled[rows], self.mins, self.maxs)
            free, supported, relation_measures = self.ns.segment_relations(
                settled[rows], settled[rows], self.mins, self.maxs, ACTIVATION,
            )
            present[rows] = fits & free & supported
        measures['floor'] = floor_measures
        measures['support_relation'] = relation_measures
        measures['settled_point_mass'] = int(present.sum())
        measures['unsettled_point_mass'] = int((~present).sum())
        return settled, present, measures

    def settle(self, p):
        settled, present, _ = self.settle_many(np.asarray(p, dtype=np.float64)[None, :])
        return settled[0].tolist() if present[0] else None

    def solve(self, poly, pin=(0,), stats=None):
        P = [list(p) for p in resample(poly, self.spacing)]
        n = len(P)
        free, supported, _ = self.path_relations(P)
        infeasible0 = int(np.count_nonzero(~(free & supported)))
        if infeasible0:
            P, present, seed_projection = self.settle_many(P)
            if not np.all(present):
                st = {'e0': tangent_energy(poly), 'e1': 0.0, 'n': 0,
                      'infeasible': infeasible0, 'infeasible_seed': infeasible0,
                      'unplaceable': int((~present).sum()),
                      'airborne': infeasible0, 'max_activation_distance': 0.0,
                      'projection': seed_projection}
                if stats is not None:
                    stats.update(st)
                return [], st
            free, supported, _ = self.path_relations(P)
            infeasible0 = int(np.count_nonzero(~(free & supported)))
            if infeasible0:
                st = {'e0': tangent_energy(P), 'e1': tangent_energy(P), 'n': n,
                      'infeasible': infeasible0, 'infeasible_seed': infeasible0,
                      'unplaceable': infeasible0, 'airborne': infeasible0,
                      'max_activation_distance': 0.0}
                if stats is not None:
                    stats.update(st)
                return [], st
            P = P.tolist()
        if n < 3:
            return P, {'e0': 0.0, 'e1': 0.0, 'n': n,
                       'infeasible': 0, 'infeasible_seed': 0,
                       'unplaceable': 0, 'airborne': 0,
                       'max_activation_distance': 0.0}
        pins = set()
        for i in pin:
            pins.add(i % n)
        pins.add(0)
        pins.add(n - 1)

        e0 = tangent_energy(P)
        reference = np.asarray(P, dtype=np.float64)
        current = reference.copy()
        gradient, initial_measures = CR.accumulate(current, reference, self.weights)
        measures = initial_measures
        iteration_mass = 0
        projection_measures = {
            'projection_sweep_mass': 0,
            'candidate_pair_mass': 0,
            'plane_evaluation_mass': 0,
            'directional_null_pair_mass': 0,
            'world_boundary_reconciliation_mass': 0,
        }
        movable = np.asarray([index for index in range(n) if index not in pins], dtype=np.int64)
        while True:
            for index in pins:
                gradient[index] = 0.0
            gradient_norm = np.linalg.norm(gradient, axis=1)
            maximum_gradient = float(gradient_norm.max())
            if maximum_gradient <= self.energy_resolution:
                break
            direction = -gradient * (self.spacing / maximum_gradient)
            maximum_step = self.spacing
            separation = measures['minimum_nonneighbor_segment_distance']
            if separation > 0.0:
                maximum_step = min(maximum_step, math.nextafter(0.5, 0.0) * separation)
            line = min(float(self.step), maximum_step / self.spacing)
            next_state = None
            while line * self.spacing >= self.spatial_resolution:
                candidate = current + line * direction
                for index in pins:
                    candidate[index] = reference[index]
                realized = candidate.copy()
                settled, present, projection = self.settle_many(candidate[movable])
                for name in projection_measures:
                    projection_measures[name] += projection[name]
                projection_mass = int((~present).sum())
                realized[movable[present]] = settled[present]
                free, supported, _ = self.path_relations(realized)
                segment_residual = int(np.count_nonzero(~(free & supported)))
                next_gradient, next_measures = CR.accumulate(
                    realized, reference, self.weights
                )
                if (projection_mass == 0 and segment_residual == 0
                        and next_measures['total_energy'] < measures['total_energy']):
                    next_state = realized, next_gradient, next_measures
                    break
                line *= 0.5
            if next_state is None:
                break
            previous_energy = measures['total_energy']
            current, gradient, measures = next_state
            iteration_mass += 1
            if (previous_energy - measures['total_energy']
                    <= self.energy_resolution * max(1.0, previous_energy)):
                break
        P = current.tolist()
        free, supported, _ = self.path_relations(P)
        infeasible = int(np.count_nonzero(~(free & supported)))
        unplaceable = infeasible
        airborne = infeasible
        maxdev = 0.0
        st = {'e0': e0, 'e1': tangent_energy(P), 'n': n,
              'infeasible': infeasible, 'infeasible_seed': infeasible0,
              'unplaceable': unplaceable,
              'airborne': airborne, 'max_activation_distance': maxdev,
              'reconciliation_iteration_mass': iteration_mass,
              'projection': projection_measures,
              'reconciliation_initial': initial_measures,
              'reconciliation_final': measures}
        if stats is not None:
            stats.update(st)
        return P, st
