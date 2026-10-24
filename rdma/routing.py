"""design/whole-function.md#ahead-of-time"""
import math

class Fabric:
    def __init__(self, nodes, cables):
        self.nodes, self.cables = int(nodes), list(cables)
        self.adjacent = [[] for _ in range(self.nodes)]
        for e, c in enumerate(self.cables):
            a, b = c['nodes']
            self.adjacent[a].append((b, e))
            self.adjacent[b].append((a, e))
        self.paths = {}
        for source in range(self.nodes):
            for destination in range(self.nodes):
                if source == destination:
                    continue
                paths = []

                def visit(at, seen, path):
                    if at == destination:
                        paths.append(path)
                        return
                    for other, cable in self.adjacent[at]:
                        if other not in seen:
                            visit(other, seen | {other}, path + [cable])

                visit(source, {source}, [])
                if not paths:
                    raise ValueError(f'no path from {source} to {destination}')
                self.paths[source, destination] = paths

    @classmethod
    def of(cls, comm):
        import mesh
        out = mesh.ffi.new('struct mesh_fabric **')
        mesh.check(mesh.lib.ncclMeshCommFabric(comm, out))
        f = out[0]
        return cls(f.nodes, [{'wire': int(f.cable[e].wire), 'nodes': list(f.cable[e].node),
                              'bits_per_second': int(f.cable[e].bandwidth)} for e in range(f.cables)])

    def arcs(self, source, path):
        at, arcs = source, []
        for e in path:
            a, b = self.cables[e]['nodes']
            arcs.append(2 * e + (at == b))
            at = b if at == a else a
        return arcs

    def allocate(self, demands, units=4096):
        """design/whole-function.md#ahead-of-time"""
        import numpy as np
        from scipy.optimize import linprog, minimize
        pairs = sorted(demands)
        columns = [(pair, p) for pair in pairs for p in range(len(self.paths[pair]))]
        scale = max(float(v) for v in demands.values())
        capacities = [max(c['bits_per_second'] / 8, 1) for c in self.cables for _ in range(2)]
        rate = max(capacities)
        edge = [[0.0] * (len(columns) + 1) for _ in capacities]
        equality = [[0.0] * (len(columns) + 1) for _ in pairs]
        bounds = []
        for k, (pair, p) in enumerate(columns):
            equality[pairs.index(pair)][k] = 1.0
            for e in self.arcs(pair[0], self.paths[pair][p]):
                edge[e][k] = 1.0
            # Positive latent weights let feedback move onto any path. They do
            # not reserve a transport page on every possible path.
            floor = demands[pair] * 1e-9
            bounds.append((floor / scale, None))
        for e, capacity in enumerate(capacities):
            edge[e][-1] = -capacity / rate
        rhs = [demands[pair] / scale for pair in pairs]
        bounds.append((0, None))
        solved = linprog([0.0] * len(columns) + [1.0], A_ub=edge, b_ub=[0.0] * len(edge),
                         A_eq=equality, b_eq=rhs, bounds=bounds, method='highs')
        if not solved.success:
            raise RuntimeError(f'physical route allocation: {solved.message}')
        optimum = float(solved.x[-1])
        bounds[-1] = (0, optimum * (1 + 1e-8) + 1e-10)
        # Equalize finish times on the optimum-duration face. Minimizing hops
        # first excludes relaying on spare cables, even when it adds no load
        # at the bottleneck. Wire bytes remain distinct from useful bytes.
        incidence = np.asarray(edge)[:, :-1]
        conservation = np.asarray(equality)[:, :-1]
        service = incidence / (np.asarray(capacities) / rate)[:, None]
        limit = (np.asarray(capacities) / rate) * bounds[-1][1]

        def potential(x):
            t = service @ x - optimum
            return float(t @ t + 1e-8 * (x @ x))

        def gradient(x):
            return 2 * (service.T @ (service @ x - optimum) + 1e-8 * x)

        centered = minimize(potential, solved.x[:-1], jac=gradient, method='SLSQP', bounds=bounds[:-1],
                            constraints=[{'type': 'eq', 'fun': lambda x: conservation @ x - rhs,
                                          'jac': lambda x: conservation},
                                         {'type': 'ineq', 'fun': lambda x: limit - incidence @ x,
                                          'jac': lambda x: -incidence}],
                            options={'ftol': 1e-12, 'maxiter': 1000})
        if not centered.success:
            raise RuntimeError(f'physical route service-time minimization: {centered.message}')
        flow = centered.x
        # Average the nullspace between parallel cables. Equal aggregate loads
        # alone can put almost all of one producer on one cable and its relayed
        # complement on another, creating different release-time stalls. This
        # preserves each node-path's flow and distributes its physical paths in
        # capacity proportion; it cannot increase a parallel bundle's max load.
        bundles = {}
        for k, (pair, p) in enumerate(columns):
            nodes, at, weight = [pair[0]], pair[0], 1.0
            for e in self.paths[pair][p]:
                a, b = self.cables[e]['nodes']
                at = b if at == a else a
                nodes.append(at)
                weight *= capacities[2 * e] / rate
            bundles.setdefault(tuple(nodes), []).append((k, weight))
        for bundle in bundles.values():
            amount, weight = sum(flow[k] for k, _ in bundle), sum(w for _, w in bundle)
            for k, w in bundle:
                flow[k] = amount * w / weight
        decisions = {}
        for pair in pairs:
            flows = [float(flow[k]) * scale for k, (commodity, _) in enumerate(columns) if commodity == pair]
            total = max(units, len(flows))
            wanted = [v / sum(flows) * total for v in flows]
            name = f'route:{pair[0]}:{pair[1]}'
            decisions[name] = {'parts': wanted,
                               'grain': 0, 'low': [0] * len(flows), 'high': [total] * len(flows),
                               'edges': sorted({str(e) for path in self.paths[pair] for e in self.arcs(pair[0], path)})}
        return decisions, {'fluid_lower_bound_seconds': optimum * scale / rate,
                           'directed_load_bytes': list(incidence @ flow * scale),
                           'demand_bytes': {f'{a}:{b}': demands[a, b] for a, b in pairs}}

    def install(self, comm, parts):
        import mesh
        ffi, lib = mesh.ffi, mesh.lib
        for (source, destination), paths in self.paths.items():
            name = f'route:{source}:{destination}'
            if name not in parts:
                continue
            weights = parts[name]
            if len(weights) != len(paths) or min(weights) <= 0:
                raise ValueError(f'{name}: one positive weight per physical path required')
            first, cable = [0], []
            for path in paths:
                cable.extend(path); first.append(len(cable))
            arrays = ffi.new('uint32_t[]', first), ffi.new('uint32_t[]', cable), ffi.new('double[]', [math.log(x) for x in weights])
            routes = ffi.new('struct mesh_paths *', {'count': len(paths), 'first': arrays[0], 'cable': arrays[1], 'log_weight': arrays[2]})
            mesh.check(lib.ncclMeshCommWriteRoutes(comm, source, destination, routes))
        ROUTED.add(int(ffi.cast('uintptr_t', comm)))

    def completion_model(self, demands):
        """Directed load Jacobian for one repeated stage, before measurement."""
        return {'capacities': [c['bits_per_second'] / 8 for c in self.cables for _ in range(2)],
                'paths': {f'route:{a}:{b}': {'bytes': demands[a, b],
                          'arcs': [self.arcs(a, path) for path in self.paths[a, b]]} for a, b in sorted(demands)}}


ROUTED = set()  # the communicators (their addresses) install() gave one-way write routes


def routed(comm):
    """Whether install() gave communicator `comm` its one-way write routes (a write compiles only along them)."""
    import mesh
    return int(mesh.ffi.cast('uintptr_t', comm)) in ROUTED


class QueueSample:
    """design/whole-function.md#measurements"""

    def __init__(self, comm, rank):
        import mesh
        self.comm, self.rank = comm, int(rank)
        self.ffi, self.lib, self.check = mesh.ffi, mesh.lib, mesh.check
        count = self.ffi.new('int *')
        self.check(self.lib.ncclMeshCommSample(comm, self.ffi.NULL, 0, count))
        self.count = count[0]
        self.rows = self.ffi.new('struct ncclMeshCableSample[]', max(self.count, 1))
        self.n = count

    def read(self):
        self.check(self.lib.ncclMeshCommSample(self.comm, self.rows, self.count, self.n))
        return [{'cable': int(s.cable), 'peer': int(s.peer), 'wire': int(s.wire),
                 'ready': int(s.ready), 'sent': int(s.sent), 'landed': int(s.landed)} for s in self.rows[0:self.count]]
