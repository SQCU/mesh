"""design/whole-function.md#ahead-of-time"""
import math

from whole import project


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

    def allocate(self, demands, quantum=16384, units=4096):
        """design/whole-function.md#ahead-of-time"""
        from scipy.optimize import linprog
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
            floor = min(float(quantum), demands[pair] / len(self.paths[pair]))
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
        short = linprog([float(len(self.paths[pair][p])) for pair, p in columns] + [0.0],
                        A_ub=edge, b_ub=[0.0] * len(edge), A_eq=equality, b_eq=rhs, bounds=bounds, method='highs')
        if not short.success:
            raise RuntimeError(f'physical route hop minimization: {short.message}')
        decisions = {}
        for pair in pairs:
            flows = [float(short.x[k]) * scale for k, (commodity, _) in enumerate(columns) if commodity == pair]
            total = max(units, len(flows))
            wanted = [v / sum(flows) * total for v in flows]
            name = f'route:{pair[0]}:{pair[1]}'
            decisions[name] = {'parts': project(wanted, total, 1, [1] * len(flows), [total] * len(flows)),
                               'grain': 1, 'low': [1] * len(flows), 'high': [total] * len(flows),
                               'edges': sorted({str(e) for path in self.paths[pair] for e in self.arcs(pair[0], path)})}
        return decisions, {'fluid_lower_bound_seconds': optimum * scale / rate,
                           'directed_load_bytes': [sum(edge[e][k] * short.x[k] * scale for k in range(len(columns))) for e in range(len(edge))],
                           'demand_bytes': {f'{a}:{b}': demands[a, b] for a, b in pairs}}

    def install(self, comm, parts, chunk_pages=16):
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
            mesh.check(lib.ncclMeshCommWriteRoutes(comm, source, destination, routes, chunk_pages))


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
                 'ready': int(s.ready), 'sent': int(s.sent), 'landed': int(s.landed)} for s in self.rows[:self.count]]
