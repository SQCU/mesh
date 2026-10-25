# design/whole-function.md
import math
import numpy as np
from allocate import equal_finish, project, simplex


def declare(comm, calls):
    """Realize explicit native call descriptors once at a function boundary."""
    import mesh
    made, arrays = [], []
    for index, call in enumerate(calls):
        row = {'group': index, 'repeat': 1, **call}
        for field, ctype in (('counts', 'size_t'), ('matrix', 'size_t'),
                             ('contributors', 'uint64_t'), ('recipients', 'uint64_t')):
            if row.get(field) is not None:
                arrays.append(mesh.ffi.new(f'{ctype}[]', row[field]))
                row[field] = arrays[-1]
        row.setdefault('counts', mesh.ffi.NULL)
        made.append(row)
    result = mesh.ffi.new('ncclMeshPlan **')
    mesh.check(mesh.lib.ncclMeshCommPlan(comm, mesh.ffi.new('ncclMeshCall[]', made or 1), len(made), result))
    return [] if result[0] == mesh.ffi.NULL else [tuple(result[0].entries[g]) for g in range(result[0].groups)]


def function_demands(comm):
    """Return physical fabric and exact semantic demands of the declared graph."""
    import mesh
    from routing import Fabric
    fabric = Fabric.of(comm)
    values = mesh.ffi.new('uint64_t[]', fabric.nodes * fabric.nodes)
    mesh.check(mesh.lib.ncclMeshCommPlanDemands(comm, values, fabric.nodes * fabric.nodes))
    demands = {(a, b): int(values[a * fabric.nodes + b])
               for a in range(fabric.nodes) for b in range(fabric.nodes)
               if a != b and values[a * fabric.nodes + b]}
    return fabric, demands


def route_resolution(comm, nodes):
    """Compiler granularity, before repeats or physical relay amplification."""
    import mesh
    values = mesh.ffi.new('uint64_t[]', nodes * nodes)
    mesh.check(mesh.lib.ncclMeshCommPlanRouteUnits(comm, values, nodes * nodes))
    return {(a, b): int(values[a * nodes + b]) for a in range(nodes) for b in range(nodes)
            if a != b and values[a * nodes + b]}



def domains(decisions, graphs):
    from routing import Fabric
    result = {k: dict(v) for k, v in decisions.items()}
    for name, graph in sorted(graphs.items()):
        fabric = Fabric(graph['nodes'], graph['cables'])
        demands = {(a, b): n for a, b, n in graph['demands'] if n > 0}
        routes = fabric.allocate(demands)[0] if demands else {}
        for key, value in routes.items():
            result[f'flow:{name}/{key}'] = value
    return result


def graph(comm):
    fabric, demands = function_demands(comm)
    return {'nodes': fabric.nodes, 'cables': fabric.cables,
            'demands': [[a, b, n] for (a, b), n in demands.items()],
            'route_units': [[a, b, n] for (a, b), n in route_resolution(comm, fabric.nodes).items()]}


def routes(parts, name):
    prefix = f'flow:{name}/'
    return {k[len(prefix):]: v for k, v in parts.items() if k.startswith(prefix)}


def install(comm, parts, name):
    from routing import Fabric
    Fabric.of(comm).install(comm, routes(parts, name))


def assign(decisions, observations):
    # design/whole-function.md#numerical-operands
    from scipy.optimize import nnls, least_squares
    names = sorted(decisions)
    totals = {k: sum(decisions[k]['parts']) for k in names}
    bounds = {k: (decisions[k].get('low', [0] * len(decisions[k]['parts'])),
                  decisions[k].get('high', [totals[k]] * len(decisions[k]['parts']))) for k in names}
    def realize(k, values):
        d, total = decisions[k], totals[k]
        return (project if d.get('grain', 0) else lambda x, t, g, l, h: simplex(x, t, l, h))(
            values, total, d.get('grain', 0), *bounds[k])
    def vector(parts):
        return np.array([v / totals[k] for k in names for v in parts[k]], dtype=np.float64)
    def parts(vector):
        result, at = {}, 0
        for k in names:
            count = len(decisions[k]['parts'])
            result[k] = realize(k, (vector[at:at + count] * totals[k]).tolist())
            at += count
        return result
    rows = [o for o in observations if set(o['parts']) == set(names)
            and all(len(o['parts'][k]) == len(decisions[k]['parts'])
                    and math.isclose(sum(o['parts'][k]), totals[k]) for k in names)]
    seed, models = {}, {}
    scale = float(np.median([o['seconds'] for o in rows])) if rows else 1.0
    for k in names:
        d, total = decisions[k], totals[k]
        count = len(d['parts'])
        capacities = np.array(d.get('capacity', [1.0] * count), dtype=np.float64)
        capacities /= capacities.sum()
        coefficients, uncertainty = [], []
        for i in range(count):
            samples = [(o['parts'][k][i] / total, o['costs'][k][i] / scale)
                       for o in rows if k in o['costs'] and o['costs'][k][i] is not None]
            prior = np.array([0.0, 1.0 / max(capacities[i], 1 / (1 << 40))])
            x = np.array([[1.0, u] for u, t in samples] + [[1 / 32, 0], [0, 1 / 32]])
            y = np.array([t for u, t in samples] + (prior / 32).tolist())
            fit, _ = nnls(x, y)
            coefficients.append(fit)
            uncertainty.append(np.linalg.inv(x.T @ x).diagonal().tolist())
        if rows and not k.startswith('flow:'):
            prior = np.asarray(coefficients)
            shares = np.asarray([row['parts'][k] for row in rows], dtype=float) / total
            durations = np.asarray([row['seconds'] for row in rows]) / scale
            revisions = list(dict.fromkeys(str(row['implementation']) for row in rows))
            revision_index = [revisions.index(str(row['implementation'])) for row in rows]
            def residual(values):
                coefficients = values.reshape(count, 2)
                predicted = np.max(coefficients[:, 0] + shares * coefficients[:, 1], axis=1)
                differences = []
                for i in range(len(revisions)):
                    indices = [j for j, r in enumerate(revision_index) if r == i]
                    differences.extend((predicted[indices] - durations[indices]
                                        - np.mean(predicted[indices] - durations[indices])).tolist())
                differences.append(float(np.mean(predicted) - np.mean(durations)))
                return np.r_[differences, ((coefficients - prior) / 32).ravel()]
            fit = least_squares(residual, prior.ravel(), bounds=(0, np.inf), max_nfev=64)
            coefficients = fit.x.reshape(count, 2)
            uncertainty = np.linalg.inv(fit.jac.T @ fit.jac + np.eye(2 * count) / 1024).diagonal().reshape(count, 2).tolist()
        models[k] = {'fixed': [v[0] * scale for v in coefficients],
                     'per_unit': [v[1] * scale / total for v in coefficients],
                     'uncertainty': uncertainty}
        if k.startswith('flow:'):
            seed[k] = list(d['parts'])
        else:
            a, b = [v[0] for v in coefficients], [max(v[1], 1 / (1 << 20)) / total for v in coefficients]
            if d.get('grain', 0):
                seed[k] = equal_finish(total, d['grain'], *bounds[k], a, b)[0]
            else:
                seed[k] = realize(k, (total * capacities).tolist())
    centre = vector(seed)
    if rows:
        # Delta y within one implementation removes its fixed effect. All coordinates share one cost.
        x, y = [], []
        for previous, current in zip(rows, rows[1:]):
            if previous['implementation'] == current['implementation']:
                x.append(vector(current['parts']) - vector(previous['parts']))
                y.append((current['seconds'] - previous['seconds']) / scale)
        dimension = len(centre)
        design = np.array(x + (np.eye(dimension) / 32).tolist())
        target = np.array(y + [0.0] * dimension)
        gradient = np.linalg.solve(design.T @ design, design.T @ target)
        centre = vector(rows[-1]['parts']) - gradient * (0.25 / max(1.0, np.linalg.norm(gradient)))
        for k in names:
            if not k.startswith('flow:'):
                start = sum(len(decisions[n]['parts']) for n in names[:names.index(k)])
                centre[start:start + len(seed[k])] += vector(seed)[start:start + len(seed[k])] - vector(rows[-1]['parts'])[start:start + len(seed[k])]
    iteration = len(rows)
    radius = 0.125 / math.sqrt(1 + iteration) if iteration else 0.0
    direction = np.array([1 if ((i + 1) & (1 + iteration // 2)).bit_count() % 2 else -1
                          for i in range(len(centre))], dtype=np.float64)
    return {'parts': parts(centre + (-1 if iteration % 2 else 1) * radius * direction),
            'models': models, 'observations': iteration}
