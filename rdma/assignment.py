# design/whole-function.md
import math
import numpy as np
from allocate import equal_finish, project, simplex
from routing import NUMERICS


def declare(comm, calls, *, order=None):
    """Realize explicit native call descriptors once at a function boundary."""
    import mesh
    made, arrays = [], []
    for index, call in enumerate(calls):
        row = {'group': index, 'repeat': 1, **call}
        for field, ctype in (('counts', 'size_t'), ('matrix', 'size_t'),
                             ('contributors', 'uint64_t'), ('recipients', 'uint64_t'),
                             ('production', 'ncclMeshProduction')):
            if row.get(field) is not None:
                arrays.append(mesh.ffi.new(f'{ctype}[]', row[field]))
                row[field] = arrays[-1]
        row.setdefault('counts', mesh.ffi.NULL)
        made.append(row)
    sequence, previous = [], None
    for row in made:
        repeat = row.pop('repeat')
        if not sequence or row['group'] != previous:
            sequence.append({'first': len(sequence), 'count': 1, 'repeat': repeat})
        previous = row['group']
    order = sequence if order is None else list(order)
    result = mesh.ffi.new('ncclMeshPlan **')
    mesh.check(mesh.lib.ncclMeshCommPlan(comm, mesh.ffi.new('ncclMeshCall[]', made or 1), len(made),
                                      mesh.ffi.new('ncclMeshPlanRun[]', order or 1), len(order), result))
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



def domains(decisions, graphs, parameters=None):
    from routing import Fabric
    result = {k: dict(v) for k, v in decisions.items()}
    result.update(compiler_domains(parameters or {}))
    for name, graph in sorted(graphs.items()):
        fabric = Fabric(graph['nodes'], graph['cables'])
        demands = {(a, b): n for a, b, n in graph['demands'] if n > 0}
        routes = fabric.allocate(demands)[0] if demands else {}
        for key, value in routes.items():
            if key in graph.get('weights', {}):
                weights = graph['weights'][key]
                total = sum(value['parts'])
                value['parts'] = [total * w / sum(weights) for w in weights]
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
    from scipy.optimize import nnls, minimize
    prior_scale = NUMERICS['fit_prior_scale']
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
    if not names:
        return {'parts': {}, 'models': {}, 'observations': 0, 'operands': {}}
    if any(not math.isfinite(total) or total <= 0 for total in totals.values()):
        raise ValueError('assignment domains require finite positive totals')
    rows = [o for o in observations if set(o['parts']) == set(names)
            and all(len(o['parts'][k]) == len(decisions[k]['parts'])
                    and math.isclose(sum(o['parts'][k]), totals[k]) for k in names)]
    if any(not math.isfinite(o['seconds']) or o['seconds'] <= 0 for o in rows):
        raise ValueError('whole-function durations must be finite and positive')
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
                       for o in rows if k in o.get('costs', {}) and o['costs'][k][i] is not None]
            prior = np.array([0.0, 1.0 / max(capacities[i], 2.0 ** -NUMERICS['capacity_floor_bits'])])
            x = np.array([[1.0, u] for u, t in samples] + [[prior_scale, 0], [0, prior_scale]])
            y = np.array([t for u, t in samples] + (prior * prior_scale).tolist())
            fit, _ = nnls(x, y)
            coefficients.append(fit)
            uncertainty.append(np.linalg.inv(x.T @ x).diagonal().tolist())
        models[k] = {'fixed': [v[0] * scale for v in coefficients],
                     'per_unit': [v[1] * scale / total for v in coefficients],
                     'uncertainty': uncertainty}
        if k.startswith('flow:') or 'operand' in d:
            seed[k] = list(d['parts'])
        else:
            a, b = [v[0] for v in coefficients], [max(v[1], 2.0 ** -NUMERICS['slope_floor_bits']) / total for v in coefficients]
            if d.get('grain', 0):
                seed[k] = equal_finish(total, d['grain'], *bounds[k], a, b)[0]
            else:
                seed[k] = realize(k, (total * capacities).tolist())
    # Every coordinate uses one complete-call objective. The affine fits are
    # its prior, not a second update which overwrites measured route/compute
    # changes. Shared stage labels express concurrent work; stages add.
    offsets, at = {}, 0
    for k in names:
        offsets[k] = slice(at, at + len(decisions[k]['parts']))
        at += len(decisions[k]['parts'])
    stages = {}
    for k in names:
        if not k.startswith('flow:') and 'operand' not in decisions[k]:
            stages.setdefault(decisions[k].get('stage', k), []).append(k)
    temperature = NUMERICS['cost_smoothing']
    def prior(value):
        cost = 0.0
        for keys in stages.values():
            tails = np.concatenate([np.asarray(models[k]['fixed']) / scale
                + value[offsets[k]] * np.asarray(models[k]['per_unit']) * totals[k] / scale for k in keys])
            # Stable log-sum-exp, with no subnormal exponential operands.
            top = float(np.max(tails))
            cost += top + temperature * math.log(float(np.exp(np.maximum((tails-top)/temperature,
                                                       -NUMERICS['exponent_limit'])).sum()))
        return cost
    latest = vector({k: decisions[k]['parts'] for k in names}) if rows else vector(seed)
    gradient = np.zeros(len(latest))
    if rows:
        # Importance weights belong to the observation. Arrival order does not
        # define a finite difference: centre all episodes of each revision.
        design, target = [], []
        for revision in dict.fromkeys(str(row['implementation']) for row in rows):
            samples = [row for row in rows if str(row['implementation']) == revision]
            weights = np.array([float(row.get('importance', 1.0)) for row in samples])
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError('observation importance must be finite and positive')
            weights /= weights.sum()
            values = np.array([vector(row['parts']) for row in samples])
            residual = np.array([row['seconds'] / scale - prior(value) for row, value in zip(samples, values)])
            root = np.sqrt(weights * len(samples))
            design.extend((values - weights @ values) * root[:, None])
            target.extend((residual - weights @ residual) * root)
        design, target = np.asarray(design), np.asarray(target)
        # The dual ridge system scales with observed episodes, not path count.
        gradient = design.T @ np.linalg.solve(design @ design.T + np.eye(len(rows)) * prior_scale ** 2, target)
    lower = np.array([v / totals[k] for k in names for v in bounds[k][0]])
    upper = np.array([v / totals[k] for k in names for v in bounds[k][1]])
    def objective(value):
        delta = value - latest
        return prior(value) + gradient @ delta + NUMERICS['joint_curvature'] * (delta @ delta)
    constraints = [{'type': 'eq', 'fun': lambda value, span=span: np.sum(value[span]) - 1.0}
                   for span in offsets.values()]
    solved = minimize(objective, vector(seed), method='SLSQP', bounds=list(zip(lower, upper)), constraints=constraints,
                      options={'maxiter': NUMERICS['fit_evaluations'], 'ftol': NUMERICS['route_function_tolerance']})
    # Numerical projection is the compiler's one realization boundary.
    centre = vector(parts(solved.x))
    iteration = len(rows)
    radius = NUMERICS['probe_radius'] / math.sqrt(1 + iteration) if iteration else 0.0
    direction = np.array([1 if ((i + 1) & (1 + iteration // 2)).bit_count() % 2 else -1
                          for i in range(len(centre))], dtype=np.float64)
    made = parts(centre + (-1 if iteration % 2 else 1) * radius * direction)
    return {'parts': made, 'models': models, 'observations': iteration,
            'operands': {d['operand']: made[k][0] for k, d in decisions.items() if 'operand' in d}}


def compiler_domains(parameters):
    """Explicit bounded compiler operands in the same numerical vector.

    Each parameter supplies value/minimum/maximum/grain. Its complement keeps
    the common simplex representation. No backend resource limit is inferred.
    Callers bind the resulting operands while compiling the next function.
    """
    result = {}
    for name, p in parameters.items():
        low, high, value = p['minimum'], p['maximum'], p['value']
        if not 0 <= low <= value <= high or high <= 0:
            raise ValueError(f'{name}: invalid compiler operand bounds')
        result['compile:' + name] = {'parts': [value, high-value], 'grain': p['grain'],
            'low': [low, 0], 'high': [high, high-low], 'operand': name, 'capacity': [1, 1]}
    return result
