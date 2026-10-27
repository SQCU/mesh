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
        quanta = {(a, b): n for a, b, n in graph['route_units']}
        for key, value in routes.items():
            _, source, destination = key.split(':')
            units = quanta[int(source), int(destination)]
            if units <= 0:
                raise ValueError(f'{name}/{key}: missing physical route resolution')
            value['quantum'] = 1 / units
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
    from routing import weight_ticks
    prefix = f'flow:{name}/'
    return {k[len(prefix):]: weight_ticks(v) for k, v in parts.items() if k.startswith(prefix)}


def install(comm, parts, name):
    from routing import Fabric
    Fabric.of(comm).install(comm, routes(parts, name))


def window(observations):
    """Bound fit storage by proposal order, never by asynchronous arrival order."""
    return sorted(observations, key=lambda row: (row['proposal'], str(row.get('episode', ''))))[-NUMERICS['observation_window']:]


def assign(decisions, observations, *, step):
    """Pure preparation: explicit proposal ordinal, executed operands and observations.

    Complete-call durations fit the affine stage envelope even when no isolated
    resource timings exist. A ridge fit of its residual moves every coordinate,
    including routes and compiler operands. No observed candidate is gated.
    """
    from scipy.optimize import least_squares, minimize
    if type(step) is not int or step < 0:
        raise ValueError('assignment needs a nonnegative proposal ordinal')
    prior_scale = NUMERICS['fit_prior_scale']
    names = sorted(decisions)
    totals = {k: sum(decisions[k]['parts']) for k in names}
    if any(not math.isfinite(total) or total <= 0 for total in totals.values()):
        raise ValueError('assignment domains require finite positive totals')
    bounds = {k: (decisions[k].get('low', [0] * len(decisions[k]['parts'])),
                  decisions[k].get('high', [totals[k]] * len(decisions[k]['parts']))) for k in names}
    def realize(k, values):
        d, total = decisions[k], totals[k]
        return (project if d.get('grain', 0) else lambda x, t, g, l, h: simplex(x, t, l, h))(
            values, total, d.get('grain', 0), *bounds[k])
    def vector(parts):
        return np.array([v / totals[k] for k in names for v in parts[k]], dtype=np.float64)
    def parts(value):
        return {k: realize(k, (value[offsets[k]] * totals[k]).tolist()) for k in names}
    offsets, at = {}, 0
    for k in names:
        offsets[k] = slice(at, at + len(decisions[k]['parts']))
        at += len(decisions[k]['parts'])
    if not names:
        return {'parts': {}, 'models': {}, 'observations': 0, 'operands': {}, 'proposal': step}
    rows = window([o for o in observations if set(o['parts']) == set(names)
                   and all(len(o['parts'][k]) == len(decisions[k]['parts'])
                           and math.isclose(sum(o['parts'][k]), totals[k]) for k in names)])
    if any(not math.isfinite(o['seconds']) or o['seconds'] <= 0 for o in rows):
        raise ValueError('whole-function durations must be finite and positive')
    scale = float(np.median([o['seconds'] for o in rows])) if rows else 1.0
    weights = np.array([float(o.get('importance', 1.0)) for o in rows])
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('observation importance must be finite and positive')
    if rows:
        current = vector({k: decisions[k]['parts'] for k in names})
        # Local regression consumes late observations at their executed point.
        # Explicit importance and coarse clock uncertainty stay data operands.
        for i, row in enumerate(rows):
            delta = vector(row['parts']) - current
            lo, hi = row.get('completion_seconds_bounds', [row['seconds'], row['seconds']])
            if not 0 <= lo <= row['seconds'] <= hi or not math.isfinite(hi):
                raise ValueError('invalid whole-function completion interval')
            weights[i] /= (1 + (delta @ delta) / NUMERICS['probe_radius'] ** 2)
            weights[i] /= (NUMERICS['observation_relative_noise'] ** 2 + ((hi - lo) / scale) ** 2)
        weights *= len(rows) / weights.sum()
    stages = {}
    for k in names:
        if not k.startswith('flow:') and 'operand' not in decisions[k]:
            # Structured native stages can be lists; names are immutable operands.
            stages.setdefault(str(decisions[k].get('stage', k)), []).append(k)
    model_keys = [k for keys in stages.values() for k in keys]
    spans, coefficients, count = {}, [], 0
    for k in model_keys:
        capacities = np.array(decisions[k].get('capacity', [NUMERICS['unknown_capacity']] * len(decisions[k]['parts'])), dtype=float)
        if not np.all(np.isfinite(capacities)) or np.any(capacities <= 0):
            raise ValueError(f'{k}: capacities must be finite and positive')
        capacities /= capacities.sum()
        spans[k] = slice(count, count + len(capacities))
        count += len(capacities)
        coefficients.extend(1 / np.maximum(capacities, 2.0 ** -NUMERICS['capacity_floor_bits']) / len(stages))
    initial = np.r_[np.zeros(count), coefficients]
    temperature = NUMERICS['cost_smoothing']
    def envelope(value, fitted):
        cost, jac = 0.0, np.zeros(len(fitted))
        for keys in stages.values():
            ids = np.concatenate([np.arange(spans[k].start, spans[k].stop) for k in keys])
            shares = np.concatenate([value[offsets[k]] for k in keys])
            tails = fitted[ids] + fitted[count + ids] * shares
            top = float(np.max(tails))
            exp = np.exp(np.maximum((tails - top) / temperature, -NUMERICS['exponent_limit']))
            probabilities = exp / exp.sum()
            cost += top + temperature * math.log(float(exp.mean()))
            jac[ids], jac[count + ids] = probabilities, probabilities * shares
        return cost, jac
    values = np.array([vector(o['parts']) for o in rows])
    versions = list(dict.fromkeys(str(o['implementation']) for o in rows))
    version_ids = [versions.index(str(o['implementation'])) for o in rows]
    fitted = initial
    uncertainty = np.full(len(initial), 1 / prior_scale ** 2)
    if rows and count:
        # Every complete function constrains the model. Code-version offsets
        # separate implementation changes from placement finite differences.
        origin = np.r_[initial, np.zeros(len(versions))]
        def residual(parameters, jacobian=False):
            costs, derivatives = zip(*(envelope(v, parameters[:2 * count]) for v in values))
            predicted = np.asarray(costs) + parameters[2 * count + np.asarray(version_ids)]
            root = np.sqrt(weights)
            if jacobian:
                data = np.zeros((len(rows), len(origin)))
                data[:, :2 * count] = derivatives
                data[np.arange(len(rows)), 2 * count + np.asarray(version_ids)] = 1
                return np.vstack([data * root[:, None], np.eye(len(origin)) * prior_scale])
            return np.r_[(predicted - np.array([o['seconds'] / scale for o in rows])) * root,
                         (parameters - origin) * prior_scale]
        result = least_squares(residual, origin, jac=lambda p: residual(p, True), bounds=(0, np.inf),
                               max_nfev=NUMERICS['fit_evaluations'])
        fitted = result.x[:2 * count]
        # Diagonal posterior variance via a sample-sized regularized solve.
        data = residual(result.x, True)[:len(rows)]
        inverse = np.linalg.solve(data @ data.T + np.eye(len(rows)) * prior_scale ** 2, data)
        uncertainty = np.maximum(0, 1 - np.sum(data * inverse, axis=0))[:2 * count] / prior_scale ** 2
    models = {k: {'fixed': (fitted[spans[k]] * scale).tolist(),
                  'per_unit': (fitted[count + spans[k].start:count + spans[k].stop] * scale / totals[k]).tolist(),
                  'fixed_variance': (uncertainty[spans[k]] * scale ** 2).tolist(),
                  'per_unit_variance': (uncertainty[count + spans[k].start:count + spans[k].stop] * (scale / totals[k]) ** 2).tolist()}
              for k in model_keys}
    def prior(value):
        return envelope(value, fitted)[0] if count else 0.0
    seed = {k: realize(k, decisions[k]['parts']) for k in names}
    for k in model_keys:
        if decisions[k].get('grain', 0):
            m = models[k]
            seed[k] = equal_finish(totals[k], decisions[k]['grain'], *bounds[k], m['fixed'],
                                  [max(v, scale / totals[k] * 2.0 ** -NUMERICS['slope_floor_bits']) for v in m['per_unit']])[0]
    latest = vector({k: decisions[k]['parts'] for k in names}) if rows else vector(seed)
    gradient = np.zeros(len(latest))
    if rows:
        design, target = [], []
        for revision in versions:
            indices = [i for i, o in enumerate(rows) if str(o['implementation']) == revision]
            w, v = weights[indices], values[indices]
            w = w / w.sum()
            y = np.array([rows[i]['seconds'] / scale - prior(values[i]) for i in indices])
            root = np.sqrt(w * len(indices))
            design.extend((v - w @ v) * root[:, None])
            target.extend((y - w @ y) * root)
        design, target = np.asarray(design), np.asarray(target)
        gradient = design.T @ np.linalg.solve(design @ design.T + np.eye(len(rows)) * prior_scale ** 2, target)
    lower = np.array([v / totals[k] for k in names for v in bounds[k][0]])
    upper = np.array([v / totals[k] for k in names for v in bounds[k][1]])
    def objective(value):
        delta = value - latest
        return prior(value) + gradient @ delta + NUMERICS['joint_curvature'] * (delta @ delta)
    def derivative(value):
        _, jac = envelope(value, fitted) if count else (0, [])
        result = gradient + 2 * NUMERICS['joint_curvature'] * (value - latest)
        for k in model_keys:
            result[offsets[k]] += jac[spans[k]] * fitted[count + spans[k].start:count + spans[k].stop]
        return result
    constraints = [{'type': 'eq', 'fun': lambda value, span=span: np.sum(value[span]) - 1.0,
                    'jac': lambda value, span=span: np.r_[np.zeros(span.start), np.ones(span.stop - span.start), np.zeros(len(value) - span.stop)]}
                   for span in offsets.values() if np.any(lower[span] != upper[span])]
    solved = minimize(objective, vector(seed), jac=derivative, method='SLSQP', bounds=list(zip(lower, upper)), constraints=constraints,
                      options={'maxiter': NUMERICS['fit_evaluations'], 'ftol': NUMERICS['route_function_tolerance']})
    if not np.all(np.isfinite(solved.x)):
        raise ArithmeticError('nonfinite assignment solution')
    centre = vector(parts(solved.x))
    # Paired Walsh directions advance with issued proposals, not retained rows.
    # Each simplex gets centred directions; fixed and one-member axes stay still.
    radius = NUMERICS['probe_radius'] / math.sqrt(1 + step) if step else 0.0
    cycle = 1 << max(1, len(centre).bit_length())
    row = 1 + ((step - 1) // 2) % (cycle - 1)
    direction = np.array([1 if ((i + 1) & row).bit_count() % 2 else -1 for i in range(len(centre))], dtype=float)
    for k, span in offsets.items():
        direction[span] -= direction[span].mean()
        quantum = decisions[k].get('quantum', decisions[k].get('grain', 0) / totals[k])
        direction[span] *= max(radius, quantum) if step else 0
    made = parts(centre + (1 if step % 2 else -1) * direction)
    return {'parts': made, 'models': models, 'observations': len(rows), 'proposal': step,
            'operands': {d['operand']: made[k][0] for k, d in decisions.items() if 'operand' in d}}


def producer_seconds(assigned, decisions, groups):
    """Map model work shares to physical producers using declared holder groups."""
    result = [0.0] * (1 + max((r for members in groups.values() for group in members for r in group), default=-1))
    stages = {}
    for key, holders in groups.items():
        model = assigned['models'][key]
        costs = [a + b * n for a, b, n in zip(model['fixed'], model['per_unit'], assigned['parts'][key])]
        stage = stages.setdefault(str(decisions[key].get('stage', key)), [0.0] * len(result))
        for cost, members in zip(costs, holders, strict=True):
            for rank in members:
                stage[rank] = max(stage[rank], cost)
    return [sum(stage[rank] for stage in stages.values()) for rank in range(len(result))]


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
