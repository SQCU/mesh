# design/whole-function.md
import math
import json
import numpy as np
from allocate import equal_finish, project, simplex
from routing import NUMERICS


def declare(comm, calls, *, order=None, routing=None):
    """Realize explicit native call descriptors once at a function boundary."""
    import mesh
    routes, arrays = routing_bindings(routing)
    made = []
    for index, call in enumerate(calls):
        row = {'group': index, 'repeat': 1, **call}
        sources, held = source_bindings(row.pop('sources'))
        arrays.extend(held)
        row.update(sources=sources, source_count=len(call['sources']),
                   route_scope=json_operand(row['route_scope'], arrays),
                   routing=routes if index == 0 else mesh.ffi.NULL,
                   routing_count=len(routing or ()) if index == 0 else 0)
        for field, ctype in (('counts', 'size_t'), ('matrix', 'size_t'),
                             ('contributors', 'uint64_t'), ('recipients', 'uint64_t'),
                             ('production', 'ncclMeshProduction')):
            if row.get(field) is not None:
                arrays.append(mesh.ffi.new(f'{ctype}[]', row[field]))
                row[field] = arrays[-1]
        # the (send, receive) arrays the call is issued with, as the host path passes them: {NULL, address}
        # (nccl.h ncclMeshCall bindings; design/eager-execution.md#what-follows)
        if row.get('bindings') is not None:
            arrays.append(mesh.ffi.new('ncclMeshBufferSlots[2]', [{'buffer': (mesh.ffi.NULL, 0 if a is None else a.ctypes.data),
                                                             'instances': 1} for a in row['bindings']]))
            row['bindings'] = arrays[-1]
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
    return result[0]


def json_operand(value, keepalive):
    """Bind an explicitly declared structured operand without inventing identity."""
    import mesh
    encoded = mesh.ffi.new('char[]', json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=False).encode())
    keepalive.append(encoded)
    return encoded


def source_bindings(records):
    """Bind exact source statements and their supplied occurrence/work relations."""
    import mesh
    held = []
    fields = [{key: json_operand(record[key], held) for key in ('source', 'occurrences', 'work')}
              for record in records]
    source = mesh.ffi.new('ncclMeshSource[]', fields) if fields else mesh.ffi.NULL
    held.append(source)
    return source, held


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



def domains(decisions, graphs, parameters=None, *, layouts=()):
    """Work and retained native choices in one whole-function numerical vector."""
    result = {k: {'model': 'affine', 'identity': {'work': k}, 'total': sum(v['parts']), **v}
              for k, v in decisions.items()}
    result.update(compiler_domains(parameters or {}))
    for name, graph in graphs.items():
        for index, choice in enumerate(route_choices(graph)):
            values, resolution = choice['bytes'], choice['resolution']
            if not sum(values):
                values = choice['weights']
            total = sum(values)
            result[route_key(name, index)] = {
                'parts': [v / total for v in values], 'grain': 0,
                'quantum': 1 / resolution if resolution else 0,
                'low': [0] * len(values), 'high': [1] * len(values), 'model': 'residual', 'total': 1,
                'identity': {'routing': name, **choice['identity']}}
    for index, choice in enumerate(layout_axes(layouts)):
        values = choice['weights']
        if values:
            total = sum(values)
            result[f'layout:{index}'] = {'parts': [v / total for v in values],
                'grain': 0, 'low': [0] * len(values), 'high': [1] * len(values), 'model': 'residual', 'total': 1,
                'identity': {'layout': choice['family']}}
    return result


def choice_domains(decisions):
    """Explicit coordinate meaning retained beside each executed vector."""
    return {key: {'identity': value['identity'], 'model': value['model'],
                  'total': value['total'], 'width': len(value['parts'])}
            for key, value in decisions.items()}


def choice_key(domain):
    # A reversible structural key, not an inferred correspondence or digest.
    return json.dumps(domain, sort_keys=True, separators=(',', ':'))


def routing_arrays(route):
    """Array extents of the public ncclMeshRouting declaration."""
    words = (route['nodes'] + 63) // 64
    return {'cable': ('struct mesh_cable', route['cables']),
            'demand': ('ncclMeshRouteDemand', route['demands']),
            'leaves': ('uint64_t', route['demands'] * words),
            'contributors': ('uint64_t', route['calls'] * words),
            'recipients': ('uint64_t', route['calls'] * words),
            'tree': ('ncclMeshRouteTree', route['trees']),
            'parents': ('uint32_t', route['trees'] * route['nodes']),
            'path': ('ncclMeshRoutePath', route['paths']),
            'first': ('uint32_t', route['paths'] + 1), 'links': ('uint32_t', route['hops']),
            'source': ('ncclMeshSource', route['sources']),
            'source_first': ('uint32_t', route['calls'] + 1),
            'scope': ('char *', route['scopes']), 'call_scope': ('uint32_t', route['calls'])}


def routing_records(plan):
    """Read the retained native records; no path or tree is reconstructed here."""
    import mesh
    def value(item):
        if isinstance(item, (int, float)):
            return item
        kind = mesh.ffi.typeof(item)
        if kind.kind == 'array':
            return [value(v) for v in item]
        if kind.kind == 'pointer' and kind.item.cname == 'char':
            return None if item == mesh.ffi.NULL else json.loads(mesh.ffi.string(item))
        return {name: value(getattr(item, name)) for name, _ in kind.fields}
    result = []
    for group in range(plan.groups if plan != mesh.ffi.NULL else 0):
        native = plan.routing[group]
        route = {key: int(getattr(native, key)) for key in ('nodes', 'cables', 'calls', 'demands', 'trees', 'paths', 'hops', 'sources', 'scopes')}
        route.update({key: [value(getattr(native, key)[i]) for i in range(count)]
                      for key, (_, count) in routing_arrays(route).items()})
        result.append(route)
    return result


def routing_bindings(routing):
    """Bind explicit prepared route operands for canonical group declarations."""
    import mesh
    made, kept = [], []
    for route in routing or ():
        fields = {key: route[key] for key in ('nodes', 'cables', 'calls', 'demands', 'trees', 'paths', 'hops', 'sources', 'scopes')}
        for key, (ctype, count) in routing_arrays(route).items():
            if len(route[key]) != count:
                raise ValueError(f'routing {key} must contain its declared {count} entries')
            if key == 'source':
                fields[key], source_held = source_bindings(route[key])
                kept.extend(source_held)
            else:
                values = ([mesh.ffi.NULL if value is None else json_operand(value, kept) for value in route[key]]
                          if key == 'scope' else route[key])
                kept.append(mesh.ffi.new(f'{ctype}[]', values or 1))
                fields[key] = kept[-1]
        made.append(fields)
    return (mesh.ffi.new('ncclMeshRouting[]', made) if made else mesh.ffi.NULL), kept


def graph(comm, plan):
    import mesh
    fabric = mesh.ffi.new('const struct mesh_fabric **')
    mesh.check(mesh.lib.ncclMeshCommFabric(comm, fabric))
    f = fabric[0]
    values = mesh.ffi.new('uint64_t[]', f.nodes * f.nodes)
    mesh.check(mesh.lib.ncclMeshCommPlanDemands(comm, values, f.nodes * f.nodes))
    return {'nodes': f.nodes, 'cables': [{'wire': int(c.wire), 'nodes': list(c.node),
                                        'bits_per_second': int(c.bandwidth)} for c in mesh.ffi.unpack(f.cable, f.cables)],
            'demands': [[a, b, int(values[a * f.nodes + b])] for a in range(f.nodes)
                        for b in range(f.nodes) if a != b and values[a * f.nodes + b]],
            'routing': routing_records(plan)}


def graph_family(graph):
    """Identity of choices, independent of current extents, costs and weights."""
    groups = []
    for route in graph['routing']:
        groups.append({
            **{key: route[key] for key in ('nodes', 'calls', 'leaves', 'contributors', 'recipients', 'parents', 'first', 'links', 'scope', 'call_scope')},
            'cable': [{key: c[key] for key in ('wire', 'node')} for c in route['cable']],
            'demand': [{key: d[key] for key in ('call', 'what', 'type', 'root', 'element_bytes', 'directions', 'form')} for d in route['demand']],
            'tree': [{key: t[key] for key in ('demand', 'adjustable')} for t in route['tree']],
            'path': [{key: p[key] for key in ('kind', 'source', 'destination', 'scope')} for p in route['path']]})
    return {'forms': graph.get('forms', []), 'groups': groups}


def route_axes(graph):
    """Native columns grouped by the allocation simplex they actually control."""
    for group, route in enumerate(graph['routing']):
        axes = {}
        for i, tree in enumerate(route['tree']):
            if tree['adjustable']:
                axes.setdefault(('tree', tree['demand']), []).append(i)
        for i, path in enumerate(route['path']):
            axes.setdefault(('path', (path['scope'], path['kind'], path['source'], path['destination'])), []).append(i)
        for (role, axis), indices in axes.items():
            rows = [route[role][i] for i in indices]
            resolution = (route['demand'][axis]['units'] if role == 'tree'
                          else max((row['resolution'] for row in rows), default=0))
            yield group, role, axis, indices, rows, resolution


def route_choices(graph):
    """One operand per explicit scope and ordered native choice basis.

    Physical group ordinals locate records to update, never identify choices.
    Shared scoped choices retain all their realized byte allocations.
    """
    choices = {}
    for group, role, axis, indices, rows, resolution in route_axes(graph):
        route = graph['routing'][group]
        if role == 'tree':
            demand = route['demand'][axis]
            scope = route['scope'][route['call_scope'][demand['call']]]
            words = (route['nodes'] + 63) // 64
            identity = {'scope': scope, 'role': role,
                'demand': {key: demand[key] for key in ('what', 'type', 'root', 'element_bytes', 'directions', 'form')},
                'leaves': route['leaves'][axis * words:(axis + 1) * words],
                'columns': [route['parents'][i * route['nodes']:(i + 1) * route['nodes']] for i in indices]}
        else:
            scope, kind, source, destination = axis
            scope = route['scope'][scope]
            identity = {'scope': scope, 'role': role, 'kind': kind, 'source': source, 'destination': destination,
                'columns': [[{key: route['cable'][link][key] for key in ('wire', 'node')}
                             for link in route['links'][route['first'][i]:route['first'][i + 1]]] for i in indices]}
        if scope is None:
            raise ValueError('routing feedback requires its declared source choice scope')
        key = choice_key(identity)
        weights = [row['share' if role == 'tree' else 'weight'] for row in rows]
        total = sum(weights)
        weights = [weight / total for weight in weights]
        if key not in choices:
            choices[key] = {'identity': identity, 'locations': [], 'bytes': [0] * len(rows),
                            'weights': weights, 'resolution': 0}
        choice = choices[key]
        if any(not math.isclose(a, b) for a, b in zip(choice['weights'], weights, strict=True)):
            raise ValueError('one routing choice scope has conflicting prepared weights')
        choice['locations'].append((group, role, indices))
        choice['bytes'] = [a + row['bytes'] for a, row in zip(choice['bytes'], rows, strict=True)]
        choice['resolution'] = max(choice['resolution'], resolution)
    return list(choices.values())


def route_key(name, index):
    return 'routing:' + json.dumps([name, index], separators=(',', ':'))


def route_operands(parts, name, graph):
    """Change weights on the native basis; native preparation realizes pages."""
    import copy
    from allocate import project_parts
    routes = copy.deepcopy(graph['routing'])
    for index, choice in enumerate(route_choices(graph)):
        weights = parts[route_key(name, index)]
        if choice['identity']['role'] == 'path':
            weights = project_parts(weights, NUMERICS['route_weight_units'], 1)
        for group, role, indices in choice['locations']:
            for i, weight in zip(indices, weights, strict=True):
                routes[group][role][i]['share' if role == 'tree' else 'weight'] = weight
    return routes


def layout_operands(parts, choices):
    return [{'family': choice['family'], 'weights': list(parts[f'layout:{i}'])}
            for i, choice in enumerate(layout_axes(choices)) if choice['weights']]


def layout_axes(choices):
    """Repeated uses of one declared choice share one numerical operand."""
    result = []
    for choice in choices:
        matches = [prior for prior in result if prior['family'] == choice['family']]
        if matches:
            if matches[0]['weights'] != choice['weights']:
                raise ValueError('one layout choice family has conflicting prepared weights')
        else:
            result.append(choice)
    return result


def layout_family(choices):
    return [choice['family'] for choice in layout_axes(choices)]


def window(observations):
    """Bound fit storage by proposal order, never by asynchronous arrival order."""
    return sorted(observations, key=lambda row: (row['proposal'], str(row.get('episode', ''))))[-NUMERICS['observation_window']:]


def assign(decisions, observations, *, step, family):
    """Pure preparation: explicit proposal ordinal, executed operands and observations.

    Complete-call durations fit the affine stage envelope even when no isolated
    resource timings exist. A ridge fit of its residual moves every coordinate,
    including routes and compiler operands. No observed candidate is gated.
    """
    from scipy.optimize import least_squares, minimize
    if type(step) is not int or step < 0:
        raise ValueError('assignment needs a nonnegative proposal ordinal')
    if family is None:
        raise ValueError('assignment needs its explicit choice family')
    prior_scale = NUMERICS['fit_prior_scale']
    names = sorted(decisions)
    declared = choice_domains(decisions)
    tokens = {k: choice_key(declared[k]) for k in names}
    totals = {k: declared[k]['total'] for k in names}
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
        return {'parts': {}, 'domains': {}, 'models': {}, 'observations': 0, 'operands': {}, 'proposal': step, 'family': family}
    required = {tokens[k] for k in names if declared[k]['model'] == 'affine'}
    rows = []
    for row in observations:
        if row.get('family') != family:
            continue
        schema = row['domains']
        if set(schema) != set(row['parts']):
            raise ValueError('executed parts must carry their exact coordinate domains')
        if required <= {choice_key(value) for value in schema.values()}:
            rows.append(row)
    rows = window(rows)
    # A changing command grouping creates different conditional choice axes.
    # Keep every observed axis as a separate regression block. An absent axis
    # has no value; it is never interpreted as a vector of zero shares.
    columns, width = {}, 0
    for schema in [declared, *(row['domains'] for row in rows)]:
        for domain in schema.values():
            token = choice_key(domain)
            if token not in columns:
                columns[token] = slice(width, width + domain['width'])
                width += domain['width']
    values = np.zeros((len(rows), at))
    available = np.zeros((len(rows), at), dtype=bool)
    observed = np.zeros((len(rows), width))
    contexts = []
    for i, row in enumerate(rows):
        present = {}
        for key, domain in row['domains'].items():
            token = choice_key(domain)
            data = np.asarray(row['parts'][key], dtype=float)
            if (token in present or len(data) != domain['width'] or not np.all(np.isfinite(data))
                    or domain['total'] <= 0 or not math.isclose(float(data.sum()), domain['total'])):
                raise ValueError('invalid executed coordinate domain')
            present[token] = data / domain['total']
            observed[i, columns[token]] = present[token]
        for key in names:
            if tokens[key] in present:
                values[i, offsets[key]] = present[tokens[key]]
                available[i, offsets[key]] = True
        contexts.append((str(row['implementation']), tuple(sorted(present))))
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
            shared = available[i]
            delta = values[i, shared] - current[shared]
            lo, hi = row.get('completion_seconds_bounds', [row['seconds'], row['seconds']])
            if not 0 <= lo <= row['seconds'] <= hi or not math.isfinite(hi):
                raise ValueError('invalid whole-function completion interval')
            weights[i] /= (1 + (delta @ delta) / NUMERICS['probe_radius'] ** 2)
            weights[i] /= (NUMERICS['observation_relative_noise'] ** 2 + ((hi - lo) / scale) ** 2)
        weights *= len(rows) / weights.sum()
    stages = {}
    for k in names:
        if decisions[k]['model'] not in ('affine', 'residual'):
            raise ValueError(f'{k}: unknown declared cost model {decisions[k]["model"]}')
        if decisions[k]['model'] == 'affine':
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
    versions = list(dict.fromkeys(contexts))
    version_ids = [versions.index(context) for context in contexts]
    fitted = initial
    uncertainty = np.full(len(initial), 1 / prior_scale ** 2)
    if rows and count:
        # Separate implementation and conditional-basis offsets from the shared
        # fixed/work-dependent model. An offset may be positive or negative.
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
        result = least_squares(residual, origin, jac=lambda p: residual(p, True),
                               bounds=(np.r_[np.zeros(2 * count), np.full(len(versions), -np.inf)], np.inf),
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
        design = np.zeros_like(observed)
        target = np.zeros(len(rows))
        for revision in versions:
            indices = [i for i, context in enumerate(contexts) if context == revision]
            active = np.array([i for token in revision[1] for i in range(columns[token].start, columns[token].stop)], dtype=int)
            w, v = weights[indices], observed[np.ix_(indices, active)]
            w = w / w.sum()
            y = np.array([rows[i]['seconds'] / scale - prior(values[i]) for i in indices])
            root = np.sqrt(w * len(indices))
            # Zero denotes no deviation for a structurally absent block, not a
            # fabricated execution value. Historical-only axes remain nuisance
            # columns, so their changes are not attributed to compute shares.
            design[np.ix_(indices, active)] = (v - w @ v) * root[:, None]
            target[indices] = (y - w @ y) * root
        fitted_gradient = design.T @ np.linalg.solve(design @ design.T + np.eye(len(rows)) * prior_scale ** 2, target)
        for key in names:
            gradient[offsets[key]] = fitted_gradient[columns[tokens[key]]]
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
    return {'parts': made, 'domains': declared, 'models': models, 'observations': len(rows), 'proposal': step, 'family': family,
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
            'low': [low, 0], 'high': [high, high-low], 'operand': name, 'model': 'residual', 'total': high,
            'identity': {'compiler_operand': name}}
    return result
