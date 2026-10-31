"""Frontend sharding declarations normalized to preparation-time page owners.

The frontend supplies types and placements; the retained numerical IR supplies
access relations. Neither record is an execution object.
"""
import itertools

from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._sharding_prop import ShardingPropagator
from torch.utils import _pytree as pytree


def from_spec(spec):
    """Keep a declared global type and every rank's ordered page selection."""
    from .partition import _rank_selection
    mesh = spec.mesh
    coordinates = itertools.product(*(range(mesh.size(i)) for i in range(mesh.ndim)))
    owners = {rank: dict(zip(('selection', 'shape'), _rank_selection(spec, rank)))
              for rank in coordinates}
    return dict(type=spec.tensor_meta, owners=owners,
                dimensions=tuple(i for i, placement in enumerate(spec.placements) if placement.is_shard()),
                reductions=tuple((i, placement) for i, placement in enumerate(spec.placements)
                                 if placement.is_partial()),
                mesh=mesh, spec=spec)


def partial_expression(source, value, placement, mesh, axis):
    """Declare a pending reduction, including its existing numerical meaning.

    A placement can consume preparation state while describing its value. Its
    complete expression is therefore captured once per source and placement,
    before output domains are split among uses. State tensor identities become
    ordinary operands; the placement object never enters the runtime graph.
    """
    from .compilation import source_expression
    from .declarations import _bound
    held = source.meta.setdefault('mesh_partial_expressions', {})
    key = (id(value), id(placement), id(mesh), axis)
    if key not in held:
        module, operands = source_expression(lambda tensor: placement._reduce_value(tensor, mesh, axis), value)
        origins = source.meta.get('mesh_source_nodes', (source,))
        for node in module.graph.nodes:
            node.meta.setdefault('mesh_source_nodes', origins)
        result = next(node for node in module.graph.nodes if node.op == 'output').args[0]
        held[key] = dict(graph=module, operands=tuple(operands), result=result,
                         boundaries=tuple(node for node in module.graph.nodes if _bound(node)),
                         placement=placement, mesh=mesh, axis=axis)
    return held[key]


def _source_leaves(nodes):
    """Original graph values and result fields, independent of lowered names."""
    for node in nodes:
        for path, value in pytree.tree_flatten_with_path(node.meta['val'])[0]:
            yield (node, tuple(path)), value


def port_bindings(bindings, ports, algebra):
    """Compose semantic value coordinates with a section's actual storage.

    A port selection maps local expression ordinals to the declared value's
    coordinates. It changes neither that value's identity nor the expression's
    scalar type, storage extent or numerical meaning.
    """
    result = {}
    for key, binding in bindings.items():
        port = ports.get(key)
        if port is None:
            result[key] = binding
            continue
        order = algebra.selection_relation(port['selection'])
        target = port['field']
        value = pytree.key_get(target[0].meta['val'], target[1])
        made = dict(binding, relation=order.inverse((value.numel(),)).then(binding['relation']))
        if target not in result:
            result[target] = made
        else:
            previous = result[target]
            if all(previous[name] == made[name] for name in ('name', 'dtype', 'storage_bytes')):
                result[target] = dict(previous, relation=previous['relation'].union(made['relation']))
            else:
                raise ValueError('distinct storage ports need distinct source fields and a declared page equality')
    return result


def port_requests(requested, ports, algebra, *, coordinate=None, incoming=False):
    """Transport requested coordinates across declared source-expression ports."""
    result = {}
    if incoming:
        for key, positions in requested.items():
            port = ports.get(key)
            if port is None:
                _union(result, key, positions)
            else:
                selection = port.get('selections', {}).get(coordinate, port['selection'])
                _union(result, port['field'], algebra.selection_relation(selection).image(positions))
    else:
        bound = set()
        for key, port in ports.items():
            if port['field'] not in requested:
                continue
            selection = port.get('selections', {}).get(coordinate, port['selection'])
            result[key] = algebra.selection_relation(selection).preimage(requested[port['field']])
            bound.add(port['field'])
        for key, positions in requested.items():
            if key not in bound:
                result[key] = positions
    return result


def section_fields(section, side):
    ports = section.get('ports', {})
    return tuple(dict.fromkeys(ports.get(key, {}).get('field', key)
                              for key, _ in _source_leaves(section[side])))


def bind_ports(section, ports, algebra):
    """Apply the same port relation to numerical and collective sections."""
    made = dict(section, ports=ports)
    if not section['boundaries']:
        for side in ('input_bindings', 'output_bindings'):
            made[side] = port_bindings(section[side], ports, algebra)
    return made


def source_uses(module, replacements, results):
    """Give distinct typed source versions their own ordinary FX inputs.

    Only source edges are copied. No numerical operation is evaluated or
    captured again. The returned port map retains original result identities.
    """
    import operator
    import torch
    from torch.fx.node import map_arg
    graph, bound, versions, ports = torch.fx.Graph(), {}, {}, {}
    for node in module.graph.nodes:
        if node.op == 'placeholder':
            made = graph.placeholder(node.name)
            made.meta = node.meta.copy()
            bound[node] = made
    for replacement in replacements.values():
        for field in replacement.values():
            if field not in versions:
                node, path = field
                made = graph.placeholder(f'value_{len(versions)}')
                made.meta['val'] = pytree.key_get(node.meta['val'], path)
                versions[field] = made
                ports[made, ()] = dict(field=field, selection=dict(shape=list(made.meta['val'].shape), steps=[]))

    def field(node, path):
        value, declared = bound[node], node.meta['val']
        for key in path:
            if isinstance(key, pytree.GetAttrKey):
                value = graph.call_function(getattr, (value, key.name))
            else:
                value = graph.call_function(operator.getitem, (value, key.idx if isinstance(key, pytree.SequenceKey) else key.key))
            declared = key.get(declared)
            value.meta['val'] = declared
        return value

    def arguments(node, side):
        original = getattr(node, side)
        def replace(path, value):
            chosen = replacements.get((node, side, tuple(path)))
            if chosen is None:
                return bound[value] if isinstance(value, torch.fx.Node) else value
            return pytree.tree_map_with_path(lambda field_path, described:
                versions[chosen[tuple(field_path)]] if tuple(field_path) in chosen else field(value, field_path)
                if isinstance(described, torch.Tensor) else described, value.meta['val'])
        return pytree.tree_map_with_path(replace, original)

    for node in module.graph.nodes:
        if node.op == 'placeholder':
            continue
        if node.op == 'output':
            graph.output(map_arg(node.args[0], bound.__getitem__))
            continue
        made = graph.create_node(node.op, node.target, arguments(node, 'args'), arguments(node, 'kwargs'), node.name)
        made.meta = node.meta.copy()
        bound[node] = made
    for node, made in bound.items():
        for key, value in _source_leaves((node,)):
            if isinstance(value, torch.Tensor):
                ports[made, key[1]] = dict(field=results.get(key, key), selection=dict(shape=list(value.shape), steps=[]))
    from .compilation import _source_results
    copied = torch.fx.GraphModule(module, graph)
    copied.meta = dict(module.meta, mesh_source_results=_source_results(module, bound))
    metadata = tuple(node.meta['val'] for node in graph.nodes if node.op == 'placeholder')
    return dict(graph=copied, operands=metadata), ports, bound


def prepare_expression(expression):
    """Retain one numerical IR per section of an already captured expression.

    Collective and provider sections remain their original declarations. A
    section's source inputs/results keep their tensor-field identities while
    the common numerical compiler performs its ordinary source rewrites.
    """
    from .compilation import operands as compiler_operands
    configuration = dict(compiler_operands())
    if 'prepared' in expression and expression['prepared']['configuration'] == configuration:
        return expression['prepared']
    import torch
    from torch.fx.node import map_arg
    from mesh_relation import Algebra
    from . import host, prepare_source
    from .declarations import _bound, normalize_edges, sections
    from .native import prepare_graph
    module = expression['graph']
    aliases = expression.setdefault('aliases', {})
    aliases.update(normalize_edges(module))
    def current(node):
        while node in aliases:
            node = aliases[node]
        return node
    algebra = expression.get('algebra') or Algebra()
    arguments = dict(zip((node for node in module.graph.nodes if node.op == 'placeholder'),
                         expression['operands'], strict=True))
    order = {node: index for index, node in enumerate(module.graph.nodes)}
    cuts = {at for node, index in order.items() if _bound(node) or host.bound(node) for at in (index, index + 1)}
    materializations = tuple(dict.fromkeys(current(node) for node in expression.get('materializations', ())))
    cuts.update(order[node] + 1 for node in materializations)
    cuts.update(order[current(node)] + {'before': 0, 'after': 1}[side] for node, side in expression.get('cuts', ()))
    spans = []
    for fragment, inputs, outputs in sections(module, sorted(cuts), requested_outputs=materializations):
        boundary = tuple(node for node in fragment.graph.nodes if _bound(node) or host.bound(node))
        span = dict(graph=fragment, inputs=inputs, outputs=outputs, boundaries=boundary)
        if not boundary:
            actual = tuple(arguments[node] if node in arguments else node.meta['val'] for node in inputs)
            lowered, metadata, operands = prepare_source(fragment, actual)
            results = next(node.args[0] for node in lowered.graph.nodes if node.op == 'output')
            materializations = [value for value in pytree.tree_leaves(map_arg(results, lambda node: node.meta['val']))
                                if isinstance(value, torch.Tensor)]
            retained = prepare_graph(lowered, metadata, materializations=materializations)
            input_values = pytree.tree_leaves(tuple(retained.source_values[node] for node in lowered.graph.nodes
                                                   if node.op == 'placeholder'))
            output_values = pytree.tree_leaves(retained.graph_outputs)
            def bindings(original, values):
                return {key: retained.value_binding(value, algebra)
                        for (key, described), value in zip(_source_leaves(original), values, strict=True)
                        if isinstance(described, torch.Tensor)}
            span.update(lowered=lowered, metadata=metadata, operands=operands, retained=retained,
                        input_bindings=bindings(inputs, input_values), output_bindings=bindings(outputs, output_values),
                        source_values={node: map_arg(tree, retained.source_values.__getitem__)
                                       for node, tree in lowered.meta.get('mesh_source_results', {}).items()},
                        effects=retained.effects(algebra))
        spans.append(span)
    prepared = dict(expression=expression, algebra=algebra, aliases=aliases, sections=tuple(spans),
                    configuration=configuration)
    expression['prepared'] = prepared
    return prepared


def prepare_sources(sources, algebra):
    """Join captured state edges, then retain each declaration's own sections."""
    import torch
    from .compilation import compose_sources
    for index, source in enumerate(sources):
        for node in source['expression']['graph'].graph.nodes:
            if node.op not in ('placeholder', 'output'):
                node.meta['mesh_placement_site'] = index
    joined = compose_sources([source['expression'] for source in sources])
    ports, cuts, results = {}, [], []
    for source, mapping in zip(sources, joined['source_maps'], strict=True):
        for (node, path), port in source['ports'].items():
            ports[mapping[node], path] = port
        original = source['expression']['graph']
        last = next(node for node in reversed(list(original.graph.nodes)) if node.op not in ('placeholder', 'output'))
        cuts.append((mapping[last], 'after'))
        results.extend(node for node in pytree.tree_leaves(mapping[source['expression']['result']])
                       if isinstance(node, torch.fx.Node))
    joined.update(algebra=algebra, cuts=tuple(cuts), materializations=tuple(results))
    prepared = prepare_expression(joined)
    normalized = {}
    for (node, path), port in ports.items():
        while node in prepared['aliases']:
            node = prepared['aliases'][node]
        normalized[node, path] = port
    spans = [[] for _ in sources]
    for section in prepared['sections']:
        sites = {node.meta['mesh_placement_site'] for node in section['graph'].graph.nodes
                 if node.op not in ('placeholder', 'output') and 'mesh_placement_site' in node.meta}
        if len(sites) != 1:
            raise ValueError('placement source section lost its declared source-order boundary')
        spans[sites.pop()].append(bind_ports(section, normalized, algebra))
    result = []
    for source, sections in zip(sources, spans, strict=True):
        outputs = set(key for section in sections for key in section_fields(section, 'outputs'))
        uses = set(key for others in spans if others is not sections for section in others
                   for key in section_fields(section, 'inputs'))
        result.append(dict(source, prepared=dict(prepared, sections=tuple(sections)),
                           state_outputs=tuple(outputs.intersection(uses).difference((source['result'],)))))
    return tuple(result)


def _union(mapping, key, positions):
    mapping[key] = positions if key not in mapping else mapping[key].binary(positions, 'union')


def _empty(algebra, rank):
    return algebra.set('{ [' + ','.join(f'i{axis}' for axis in range(rank)) + '] : false }', rank)


def contribution_domains(prepared, values, coordinate, *, uses=None):
    """Retain declared partial fibers from their required source owners.

    Other mesh dimensions may deliver their needed pages. Only a pending
    contribution's declared mesh axes restrict which terms this rank produces.
    Address predicates preserve work that makes no read on a particular edge.
    """
    from .operand import byte_relation
    algebra, limits, pending = prepared['algebra'], {}, []
    uses = {} if uses is None else uses
    for section in prepared['sections']:
        if section['boundaries']:
            continue
        axes, available = {}, {}
        for key, binding in section['output_bindings'].items():
            owner = values.get(key)
            if owner is not None:
                axes.setdefault(binding['name'], set()).update(axis for axis, _ in owner.get('reductions', ()))
        members = section['graph'].meta.get('mesh_source_results', {})
        for key, binding in section['input_bindings'].items():
            required = [use['required'] for use in uses.values()
                        if use['consumer'] in members and use['source'] == key and use['required'] is not None]
            owners = required or ([values[key]] if key in values else [])
            available.setdefault(binding['name'], []).extend((binding, owner) for owner in owners)
        for effect in reversed(section['effects']):
            for domain in reversed(effect['domains']):
                partial = set().union(*(axes.get(access['name'], set()) for access in domain['writes'].values()))
                if not partial:
                    continue
                work = algebra.map(domain['work'], (), lambda x, y: []).domain()
                for access in domain['reads'].values():
                    axes.setdefault(access['name'], set()).update(partial)
                    if access['name'] not in available:
                        continue
                    owned = _empty(algebra, 1)
                    for binding, owner in available[access['name']]:
                        positions = _empty(algebra, 1)
                        for rank, selection in owner['owners'].items():
                            if all(rank[axis] == coordinate[axis] for axis in partial):
                                positions = positions.binary(algebra.selection(selection['selection']), 'union')
                        owned = owned.binary(byte_relation(binding).image(positions), 'union')
                    read = byte_relation(access)
                    if read is None or access.get('unresolved_predicates'):
                        pending.append(dict(domain=domain, access=access, owned_bytes=owned,
                                            requirement='contribution_access'))
                        continue
                    absent = read.image(work).binary(owned, 'subtract')
                    work = work.binary(read.preimage(absent), 'subtract')
                limits[id(domain)] = work
    return limits, tuple(pending)


def project_numerical(span, demands, algebra, *, constraints=None, work=None):
    """Write preimages and read images over the same retained numerical body.

    Demands name original tensor fields in logical scalar positions. Aliases
    meet in explicitly sized storage-byte domains. Unknown index/predicate
    relations remain actual source expressions; no full-storage dependency is
    substituted for them. A selected reduction finish survives an empty fiber.
    """
    from .operand import access_domain, byte_relation
    constraints = {} if constraints is None else constraints
    assigned = work
    needed, exact, pending, records, live_reads = {}, {}, [], [], []
    for key, binding in span['output_bindings'].items():
        if key in demands:
            positions = byte_relation(binding).image(demands[key])
            _union(needed, binding['name'], positions)
            _union(exact, binding['name'], positions)

    def selected(domain):
        work = _empty(algebra, len(domain['work'])) if assigned is None else assigned[id(domain)]
        unresolved = []
        if assigned is None:
            for slot, access in domain['writes'].items():
                requested = needed.get(access['name'])
                if requested is None or requested.empty():
                    continue
                relation = byte_relation(access)
                if relation is None or access.get('unresolved_predicates'):
                    unresolved.append(dict(slot=slot, access=access, requested=requested))
                else:
                    work = work.binary(relation.preimage(requested), 'union')
        if assigned is None and id(domain) in constraints:
            work = work.binary(constraints[id(domain)], 'intersect')
        reads, writes = {}, {}
        for side, result in (('reads', reads), ('writes', writes)):
            for slot, access in domain[side].items():
                projected = access_domain(access, work)
                positions = projected['positions']
                result[slot] = projected
                if side == 'writes':
                    result[slot]['requested'] = needed.get(access['name'], _empty(algebra, 1))
                if side == 'reads' and not work.empty():
                    if positions is None:
                        pending.append(dict(domain=domain, access=access, work=work))
                    else:
                        _union(needed, access['name'], positions)
                        space = access.get('index_space')
                        if access.get('unresolved_predicates') or space is not None and space['values']:
                            live_reads.append(projected)
                        else:
                            _union(exact, access['name'], positions)
        pending.extend(dict(domain=domain, **record) for record in unresolved)
        return dict(work=work, reads=reads, writes=writes, unresolved=tuple(unresolved))

    for effect in reversed(span['effects']):
        projected = []
        if not effect['domains']:
            requested = {dependency.name: needed[dependency.name]
                         for dependency, _ in effect['scheduling_writes'] if dependency.name in needed}
            if requested:
                pending.append(dict(effect=effect, requested=requested))
        for domain in reversed(effect['domains']):
            # The finish and main fiber both name the original result bytes.
            # Selecting either never substitutes one for the other's work.
            finish = selected(domain['finish']) if domain.get('finish') is not None else None
            main = selected(domain)
            projected.append(dict(effect=effect, domain=domain, **main, finish=finish))
        records.extend(projected)
    records.reverse()
    inputs = {key: byte_relation(binding).preimage(exact[binding['name']])
              for key, binding in span['input_bindings'].items() if binding['name'] in exact}
    live_inputs = {key: tuple(record for record in live_reads if record['access']['name'] == binding['name'])
                   for key, binding in span['input_bindings'].items()}
    live_inputs = {key: records for key, records in live_inputs.items() if records}
    return dict(section=span, outputs=dict(demands), inputs=inputs, storage_bytes=needed,
                records=tuple(records), live_reads=tuple(live_reads), live_inputs=live_inputs, pending=tuple(pending))


def compile_assignments(prepared, assignments, outputs, *, delivery, boundary_projection, constraints=None, transitions=()):
    """Compile supplied materialized-value assignments in producer order.

    A section's retained equations determine its input footprint. Delivering
    those inputs does not enlarge or relocate any producer's assigned work.
    Repeated result uses retain their ordered selections independently of work.
    """
    algebra, aliases = prepared['algebra'], prepared['aliases']
    constraints = {} if constraints is None else constraints
    def identity(key):
        node, path = key if isinstance(key, tuple) else (key, ())
        while node in aliases:
            node = aliases[node]
        return node, path
    assigned = {rank: {} for rank in assignments}
    for rank, values in assignments.items():
        for key, selection in values.items():
            _union(assigned[rank], identity(key), algebra.selection(selection))
    ordered = {rank: {key: algebra.selection_relation(value) for key, value in values.items()}
               for rank, values in outputs.items()}
    sections, pending = {rank: [] for rank in assignments}, {rank: [] for rank in assignments}
    inputs, deliveries = {rank: {} for rank in assignments}, []
    external = set(prepared.get('external_fields', ())) or {
        identity(key) for key, _ in _source_leaves(node for node in prepared['expression']['graph'].graph.nodes
                                                if node.op == 'placeholder')}
    produced = {rank: {key: positions for key, positions in values.items() if key in external}
                for rank, values in assigned.items()}

    def route(section, incoming):
        for key in dict.fromkeys(key for values in incoming.values() for key in values):
            source = identity(key)
            needs = {rank: values[key] for rank, values in incoming.items() if key in values}
            record = delivery(section, source, needs, algebra)
            deliveries.append(record)
            for rank, selection in record['sources'].items():
                positions = algebra.selection(selection)
                missing = positions.binary(produced[rank].get(source, _empty(algebra, 1)), 'subtract')
                if not missing.empty():
                    pending[rank].append(dict(source=source, section=section, positions=missing,
                                              requirement='assigned_source_pages'))
                if source in external:
                    _union(inputs[rank], source, positions)

    def append(section, records):
        route(section, {rank: record['inputs'] for rank, record in records.items()})
        for rank, record in records.items():
            record['section'] = section
            sections[rank].append(record)
            pending[rank].extend(record['pending'])
            for key, positions in record['outputs'].items():
                produced[rank][identity(key)] = positions

    def conversion(program, needs):
        requested = {}
        for rank in assignments:
            requested[rank] = {}
            for key, positions in needs.get(rank, {}).items():
                value = pytree.key_get(key[0].meta['val'], key[1])
                requested[rank][key] = dict(shape=list(value.shape), steps=[['reshape', [value.numel()]],
                                                                          ['select', 0, positions.spans()]])
        selected = project_expressions(program['prepared'], requested, boundary_projection=boundary_projection)
        for index, span in enumerate(program['prepared']['sections']):
            records = {rank: projection['sections'][index] for rank, projection in selected['projections'].items()}
            for record in records.values():
                for key in program['state_outputs']:
                    binding = span.get('output_bindings', {}).get(key)
                    if binding is None:
                        continue
                    positions = _empty(algebra, 1)
                    for effect in record['records']:
                        for domain in (effect, effect.get('finish')):
                            if domain is not None:
                                for written in domain['writes'].values():
                                    if written['access']['name'] == binding['name'] and written['positions'] is not None:
                                        positions = positions.binary(written['positions'], 'union')
                    from .operand import byte_relation
                    record['outputs'][key] = byte_relation(binding).preimage(positions)
            append(span, records)

    for section in prepared['sections']:
        fields = section_fields(section, 'outputs')
        needs = {rank: {key: values[identity(key)] for key in fields if identity(key) in values}
                 for rank, values in assigned.items()}
        for rank in assignments:
            for key in fields:
                if identity(key) not in assigned[rank]:
                    pending[rank].append(dict(source=key, section=section, requirement='section_assignment'))
        records = (boundary_projection(section, needs, algebra) if section['boundaries'] else
                   {rank: project_numerical(section, values, algebra, constraints=constraints.get(rank))
                    for rank, values in needs.items()})
        sources = section['graph'].meta.get('mesh_source_results', {})
        for program in transitions:
            transition = program['transition']
            if program['before'] and transition['consumer'] in sources:
                conversion(program, {rank: {program['result']: record['inputs'][program['result']]}
                    if program['result'] in record['inputs'] else {} for rank, record in records.items()})
        from .operand import byte_relation
        for rank, record in records.items():
            for key, reads in record.get('live_inputs', {}).items():
                source = identity(key)
                binding = section['input_bindings'][key]
                available = byte_relation(binding).image(produced[rank].get(source, _empty(algebra, 1)))
                for read in reads:
                    if not read['positions'].binary(available, 'subtract').empty():
                        pending[rank].append(dict(source=source, section=section, access=read,
                                                  requirement='selected_source_reads'))
        for rank, record in records.items():
            record.update(section=section, outputs=needs[rank])
        append(section, records)
        for program in transitions:
            if not program['before'] and program['transition']['consumer'] in sources:
                conversion(program, {rank: {program['result']: assigned[rank][program['result']]}
                    for rank in assignments})
    route(None, {rank: {key: relation.image(relation.domain()) for key, relation in values.items()}
                 for rank, values in ordered.items()})
    return dict(projections={rank: dict(prepared=prepared, outputs=ordered[rank], output_selections=dict(outputs[rank]),
        inputs=inputs[rank], sections=tuple(sections[rank]), pending=tuple(pending[rank])) for rank in assignments},
        deliveries=tuple(deliveries))


def project_expressions(prepared, outputs, *, delivery=None, boundary_projection=None, constraints=None):
    """Project one source program for all declared owners together.

    Result requests retain their ordered selections. At a section boundary,
    ``delivery`` routes the complete set of consumer demands to the declared
    producers before their work is selected. It returns an ordinary delivery
    record whose ``sources`` are the actual producer selections. The boundary
    hook receives all ranks' needs and returns one record per rank.
    """
    algebra, aliases = prepared['algebra'], prepared['aliases']
    constraints = {} if constraints is None else constraints
    def identity(key):
        node, path = key if isinstance(key, tuple) else (key, ())
        while node in aliases:
            node = aliases[node]
        return node, path
    requested = {rank: {} for rank in outputs}
    ordered, selections, deliveries = {}, {}, []
    sections, pending = {rank: [] for rank in outputs}, {rank: [] for rank in outputs}

    def route(section, incoming):
        fields = dict.fromkeys(key for values in incoming.values() for key in values)
        for key in fields:
            needs = {rank: values[key] for rank, values in incoming.items() if key in values}
            if delivery is not None:
                record = delivery(section, identity(key), needs, algebra)
                deliveries.append(record)
                for rank, selection in record['sources'].items():
                    positions = algebra.selection(selection)
                    if not positions.empty():
                        _union(requested[rank], identity(key), positions)
            else:
                for rank, positions in needs.items():
                    _union(requested[rank], identity(key), positions)

    initial = {rank: {} for rank in outputs}
    for rank, values in outputs.items():
        ordered[rank], selections[rank] = {}, dict(values)
        for key, selection in values.items():
            relation = algebra.selection_relation(selection)
            ordered[rank][key] = relation
            _union(initial[rank], identity(key), relation.image(relation.domain()))
    route(None, initial)
    for section in reversed(prepared['sections']):
        needs = {rank: {key: values[key] for key in section_fields(section, 'outputs') if key in values}
                 for rank, values in requested.items()}
        if section['boundaries']:
            records = boundary_projection(section, needs, algebra)
        else:
            records = {rank: project_numerical(section, values, algebra, constraints=constraints.get(rank))
                       for rank, values in needs.items()}
        for rank, record in records.items():
            record.update(section=section, outputs=needs[rank])
            sections[rank].append(record)
            pending[rank].extend(record['pending'])
            for key in needs[rank]:
                requested[rank].pop(key)
        route(section, {rank: record['inputs'] for rank, record in records.items()})
    return dict(projections={rank: dict(prepared=prepared, outputs=ordered[rank],
        output_selections=selections[rank], inputs=dict(values),
        sections=tuple(reversed(sections[rank])), pending=tuple(pending[rank]))
        for rank, values in requested.items()}, deliveries=tuple(deliveries))


def project_expression(prepared, outputs, *, boundary_projection=None, constraints=None):
    """Single-scope use of the same retained section demand pass."""
    if boundary_projection is None:
        from .declarations import boundary_projection
    def boundaries(section, needs, algebra):
        requested = needs[None]
        record = (boundary_projection(section, requested, algebra)
                  if requested or section.get('peer_outputs') else dict(inputs={}, pending=()))
        return {None: record}
    return project_expressions(prepared, {None: outputs}, boundary_projection=boundaries,
                               constraints={None: constraints})['projections'][None]


class _DeclaredTypes(ShardingPropagator):
    """Use the captured result types with the existing sharding registries."""
    def __init__(self, source, result_types):
        self.source, self.result_types = source, result_types

    def __getattr__(self, name):
        return getattr(self.source, name)

    def _propagate_tensor_meta_non_cached(self, schema):
        return self.result_types


def propagate(operation, arguments, keywords, mesh, *, result_types):
    """Consume existing operator schemas without invoking numerical dispatch.

    Tensor leaves are DTensorSpecs, including explicit replicated input specs.
    Result types come from the captured graph; propagation does not infer them
    again by running the operation on another set of fake inputs.
    OutputSharding retains compound results and the required input schema. Its
    input redistributions remain declarations for the shared access compiler.
    Missing operator declarations raise in the upstream propagator; this adapter
    never substitutes replication. Ragged placements retain their configured
    shares when upstream rules change which tensor axis a mesh axis partitions.
    """
    from .partition import _keys, _plain, _retag, _schema
    specs = [value for value in pytree.tree_leaves((arguments, keywords)) if isinstance(value, DTensorSpec)]
    if any(spec.mesh != mesh for spec in specs):
        raise ValueError('operator input specs must name their enclosing ownership mesh')
    keys = _keys(specs)
    propagator = _DeclaredTypes(DTensor._op_dispatcher.sharding_propagator, result_types)
    schema_info = propagator.op_to_schema_info.get(operation)
    if schema_info is None:
        schema_info = propagator.op_to_schema_info_for_single_dim_strategy.get(operation)
    schema = OpSchema(operation, arguments, keywords, schema_info=schema_info)
    result = propagator.propagate_op_sharding_non_cached(_schema(schema, _plain))
    output = pytree.tree_map_only(DTensorSpec, lambda spec: _retag(spec, keys), result.output_spec)
    required = (_schema(result.redistribute_schema, lambda spec: _retag(spec, keys))
                if result.redistribute_schema is not None else None)
    return OutputSharding(output, required, needs_redistribute=result.needs_redistribute,
                          use_val_from_redistribute_schema=result.use_val_from_redistribute_schema)
