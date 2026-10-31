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


def owner_layouts(typ, owners):
    """Retain explicit local layouts or the frontend's dense DTensor layout."""
    import torch
    from torch._prims_common import is_non_overlapping_and_dense_or_false
    from torch.distributed.tensor._utils import compute_local_stride
    dense = is_non_overlapping_and_dense_or_false(torch.empty_strided(typ.shape, typ.stride, device='meta'))
    return {rank: dict(owner, stride=tuple(owner['stride'])) if 'stride' in owner else
            dict(owner, stride=compute_local_stride(typ.stride, owner['shape'])) if dense else dict(owner)
            for rank, owner in owners.items()}


def from_spec(spec):
    """Keep a declared global type and every rank's ordered page selection."""
    from .partition import _rank_selection
    mesh = spec.mesh
    coordinates = itertools.product(*(range(mesh.size(i)) for i in range(mesh.ndim)))
    owners = owner_layouts(spec.tensor_meta, {rank: dict(zip(('selection', 'shape'), _rank_selection(spec, rank)))
                                              for rank in coordinates})
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
    from .compilation import source_expression, source_records
    from .declarations import _bound
    held = source.meta.setdefault('mesh_partial_expressions', {})
    key = (id(value), id(placement), id(mesh), axis)
    if key not in held:
        module, operands = source_expression(lambda tensor: placement._reduce_value(tensor, mesh, axis), value)
        origins = source.meta.get('mesh_source_nodes', (source,))
        for node in module.graph.nodes:
            node.meta.setdefault('mesh_source_nodes', origins)
            node.meta.setdefault('mesh_source_records', source_records(source))
            node.meta.setdefault('mesh_route_scope', source.meta.get('mesh_route_scope'))
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


def binding_uses(bindings):
    """Every typed storage use of each semantic value, in declaration order."""
    return ((field, binding) for field, uses in bindings.items() for binding in uses)


def binding_for_access(bindings, field, access):
    """The declared source use touched by one retained numerical access."""
    return next(binding for binding in bindings[field] if binding['name'] == access['name'])


def declared_binding(binding):
    """Retain paired local-to-value and local-to-storage coordinates."""
    relation = binding['relation']
    identity = relation.algebra.map(relation.shape, relation.shape, lambda x, y: [f'{x[0]} = {y[0]}'])
    return dict(binding, coordinates=((identity, relation),))


def port_bindings(bindings, ports, algebra):
    """Compose semantic value coordinates with a section's actual storage.

    A port selection maps local expression ordinals to the declared value's
    coordinates. It changes neither that value's identity nor the expression's
    scalar type, storage extent or numerical meaning.
    """
    result = {}
    for key, binding in binding_uses(bindings):
        port = ports.get(key)
        if port is None:
            target, made = key, binding
        else:
            order = algebra.selection_relation(port['selection'])
            target = port['field']
            value = pytree.key_get(target[0].meta['val'], target[1])
            made = dict(binding, relation=order.inverse((value.numel(),)).then(binding['relation']),
                        coordinates=tuple((source.then(order), storage) for source, storage in binding['coordinates']))
        uses = result.setdefault(target, [])
        at = next((index for index, previous in enumerate(uses)
                   if all(previous[name] == made[name] for name in ('name', 'dtype', 'storage_bytes'))), None)
        if at is None:
            uses.append(made)
        else:
            uses[at] = dict(made, relation=uses[at]['relation'].union(made['relation']),
                            coordinates=(*uses[at]['coordinates'], *made['coordinates']))
    return {field: tuple(uses) for field, uses in result.items()}


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


def source_uses(module, replacements, results, *, scopes=None, coordinate=None):
    """Give distinct typed source versions their own ordinary FX inputs.

    Only source edges are copied. No numerical operation is evaluated or
    captured again. The returned port map retains original result identities.
    """
    import operator
    import torch
    from torch.fx.node import map_arg
    scopes = {} if scopes is None else scopes
    graph, bound, versions, ports, local_uses = torch.fx.Graph(), {}, {}, {}, {}
    declared_uses = []
    scope_cuts, scope_results = set(), set()
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

    def typed(value, owner):
        if 'stride' not in owner:
            raise ValueError('local source typing requires the declared per-rank stride for a gapped layout')
        with value.fake_mode:
            return torch.empty_strided(owner['shape'], owner['stride'], dtype=value.dtype, device=value.device,
                                      requires_grad=value.requires_grad)

    def input_use(target, value, declaration, selections=None):
        for field, owner, made in declared_uses:
            if field == target and owner == declaration:
                return made
        made = graph.placeholder(f'use_{len(declared_uses)}')
        made.meta['val'] = typed(value, declaration)
        ports[made, ()] = dict(field=target, selection=declaration['selection'])
        if selections is not None:
            ports[made, ()]['selections'] = selections
        declared_uses.append((target, declaration, made))
        return made

    for consumer, scope in scopes.items():
        for use in scope['uses']:
            source, path = use['source']
            target = replacements.get((consumer, use['side'], use['path']), {}).get(path, (source, path))
            owner = use['required']
            producer = scopes.get(source, {}).get('outputs', {}).get(path)
            if target == (source, path) and producer is not None and producer['owners'] == owner['owners']:
                # One local source scope remains one ordinary graph. No page
                # boundary is inserted between compatible producer/consumer
                # declarations merely because both have owner metadata.
                continue
            declaration = owner['owners'][coordinate]
            made = input_use(target, pytree.key_get(source.meta['val'], path), declaration,
                {rank: value['selection'] for rank, value in owner['owners'].items()})
            local_uses[consumer, use['side'], use['path'], path] = made
            scope_cuts.add((consumer, 'before'))
            if source.op != 'placeholder' and target == (source, path):
                scope_cuts.add((source, 'after'))
                scope_results.add(source)
    for consumer in module.graph.nodes:
        if consumer in scopes or consumer.op in ('placeholder', 'output'):
            continue
        for side in ('args', 'kwargs'):
            for argument, source in pytree.tree_flatten_with_path(getattr(consumer, side))[0]:
                if not isinstance(source, torch.fx.Node) or source not in scopes:
                    continue
                for path, value in _source_leaves((source,)):
                    if not isinstance(value, torch.Tensor):
                        continue
                    field_path = path[1]
                    target = replacements.get((consumer, side, tuple(argument)), {}).get(field_path, path)
                    made = input_use(target, value, dict(shape=tuple(value.shape), stride=tuple(value.stride()),
                        selection=dict(shape=list(value.shape), steps=[])))
                    local_uses[consumer, side, tuple(argument), field_path] = made
                    scope_cuts.update(((source, 'after'), (consumer, 'before')))
                    scope_results.add(source)

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
            local = {field: made for (consumer, where, argument, field), made in local_uses.items()
                     if consumer is node and where == side and argument == tuple(path)}
            if chosen is None and not local:
                return bound[value] if isinstance(value, torch.fx.Node) else value
            return pytree.tree_map_with_path(lambda field_path, described:
                local[tuple(field_path)] if tuple(field_path) in local else
                versions[chosen[tuple(field_path)]] if chosen and tuple(field_path) in chosen else field(value, field_path)
                if isinstance(described, torch.Tensor) else described, value.meta['val'])
        made = pytree.tree_map_with_path(replace, original)
        scope = scopes.get(node)
        if scope is not None and scope['sharding'] is not None and scope['sharding'].use_val_from_redistribute_schema:
            schema = scope['sharding'].redistribute_schema
            declared = schema.args_schema if side == 'args' else schema.kwargs_schema
            # The registered sharding rule already supplied local scalar/shape
            # operands. Preserve tensor edges while consuming those operands.
            def scalar(path, value):
                return pytree.key_get(made, path) if isinstance(value, DTensorSpec) else value
            made = pytree.tree_map_with_path(scalar, declared, is_leaf=lambda value: isinstance(value, DTensorSpec))
        return made

    for node in module.graph.nodes:
        if node.op == 'placeholder':
            continue
        if node.op == 'output':
            graph.output(map_arg(node.args[0], bound.__getitem__))
            continue
        scope = scopes.get(node)
        if scope is not None and scope.get('source') is not None:
            target = scope['source']
            owner = scope['outputs'][()]
            made = input_use(target, pytree.key_get(target[0].meta['val'], target[1]), owner['owners'][coordinate],
                             {rank: value['selection'] for rank, value in owner['owners'].items()})
            made = graph.call_function(torch.ops.aten.alias.default, (made,))
            made.meta = dict(node.meta, val=typed(node.meta['val'], owner['owners'][coordinate]))
            bound[node] = made
            continue
        schema = scope['sharding'].redistribute_schema if scope is not None and scope['sharding'] is not None else None
        target = schema.op if schema is not None and scope['sharding'].needs_redistribute else node.target
        made = graph.create_node(node.op, target, arguments(node, 'args'), arguments(node, 'kwargs'), node.name)
        made.meta = node.meta.copy()
        if scope is not None:
            made.meta['val'] = pytree.tree_map_with_path(lambda path, value:
                typed(value, scope['outputs'][tuple(path)]['owners'][coordinate])
                if isinstance(value, torch.Tensor) and tuple(path) in scope['outputs'] else value, node.meta['val'])
            made.meta['mesh_source_scope'] = coordinate
        bound[node] = made
    for node, made in bound.items():
        for key, value in _source_leaves((node,)):
            if isinstance(value, torch.Tensor):
                owner = scopes.get(node, {}).get('outputs', {}).get(key[1])
                ports[made, key[1]] = dict(field=results.get(key, key), selection=(
                    owner['owners'][coordinate]['selection'] if owner is not None else
                    dict(shape=list(value.shape), steps=[])))
    from .compilation import _source_results
    copied = torch.fx.GraphModule(module, graph)
    copied.meta = dict(module.meta, mesh_source_results=_source_results(module, bound))
    metadata = tuple(node.meta['val'] for node in graph.nodes if node.op == 'placeholder')
    return dict(graph=copied, operands=metadata,
                cuts=tuple((bound[node], side) for node in module.graph.nodes for side in ('before', 'after')
                           if (node, side) in scope_cuts),
                materializations=tuple(bound[node] for node in module.graph.nodes if node in scope_results)), ports, bound


def prepare_expression(expression):
    """Retain one numerical IR per section of an already captured expression.

    Collective and provider sections remain their original declarations. A
    section's source inputs/results keep their tensor-field identities while
    the common numerical compiler performs its ordinary source rewrites.
    """
    from .compilation import operands as compiler_operands, normalize_source
    configuration = dict(compiler_operands())
    if 'prepared' in expression and expression['prepared']['configuration'] == configuration:
        return expression['prepared']
    import torch
    from torch.fx.node import map_arg
    from mesh_relation import Algebra
    from . import host, prepare_source
    from .declarations import _bound, normalize_edges, sections
    from .native import prepare_graph
    from .algebra import decompositions
    original = expression['graph']
    aliases = expression.setdefault('aliases', {})
    source_nodes = tuple(dict.fromkeys((*original.graph.nodes, *aliases)))
    module = normalize_source(original, decompositions)
    aliases.update(normalize_edges(module))
    def current(node):
        while node in aliases:
            node = aliases[node]
        return node
    algebra = expression.get('algebra') or Algebra()
    arguments = dict(zip((node for node in module.graph.nodes if node.op == 'placeholder'),
                         expression['operands'], strict=True))
    order = {node: index for index, node in enumerate(module.graph.nodes)}
    fields = {}
    results = module.meta.get('mesh_source_results', {})
    for node in source_nodes:
        if 'val' not in node.meta:
            continue
        result = results.get(node, results.get(current(node), current(node)))
        for key, value in _source_leaves((node,)):
            if isinstance(value, torch.Tensor):
                fields[key] = ((current(result), key[1]) if isinstance(result, torch.fx.Node)
                               else (current(pytree.key_get(result, key[1])), ()))

    def generated(node):
        declared = tuple(fields[key][0] for key, value in _source_leaves((node,))
                         if isinstance(value, torch.Tensor)) if 'val' in node.meta else (current(node),)
        introduced = module.meta.get('mesh_source_operations', {}).get(node, declared) if module is not original else declared
        return tuple(dict.fromkeys(current(value) for value in introduced if current(value) in order))

    cuts = {at for node, index in order.items() if _bound(node) or host.bound(node) for at in (index, index + 1)}
    materializations = tuple(dict.fromkeys(fields[key][0]
        for node in expression.get('materializations', ()) for key, value in _source_leaves((node,))
        if isinstance(value, torch.Tensor)))
    cuts.update(order[node] + 1 for node in materializations)
    cuts.update((min(order[value] for value in generated(node)) if side == 'before'
                 else max(order[value] for value in generated(node)) + 1)
                for node, side in expression.get('cuts', ()))
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
                return {key: (declared_binding(retained.value_binding(value, algebra)),)
                        for (key, described), value in zip(_source_leaves(original), values, strict=True)
                        if isinstance(described, torch.Tensor)}
            span.update(lowered=lowered, metadata=metadata, operands=operands, retained=retained,
                        input_bindings=bindings(inputs, input_values), output_bindings=bindings(outputs, output_values),
                        source_values={node: map_arg(tree, retained.source_values.__getitem__)
                                       for node, tree in lowered.meta.get('mesh_source_results', {}).items()},
                        effects=retained.effects(algebra))
        spans.append(span)
    normalized = dict(expression, graph=module)
    normalized.pop('prepared', None)
    prepared = dict(expression=normalized, algebra=algebra, aliases=aliases, fields=fields, sections=tuple(spans),
                    configuration=configuration)
    expression['prepared'] = prepared
    return prepared


def source_ports(prepared, ports):
    """Carry exact source fields through the common source normalization."""
    aliases, fields, result = prepared['aliases'], prepared['fields'], {}
    for key, port in ports.items():
        node, path = fields.get(key, key)
        while node in aliases:
            node = aliases[node]
        result[node, path] = port
    return result


def cut_sections(parent, split, algebra):
    """Normalize retained read cuts to the ordinary numerical section contract.

    The compiler's existing port fields identify carried values. Original
    named storage keeps its parent semantic fields and declared relations.
    A cut's contribution and received value remain distinct:
    only the caller's declared delivery can make those two ports equal.
    """
    if not split['boundaries']:
        return dict(sections=(parent,), boundaries=())

    consumers = {boundary['input']['binding']['name']: set() for boundary in split['boundaries']}
    for section in split['sections']:
        for operation, node in section.nodes.items():
            for access in node.read_writes.reads:
                if access.name in consumers:
                    consumers[access.name].add(operation)
    received = {name: next(iter(operations)) for name, operations in consumers.items() if len(operations) == 1}
    spans = []
    for section in split['sections']:
        inputs, outputs = tuple(section.input_ports.values()), tuple(section.output_ports)
        def bindings(ports, side):
            names = {port['binding']['name'] for port in ports}
            retained = {field: tuple(binding for binding in uses if binding['name'] in names)
                        for field, uses in parent[side].items()}
            retained = {field: uses for field, uses in retained.items() if uses}
            known = {binding['name'] for _, binding in binding_uses(retained)}
            for port in ports:
                name = port['binding']['name']
                if name in known:
                    continue
                matches = [(field, binding) for other in ('input_bindings', 'output_bindings')
                           for field, binding in binding_uses(parent[other]) if binding['name'] == name]
                for field, binding in matches or [(port['field'], declared_binding(port['binding']))]:
                    retained[field] = (*retained.get(field, ()), binding)
                known.add(name)
            return port_bindings(retained, {}, algebra)
        input_bindings, output_bindings = bindings(inputs, 'input_bindings'), bindings(outputs, 'output_bindings')
        spans.append(dict(parent,
            source_section=parent.get('source_section', parent),
            inputs=tuple(dict.fromkeys(field[0] for field in input_bindings)),
            outputs=tuple(dict.fromkeys(field[0] for field in output_bindings)),
            operands=tuple(pytree.key_get(port['field'][0].meta['val'], port['field'][1]) for port in inputs),
            boundaries=(), retained=section, effects=section.effects(algebra),
            input_bindings=input_bindings, output_bindings=output_bindings,
            received_values={name: operation for name, operation in received.items() if operation in section.nodes}))
    return dict(sections=tuple(spans), boundaries=tuple(split['boundaries']))


def page_equalities(section, fixed=()):
    """Safe co-placement of a fresh received value's sole identity consumer.

    Equal values at one operation do not establish storage identity across
    later writes. Fresh receivers have no other observer when dependencies
    name one consuming operation. Its ordinary unfixed result may share those
    pages; caller-fixed or mutation-backed targets retain the normal copy.
    """
    received = {} if section is None else section.get('received_values', {})
    if not received:
        return ()
    retained = section['retained']
    aliases = retained.page_aliases()
    mutations = retained.scheduler_state['mutation_real_name']
    fixed_names = {mutations.get(record['binding']['name'], record['binding']['name']) for record in fixed}
    result = []
    for alias in aliases:
        source, target = (use['binding']['name'] for use in alias['pair'])
        if (received.get(source) == alias['operation'].get_operation_name()
                and mutations.get(target, target) == target and target not in fixed_names):
            result.append(alias['pair'])
    return tuple(result)


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
    normalized = source_ports(prepared, ports)
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
    from .operand import access_domain, byte_relation
    algebra, limits, pending = prepared['algebra'], {}, []
    uses = {} if uses is None else uses
    for section in prepared['sections']:
        section = section.get('rank_sections', {}).get(coordinate, section)
        if section['boundaries']:
            continue
        axes, available = {}, {}
        for key, binding in binding_uses(section['output_bindings']):
            owner = values.get(key)
            if owner is not None:
                axes.setdefault(binding['name'], set()).update(axis for axis, _ in owner.get('reductions', ()))
        members = section['graph'].meta.get('mesh_source_results', {})
        for key, binding in binding_uses(section['input_bindings']):
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
                        possible = access_domain(access, work)['positions']
                        if possible is None or not possible.binary(owned, 'subtract').empty():
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
    input_names = {binding['name'] for _, binding in binding_uses(span['input_bindings'])}
    written_inputs = {}
    for key, binding in binding_uses(span['output_bindings']):
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
                    if positions is not None and access['name'] in input_names:
                        # Supplied input pages also bind explicit mutations.
                        # A write-only input is not a read/transfer demand, but
                        # its exact written bytes must reach fixed placement.
                        _union(written_inputs, access['name'], positions)
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
                         for dependency, _ in effect['scheduling_writes']
                         if dependency.name in needed and not needed[dependency.name].empty()}
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
    inputs, live_inputs = {}, {}
    for key, binding in binding_uses(span['input_bindings']):
        if binding['name'] in exact:
            _union(inputs, key, byte_relation(binding).preimage(exact[binding['name']]))
        live_inputs[key] = (*live_inputs.get(key, ()),
            *(record for record in live_reads if record['access']['name'] == binding['name']))
    live_inputs = {key: records for key, records in live_inputs.items() if records}
    storage = dict(needed)
    for name, positions in written_inputs.items():
        _union(storage, name, positions)
    return dict(section=span, outputs=dict(demands), inputs=inputs, storage_bytes=storage,
                records=tuple(records), live_reads=tuple(live_reads), live_inputs=live_inputs, pending=tuple(pending))


def assigned_read_sections(section, records, produced, algebra, *, delivery, transition):
    """Splice cached owner reads into supplied numerical work.

    Possible read positions select providers and resident pages only. The cut's
    returned values, ordered by the original work coordinates, are the payload
    of the ordinary placement expression supplied by the frontend adapter.
    """
    groups = {}
    for rank, record in records.items():
        member = record['section']
        groups.setdefault((member['retained'], id(member['input_bindings'])), {})[rank] = record
    sections, boundaries, programs, order, pending = {rank: [] for rank in records}, {}, {}, [], []
    changed = False
    for declaration, recipients in groups.items():
        split = _assigned_read_group(section, recipients, produced, algebra,
                                     delivery=delivery, transition=transition, declaration=declaration)
        if split is None:
            tail = (declaration, None)
            order.append(tail)
            for rank, record in recipients.items():
                sections[rank].append(dict(record, read_cut=tail))
            continue
        changed = True
        pending.extend(split['pending'])
        if split['pending']:
            continue
        for rank, spans in split['sections'].items():
            sections[rank].extend(spans)
        for rank, values in split['boundaries'].items():
            boundaries.setdefault(rank, {}).update(values)
        programs.update(split['programs'])
        order.extend(split['order'])
    if not changed:
        return None
    return dict(sections=sections, boundaries=boundaries, programs=programs,
                order=tuple(order), pending=tuple(pending))


def _assigned_read_group(section, records, produced, algebra, *, delivery, transition, declaration):
    """One exact retained declaration, evaluated on its declared providers.

    Independently captured local scopes do not assert equivalent scalar uses.
    Each keeps its own coordinates and ports. Providers prepare only the read
    prefixes; original assigned work remains on the declaration's recipients.
    """
    import math
    import torch
    from mesh_relation import flat
    from .operand import byte_relation
    retained = declaration[0]
    sites, pending = {}, []
    for rank, record in records.items():
        member = record['section']
        for source, reads in record.get('live_inputs', {}).items():
            for read in reads:
                binding = binding_for_access(member['input_bindings'], source, read['access'])
                owned = byte_relation(binding).image(produced[rank].get(source, _empty(algebra, 1)))
                if read['positions'].binary(owned, 'subtract').empty():
                    continue
                for projected in record['records']:
                    for slot, value in projected['reads'].items():
                        if value['access'] is read['access']:
                            key = (declaration, source, id(projected['domain']), slot)
                            sites.setdefault(key, dict(source=source, effect=projected['effect'],
                                domain=projected['domain'], access=read['access'], slot=slot))
    if not sites:
        return None
    requests = {rank: [] for rank in produced}
    member = next(iter(records.values()))['section']
    fields = torch.fx.Graph()
    for key, site in sites.items():
        source = site['source']
        needed = {rank: _empty(algebra, 1) for rank in produced}
        work = {rank: _empty(algebra, len(site['domain']['work'])) for rank in produced}
        for rank, record in records.items():
            matching = next(row for row in record['records'] if row['domain'] is site['domain'])
            work[rank] = matching['work']
            read = matching['reads'][site['slot']]
            binding = binding_for_access(record['section']['input_bindings'], source, read['access'])
            needed[rank] = byte_relation(binding).preimage(read['positions'])
        providers = delivery(section, source, needed, algebra)
        pending.extend(providers['pending'])
        provided = providers['provided']
        if provided is None:
            pending.append(dict(source=source, requirement='live_read_owners'))
            continue
        dimensions = tuple(provided['dimensions'])
        fixed = tuple(axis for axis in range(provided['mesh'].ndim) if axis not in dimensions)
        owners = {rank: algebra.selection(selection) for rank, selection in providers['sources'].items()}
        for rank, selected in owners.items():
            if not selected.binary(produced[rank].get(source, _empty(algebra, 1)), 'subtract').empty():
                pending.append(dict(source=source, coordinate=rank, requirement='assigned_source_pages'))
            peers = tuple(peer for peer in produced if all(peer[axis] == rank[axis] for axis in fixed))
            if any(not selected.binary(owners[peer], 'intersect').empty() for peer in peers if peer < rank):
                pending.append(dict(source=source, coordinate=rank, requirement='delivery_sources'))
            available = _empty(algebra, 1)
            selected_work = _empty(algebra, len(site['domain']['work']))
            for peer in peers:
                available = available.binary(owners[peer], 'union')
                selected_work = selected_work.binary(work[peer], 'union')
            if not needed[rank].binary(available, 'subtract').empty():
                pending.append(dict(source=source, coordinate=rank, requirement='live_read_owners'))
            binding = binding_for_access(member['input_bindings'], source, site['access'])
            requests[rank].append(dict(site, key=key, work=selected_work,
                owner=byte_relation(binding).image(selected)))
        value = pytree.key_get(source[0].meta['val'], source[1])
        site['fields'] = tuple(fields.placeholder(f'read_{len(fields.nodes)}') for _ in range(2))
        from .compilation import source_records
        origins = tuple(site['effect']['operation'].get_origins())
        provenance = []
        for original in (*origins, source[0]):
            for record in source_records(original):
                if record not in provenance:
                    provenance.append(record)
        for node in site['fields']:
            node.meta.update(mesh_source_nodes=origins, mesh_source_records=tuple(provenance),
                             mesh_route_scope=source[0].meta.get('mesh_route_scope'))
            with value.fake_mode:
                node.meta['val'] = torch.empty(site['domain']['work'], dtype=site['access']['dtype'], device=value.device)
        site.update(providers=providers, work=work, dimensions=dimensions)
    if pending:
        return dict(pending=tuple(pending))
    split, boundaries = {}, {}
    for rank, requested in requests.items():
        cut = cut_sections(member, retained.cut_reads(requested), algebra)
        ports = {}
        boundaries[rank] = {boundary['key']: boundary for boundary in cut['boundaries']}
        for boundary in cut['boundaries']:
            site = sites[boundary['key']]
            shape = site['domain']['work']
            flatten = algebra.map(shape, (math.prod(shape),), lambda x, y: [f'{y[0]} = {flat(shape, x)}'])
            order = boundary['ordinals'].then(flatten)
            selecting = dict(shape=list(map(int, shape)),
                             steps=[['reindex', dict(shape=list(map(int, order.shape)), index=order.terms())]])
            for side, field in zip(('output', 'input'), site['fields'], strict=True):
                ports[boundary[side]['field']] = dict(field=(field, ()), selection=selecting)
        split[rank] = tuple(bind_ports(span, ports, algebra) for span in cut['sections'])
    programs = []
    for key, site in sites.items():
        ports = {rank: dict(boundary=boundaries[rank].get(key), source=(site['fields'][0], ()),
                           result=(site['fields'][1], ())) for rank in produced}
        program = transition(site, ports, algebra)
        pending.extend(program.get('pending', ()))
        programs.append(program)
    if pending:
        return dict(pending=tuple(pending))
    projected = {rank: [] for rank in produced}
    tail = (declaration, None)
    for rank, spans in split.items():
        original = records.get(rank)
        old = {} if original is None else {id(row['effect']['operation']): row for row in original['records']}
        prefixes = {}
        for key, boundary in boundaries[rank].items():
            shape = sites[key]['domain']['work']
            flatten = algebra.map(shape, (math.prod(shape),), lambda x, y: [f'{y[0]} = {flat(shape, x)}'])
            positions = flatten.image(boundary['work'])
            prefixes[sites[key]['fields'][0], ()] = dict(shape=list(map(int, shape)),
                steps=[['reshape', [int(math.prod(shape))]], ['select', 0, positions.spans()]])
        prefix_work = project_expression(dict(algebra=algebra, aliases={}, sections=spans), prefixes)
        for at, span in enumerate(spans):
            prefix = prefix_work['sections'][at]
            if original is None and not any(not row['work'].empty() or
                    row.get('finish') is not None and not row['finish']['work'].empty()
                    for row in prefix['records']):
                continue
            required = {id(row['domain']): row for row in prefix['records']}
            assigned = {}
            for effect in span['effects']:
                prior = old.get(id(effect['operation']))
                for domain in effect['domains']:
                    selected = required[id(domain)]
                    assigned[id(domain)] = selected['work'] if prior is None else selected['work'].binary(prior['work'], 'union')
                    if domain.get('finish') is not None:
                        finish = selected['finish']['work']
                        assigned[id(domain['finish'])] = finish if prior is None else finish.binary(prior['finish']['work'], 'union')
            demanded = dict(prefix['outputs'])
            if original is not None:
                for key, value in original['outputs'].items():
                    if key in span['output_bindings']:
                        _union(demanded, key, value)
            record = project_numerical(span, demanded, algebra, work=assigned)
            record['read_cut'] = next((key for key, boundary in boundaries[rank].items()
                                       if boundary['before'] == at), tail)
            # Publish the stores of that supplied work, including carried
            # intermediate values and the cut's ordinary contribution port.
            for field, binding in binding_uses(span['output_bindings']):
                positions = _empty(algebra, 1)
                for row in record['records']:
                    for domain in (row, row.get('finish')):
                        if domain is not None:
                            for write in domain['writes'].values():
                                if write['access']['name'] == binding['name'] and write['positions'] is not None:
                                    positions = positions.binary(write['positions'], 'union')
                _union(record['outputs'], field, byte_relation(binding).preimage(positions))
            projected[rank].append(record)
    ordered = sorted(sites, key=lambda key: next(values[key]['source_order']
                     for values in boundaries.values() if key in values))
    return dict(sections=projected, boundaries=boundaries, programs=dict(zip(sites, programs, strict=True)),
                order=(*ordered, tail), pending=())


def compile_assignments(prepared, assignments, outputs, *, delivery, boundary_projection, constraints=None, transitions=(),
                        read_transition=None):
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
            record.setdefault('section', section)
            sections[rank].append(record)
            pending[rank].extend(record['pending'])
            for key, positions in record['outputs'].items():
                _union(produced[rank], identity(key), positions)

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
                for key, binding in binding_uses(record['section'].get('output_bindings', {})):
                    if key not in program['state_outputs']:
                        continue
                    positions = _empty(algebra, 1)
                    for effect in record['records']:
                        for domain in (effect, effect.get('finish')):
                            if domain is not None:
                                for written in domain['writes'].values():
                                    if written['access']['name'] == binding['name'] and written['positions'] is not None:
                                        positions = positions.binary(written['positions'], 'union')
                    from .operand import byte_relation
                    _union(record['outputs'], key, byte_relation(binding).preimage(positions))
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
                   {rank: project_numerical(section.get('rank_sections', {}).get(rank, section), values, algebra,
                                            constraints=constraints.get(rank))
                    for rank, values in needs.items()})
        sources = section['graph'].meta.get('mesh_source_results', {})
        for program in transitions:
            transition = program['transition']
            if program['before'] and transition['consumer'] in sources:
                conversion(program, {rank: {program['result']: record['inputs'][program['result']]}
                    if program['result'] in record['inputs'] else {} for rank, record in records.items()})
        split = (assigned_read_sections(section, records, produced, algebra, delivery=delivery,
                                        transition=read_transition)
                 if read_transition is not None and not section['boundaries'] else None)
        if split is not None and not split['pending']:
            for event in split['order']:
                stage = {rank: next((record for record in spans if record['read_cut'] == event), None)
                         for rank, spans in split['sections'].items()}
                stage = {rank: record for rank, record in stage.items() if record is not None}
                if not stage:
                    continue
                wrapper = dict(next(iter(stage.values()))['section'])
                for rank, record in stage.items():
                    record['section'] = dict(record['section'], source_section=wrapper)
                append(wrapper, stage)
                if event in split['programs']:
                    program = split['programs'][event]
                    needs = {rank: {} for rank in assignments}
                    for rank, spans in split['sections'].items():
                        for record in spans:
                            if program['result'] in record['inputs']:
                                _union(needs[rank], program['result'], record['inputs'][program['result']])
                    if any(needs.values()):
                        conversion(program, needs)
            for program in transitions:
                if not program['before'] and program['transition']['consumer'] in sources:
                    conversion(program, {rank: {program['result']: assigned[rank][program['result']]}
                        for rank in assignments})
            continue
        if split is not None:
            for rank in pending:
                pending[rank].extend(split['pending'])
        from .operand import byte_relation
        for rank, record in records.items():
            for key, reads in record.get('live_inputs', {}).items():
                source = identity(key)
                for read in reads:
                    binding = binding_for_access(record['section']['input_bindings'], key, read['access'])
                    available = byte_relation(binding).image(produced[rank].get(source, _empty(algebra, 1)))
                    if not read['positions'].binary(available, 'subtract').empty():
                        pending[rank].append(dict(source=source, section=section, access=read,
                                                  requirement='selected_source_reads'))
        for rank, record in records.items():
            record.setdefault('section', section)
            record['outputs'] = needs[rank]
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
        members = {rank: section.get('rank_sections', {}).get(rank, section) for rank in requested}
        needs = {rank: {key: values[key] for key in section_fields(members[rank], 'outputs') if key in values}
                 for rank, values in requested.items()}
        if section['boundaries']:
            records = boundary_projection(section, needs, algebra)
        else:
            records = {rank: project_numerical(members[rank], values, algebra, constraints=constraints.get(rank))
                       for rank, values in needs.items()}
        for rank, record in records.items():
            record.update(section=members[rank], outputs=needs[rank])
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
