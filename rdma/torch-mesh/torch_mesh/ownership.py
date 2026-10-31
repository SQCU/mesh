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
    algebra = Algebra()
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


def project_numerical(span, demands, algebra, *, constraints=None):
    """Write preimages and read images over the same retained numerical body.

    Demands name original tensor fields in logical scalar positions. Aliases
    meet in explicitly sized storage-byte domains. Unknown index/predicate
    relations remain actual source expressions; no full-storage dependency is
    substituted for them. A selected reduction finish survives an empty fiber.
    """
    from .operand import byte_relation
    constraints = {} if constraints is None else constraints
    needed, pending, records = {}, [], []
    for key, binding in span['output_bindings'].items():
        if key in demands:
            _union(needed, binding['name'], byte_relation(binding).image(demands[key]))

    def selected(domain):
        work = _empty(algebra, len(domain['work']))
        unresolved = []
        for slot, access in domain['writes'].items():
            requested = needed.get(access['name'])
            if requested is None or requested.empty():
                continue
            relation = byte_relation(access)
            if relation is None or access.get('unresolved_predicates'):
                unresolved.append(dict(slot=slot, access=access, requested=requested))
            else:
                work = work.binary(relation.preimage(requested), 'union')
        if id(domain) in constraints:
            work = work.binary(constraints[id(domain)], 'intersect')
        reads, writes = {}, {}
        for side, result in (('reads', reads), ('writes', writes)):
            for slot, access in domain[side].items():
                relation = byte_relation(access)
                positions = (relation.image(work) if relation is not None
                             and not access.get('unresolved_predicates') else None)
                result[slot] = dict(access=access, relation=relation, positions=positions)
                if side == 'writes':
                    result[slot]['requested'] = needed.get(access['name'], _empty(algebra, 1))
                if side == 'reads' and not work.empty():
                    if positions is None:
                        pending.append(dict(domain=domain, access=access, work=work))
                    else:
                        _union(needed, access['name'], positions)
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
    inputs = {key: byte_relation(binding).preimage(needed[binding['name']])
              for key, binding in span['input_bindings'].items() if binding['name'] in needed}
    return dict(section=span, outputs=dict(demands), inputs=inputs, storage_bytes=needed,
                records=tuple(records), pending=tuple(pending))


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
        needs = {rank: {key: values[key] for key, _ in _source_leaves(section['outputs']) if key in values}
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
    inputs = tuple(_source_leaves(node for node in prepared['expression']['graph'].graph.nodes
                                  if node.op == 'placeholder'))
    return dict(projections={rank: dict(prepared=prepared, outputs=ordered[rank],
        output_selections=selections[rank], inputs={key: values[key] for key, _ in inputs if key in values},
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
