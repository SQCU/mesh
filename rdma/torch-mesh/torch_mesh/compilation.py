"""Compiler operands scoped to one graph realization, never runtime policy."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from . import _stream

_bound = ContextVar('mesh_compilation_operands', default=None)
_layouts = ContextVar('mesh_compilation_layouts', default=None)


def operands():
    held = _bound.get()
    return _stream.compilation if held is None else held


@contextmanager
def bind(configuration):
    values = dict(operands())
    preferences = ()
    if configuration is not None:
        for key, value in configuration.items():
            if key == 'layout_preferences':
                preferences = value
                continue
            if key not in values:
                raise ValueError(f'unknown compiler operand {key}')
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key}: compiler operand must be a positive integer')
            values[key] = value
    token = _bound.set(values)
    layouts = _layouts.set(dict(preferences=preferences, choices=[]))
    try:
        yield values
    finally:
        _layouts.reset(layouts)
        _bound.reset(token)


def layout_preferences(family, supplied=None):
    """Match complete semantic family metadata, never a vector length or hash."""
    import json
    import math
    family = json.loads(json.dumps(family))
    held = _layouts.get()
    supplied = supplied if supplied is not None else (() if held is None else held['preferences'])
    if family['scope'] is None:
        if supplied:
            raise ValueError('layout preferences require a declared source_scope')
        return [1.0] * len(family['uses'])
    matches = [record for record in supplied if json.loads(json.dumps(record['family'])) == family]
    if len(matches) > 1:
        raise ValueError('layout family has multiple preference declarations')
    weights = list(matches[0]['weights']) if matches else [1.0] * len(family['uses'])
    if len(weights) != len(family['uses']) or any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError('layout preferences must match the declared uses and be finite and nonnegative')
    return weights


def layout_choice(record):
    """Retain the actual prepared direct/conversion choice for whole-call feedback."""
    import json
    held = _layouts.get()
    if held is not None and record['family']['scope'] is not None:
        held['choices'].append(json.loads(json.dumps(record)))


def layout_choices():
    held = _layouts.get()
    return tuple(held['choices']) if held is not None else ()


def parameters():
    """The declared empirical compiler domains, read at preparation."""
    import json
    from pathlib import Path
    import mesh_compute
    path = Path(mesh_compute.__file__).resolve().parents[1] / 'etc/compute/parameters.json'
    return json.loads(path.read_text())


@lru_cache(None)
def _api_configuration():
    """Physical operand requirements of a declared native API, at preparation."""
    import json
    from pathlib import Path
    import mesh_compute
    path = Path(mesh_compute.__file__).resolve().parents[1] / 'etc/compute/metal-apis.json'
    return json.loads(path.read_text())


def api_layouts(name):
    return _api_configuration()[name]


def lift_state(module, arguments):
    """Turn captured tensor identities into ordinary source graph operands."""
    import torch
    if not any(node.op == 'get_attr' for node in module.graph.nodes):
        return module, list(arguments)
    graph, copied, aliases = torch.fx.Graph(), {}, {}
    graph.output(graph.graph_copy(module.graph, copied))
    lifted = torch.fx.GraphModule(module, graph)
    lifted.meta = module.meta.copy()
    values = []
    bindings = {id(value): node for node, value in zip(
        (node for node in graph.nodes if node.op == 'placeholder'),
        module.graph.process_inputs(*arguments), strict=True) if isinstance(value, torch.Tensor)}
    first = next(node for node in graph.nodes if node.op != 'placeholder')
    for node in list(graph.nodes):
        if node.op != 'get_attr':
            continue
        value = lifted
        for name in node.target.split('.'):
            value = getattr(value, name)
        if not isinstance(value, torch.Tensor):
            continue
        operand = bindings.get(id(value))
        if operand is None:
            with graph.inserting_before(first):
                operand = graph.placeholder(f'_mesh_attribute_{len(values)}')
            operand.meta = node.meta.copy()
            operand.meta.setdefault('mesh_source', node.name)
            bindings[id(value)] = operand
            values.append(value)
        node.replace_all_uses_with(operand)
        aliases[node] = operand
        if node is first:
            first = node.next
        graph.erase_node(node)
    lifted.recompile()
    lifted.meta['mesh_source_results'] = _source_results(module, {
        source: aliases.get(value, value) for source, value in copied.items()})
    if 'mesh_tensor_producers' in module.meta:
        lifted.meta['mesh_tensor_producers'] = tuple(
            (value, aliases.get(copied[node], copied[node]))
            for value, node in module.meta['mesh_tensor_producers'])
    return lifted, [*arguments, *values]


def lift_attributes(module, arguments):
    """Bind source state to page operands before splitting numerical spans."""
    import torch
    lifted, operands = lift_state(module, arguments)
    from torch.distributed.tensor import DTensor
    values = operands[len(arguments):]
    sources = [value._local_tensor if isinstance(value, DTensor) else value for value in values]
    imported, copies = _stream.import_attributes(sources)
    # Foreign GPU storage remains an explicit initializer operand until its
    # prepared copy completes. CPU initializers have already been copied during
    # preparation. Every numerical use names the function's imported pages;
    # the original attribute remains its source for other function declarations.
    lifted.meta['mesh_initializations'] = (*lifted.meta.get('mesh_initializations', ()), *copies)
    bound = []
    with torch.no_grad():
        for value, source, made in zip(values, sources, imported, strict=True):
            if made is source:
                bound.append(value)
                continue
            made.requires_grad_(source.requires_grad)
            if isinstance(value, DTensor):
                from .partition import _wrap
                made = _wrap(made, value._spec, value.requires_grad)
            bound.append(made)
    return lifted, [*operands[:len(arguments)], *bound]


def source_expression(function, *arguments):
    """Capture a value declaration and its exact tensor-state producer edges.

    Existing placement methods may retain a tensor for a later declaration.
    Capture them before functionalization so that retained value is the same
    ordinary FakeTensor lifted by its consumer. The common source lowerer
    functionalizes the joined expression without value-input writeback.
    """
    import torch
    from torch.fx.experimental.proxy_tensor import get_proxy_mode, make_fx
    producers = []
    def declaration(*values):
        result = function(*values)
        producers.extend((value, proxy.proxy.node)
                         for value, proxy in get_proxy_mode().tracer.tensor_tracker.items())
        return result
    with torch.no_grad():
        module = make_fx(declaration, decomposition_table={}, tracing_mode='fake',
                         _allow_non_fake_inputs=True)(*arguments)
    module.meta.update(mesh_tensor_producers=tuple(producers), mesh_input_writeback=False)
    return lift_state(module, arguments)


def compose_sources(expressions):
    """Join captured value declarations by their existing tensor identities.

    The caller supplies source order. FX copies each already captured body;
    a state tensor lifted by a later declaration binds directly to the earlier
    node that made that same tensor. No placement method is called again.
    """
    import torch
    graph, root = torch.fx.Graph(), torch.nn.Module()
    operands, identities, results, sources, mappings = [], {}, [], {}, []
    for ordinal, expression in enumerate(expressions):
        module, bound = expression['graph'], {}
        values = module.graph.process_inputs(*expression['operands'])
        for node, value in zip((node for node in module.graph.nodes if node.op == 'placeholder'), values, strict=True):
            held = identities.get(id(value)) if isinstance(value, torch.Tensor) else None
            if held is None:
                first = next((made for made in graph.nodes if made.op != 'placeholder'), graph._root)
                with graph.inserting_before(first):
                    made = graph.placeholder(f'source_{ordinal}_{node.name}')
                made.meta = node.meta.copy()
                operands.append(value)
                if isinstance(value, torch.Tensor):
                    identities[id(value)] = value, made
            else:
                _, made = held
            bound[node] = made
        for node in module.graph.nodes:
            if node.op != 'get_attr':
                continue
            value = module
            for field in node.target.split('.'):
                value = getattr(value, field)
            name = f'source_{ordinal}_{node.name}'
            setattr(root, name, value)
            made = graph.get_attr(name)
            made.meta = node.meta.copy()
            bound[node] = made
        result = graph.graph_copy(module.graph, bound)
        bound[next(node for node in module.graph.nodes if node.op == 'output')] = result
        results.append(result)
        mappings.append(bound)
        sources.update(_source_results(module, bound))
        for value, node in module.meta.get('mesh_tensor_producers', ()):
            identities[id(value)] = value, bound[node]
    graph.output(tuple(results))
    module = torch.fx.GraphModule(root, graph)
    module.meta.update(mesh_source_results=sources, mesh_tensor_producers=tuple(identities.values()),
                       mesh_input_writeback=False)
    return dict(graph=module, operands=tuple(operands), result=tuple(results), source_maps=tuple(mappings))


def inline_source(module, operands):
    """Splice an already declared expression into the current source graph."""
    import torch
    from torch.fx.experimental.proxy_tensor import get_proxy_mode, track_tensor_tree
    from torch.utils import _pytree as pytree
    mode = get_proxy_mode()
    tracer = mode.tracer
    values = module.graph.process_inputs(*operands)
    bindings = {node: tracer.create_arg(pytree.tree_map(tracer.unwrap_proxy, value)) for node, value in zip(
        (node for node in module.graph.nodes if node.op == 'placeholder'), values, strict=True)}
    result = tracer.graph.graph_copy(module.graph, bindings)
    metadata = pytree.tree_map(lambda value: value.meta['val'] if isinstance(value, torch.fx.Node) else value, result)
    proxies = pytree.tree_map(lambda value: torch.fx.Proxy(value, tracer) if isinstance(value, torch.fx.Node) else value, result)
    track_tensor_tree(metadata, proxies, constant=None, tracer=tracer)
    return metadata


def _source_origins(node, graph, before):
    """Annotate nodes created by one source equation, without name matching."""
    origins, created = node.meta.get('mesh_source_nodes', (node,)), []
    current = before.next
    while current is not graph._root:
        current.meta.setdefault('mesh_source_nodes', origins)
        for field in ('mesh_source_records', 'mesh_route_scope'):
            if field in node.meta:
                current.meta.setdefault(field, node.meta[field])
        created.append(current)
        current = current.next
    return origins, created


def source_records(node):
    """Original captured equations and occurrence domains, before physical grouping."""
    if 'mesh_source_records' in node.meta:
        return node.meta['mesh_source_records']
    return tuple(record for source in node.meta.get('mesh_source_nodes', ())
                 for record in source.meta.get('mesh_source_records', ()))


def source_scope(module, scope, route_scope=None):
    """Declare a caller namespace on the captured source, never on lowered ordinals."""
    from .operand import layout_configuration
    occurrence = layout_configuration(_stream.TensorLayout((1,), (1,)))
    for node in module.graph.nodes:
        if node.op == 'output':
            continue
        definition = dict(scope=scope, statement=node.name, definition=node.format_node())
        node.meta['mesh_source_records'] = (dict(source=definition, occurrences=occurrence, occurrence_extent=1),)
        node.meta['mesh_route_scope'] = definition if route_scope is None else route_scope
    module.meta.update(mesh_program_scope=scope, mesh_route_scope=route_scope, mesh_occurrence_count=1)
    return module


def repeat_source(module, count, first=0):
    """Compose one captured imperative graph before functionalization and fusion.

    Input and captured-state identities remain shared. Only the final return is
    published; every equation retains its original source and occurrence order.
    """
    import torch
    if count < 1 or first < 0:
        raise ValueError('source repetition requires a positive count and nonnegative first occurrence')
    graph, inputs, result, operations = torch.fx.Graph(), {}, None, {}
    extent = module.meta.get('mesh_occurrence_count', 1)
    for node in module.graph.nodes:
        if node.op == 'placeholder':
            inputs[node] = graph.node_copy(node, inputs.__getitem__)
            inputs[node].meta = node.meta.copy()
    for at in range(count):
        bound = dict(inputs)
        for node in module.graph.nodes:
            if node.op == 'placeholder':
                continue
            if node.op == 'output':
                result = torch.fx.node.map_arg(node.args[0], bound.__getitem__)
                continue
            made = graph.node_copy(node, bound.__getitem__)
            made.meta = node.meta.copy()
            made.meta.pop('mesh_collective_declaration', None)
            made.meta['mesh_source_nodes'] = node.meta.get('mesh_source_nodes', (node,))
            records = []
            for record in source_records(node):
                declared = record['occurrences']
                # Occurrences are produced here as affine ordered intervals;
                # graph composition translates that same relation, not node IDs.
                records.append(dict(record, occurrence_extent=count * extent, occurrences=dict(declared,
                    offset=declared['offset'] + first + at * extent)))
            made.meta['mesh_source_records'] = tuple(records)
            bound[node] = made
            operations.setdefault(node, []).append(made)
    graph.output(result)
    repeated = torch.fx.GraphModule(module, graph)
    repeated.meta = dict(module.meta, mesh_occurrence_count=count * extent,
                         mesh_source_results=_source_results(module, bound),
                         mesh_source_operations={node: tuple(values) for node, values in operations.items()})
    return repeated


def _source_result(node, made, path, declared, value, origins):
    """Keep arbitrary result declarations on the corresponding result only."""
    import torch
    from torch.utils import _pytree as pytree
    provenance = made.meta.get('from_node')
    metadata = dict(node.meta, val=value)
    for field in ('mesh_ownership', 'mesh_local_source', 'mesh_gradient'):
        if field in metadata and not isinstance(declared, torch.Tensor):
            metadata[field] = pytree.key_get(metadata[field], path)
    made.meta.update(metadata)
    if provenance is not None:
        made.meta['from_node'] = provenance
    made.meta.update(mesh_source=node.meta.get('mesh_source', node.name),
                     mesh_source_nodes=origins, mesh_result_path=path)


def _source_results(module, results):
    """Compose existing result identities through this source transformation."""
    import torch
    from torch.utils import _pytree as pytree
    original = {node: pytree.tree_map(lambda value: results[value] if isinstance(value, torch.fx.Node) else value, tree)
                for node, tree in module.meta.get('mesh_source_results', {}).items()}
    return dict(results) | original


def functionalize_source(module, inputs, decompositions):
    """The ordinary functionalization trace with exact source/result metadata.

    The source interpreter supplies node identity while the existing framework
    performs functionalization and decomposition. Generated FakeTensor values
    retain their transformed storage aliases; source declarations are copied
    only onto the corresponding final values, never onto intermediate math.
    """
    import torch
    import torch.fx.traceback as traceback
    from torch._subclasses.functional_tensor import FunctionalTensor, dispatch_functionalize
    from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot, make_fx
    from torch.utils import _pytree as pytree
    results = {}

    class Source(torch.fx.Interpreter):
        def run_node(self, node):
            tracer = get_proxy_mode().tracer
            before = tracer.graph._root.prev
            with traceback.set_current_meta(node, 'mesh functional source'):
                value = super().run_node(node)
                origins, created = _source_origins(node, tracer.graph, before)
                if node.op == 'output':
                    return value
                declared = node.meta.get('val', value)

                def retain(path, item):
                    if not isinstance(item, torch.Tensor):
                        return item
                    actual = item.from_functional() if isinstance(item, FunctionalTensor) else item
                    made = get_proxy_slot(actual, tracer).proxy.node
                    if made not in created and node.op not in ('placeholder', 'get_attr'):
                        item = torch.ops.aten.alias.default(item)
                        actual = item.from_functional() if isinstance(item, FunctionalTensor) else item
                        made = get_proxy_slot(actual, tracer).proxy.node
                    _source_result(node, made, path, declared, made.meta.get('val', actual), origins)
                    return item

                value = pytree.tree_map_with_path(retain, value)
                def result(item):
                    if not isinstance(item, torch.Tensor):
                        return item
                    actual = item.from_functional() if isinstance(item, FunctionalTensor) else item
                    return get_proxy_slot(actual, tracer).proxy.node
                results[node] = pytree.tree_map(result, value)
                return value

    functional = dispatch_functionalize(Source(module).run,
        propagate_input_mutations=module.meta.get('mesh_input_writeback', True))
    output = next(node for node in module.graph.nodes if node.op == 'output')
    def declaration(*arguments):
        tracer = get_proxy_mode().tracer
        before = tracer.graph._root.prev
        with traceback.set_current_meta(output, 'mesh functional result'):
            value = functional(*arguments)
            # Deferred view replay and input writeback belong to the source's
            # result boundary. The framework supplies their exact graph edges.
            _source_origins(output, tracer.graph, before)
            results[output] = pytree.tree_map(lambda item: get_proxy_slot(item, tracer).proxy.node
                if isinstance(item, torch.Tensor) else item, value)
            return value
    with traceback.preserve_node_meta():
        lowered = make_fx(declaration, decomposition_table=decompositions, tracing_mode='fake',
                          _allow_non_fake_inputs=True)(*inputs)
    lowered.meta = dict(module.meta, mesh_source_results=_source_results(module, results))
    return lowered


def normalize_source(module, decompositions, *, preserve=()):
    """Apply shared tensor equations to captured types before ownership.

    The framework's decomposition interpreter consumes only source metadata.
    Every generated operation retains its source origin; result allocations and
    captured-state identities belong only to that source operation's results.
    Native IR construction and physical page binding happen afterward.
    """
    import torch
    import torch.fx.traceback as traceback
    from torch._guards import detect_fake_mode
    from torch.fx.experimental.proxy_tensor import DecompositionInterpreter, get_proxy_slot, track_tensor_tree
    from torch.utils import _pytree as pytree
    preserve = set(preserve)
    if not any(node.op == 'call_function' and node.target in decompositions and node not in preserve
               for node in module.graph.nodes):
        return module
    graph, results, operations = torch.fx.Graph(), {}, {}

    class Normalize(DecompositionInterpreter):
        def get_attr(self, target, args, kwargs):
            value = self.source_node.meta['val']
            proxy = torch.fx.Proxy(self.new_graph.get_attr(target), self.tracer)
            track_tensor_tree(value, proxy, constant=None, tracer=self.tracer)
            return value

        def run_node(self, node):
            self.source_node = node
            before = graph._root.prev
            with traceback.set_current_meta(node, 'mesh tensor equations'):
                if node.op == 'output':
                    graph.output(torch.fx.node.map_arg(node.args[0], results.__getitem__))
                    return self.fetch_args_kwargs_from_env(node)[0][0]
                if node in preserve:
                    made = graph.node_copy(node, results.__getitem__)
                    made.meta = node.meta.copy()
                    value = node.meta['val']
                    track_tensor_tree(value, torch.fx.Proxy(made, self.tracer), constant=None, tracer=self.tracer)
                    results[node], operations[node] = made, (made,)
                    return value
                value = super().run_node(node)
                origins, created = _source_origins(node, graph, before)
                declared = node.meta.get('val', value)

                def retain(path, item):
                    if not isinstance(item, torch.Tensor):
                        return item
                    made = get_proxy_slot(item, self.tracer).proxy.node
                    if made not in created:
                        # An identity equation still has a distinct declared
                        # result. The ordinary alias is a page view, not a copy.
                        item = torch.ops.aten.alias.default(item)
                        made = get_proxy_slot(item, self.tracer).proxy.node
                    metadata = item if node.meta.get('mesh_local_scope') else pytree.key_get(declared, path)
                    _source_result(node, made, path, declared, metadata, origins)
                    return item

                value = pytree.tree_map_with_path(retain, value)
                results[node] = pytree.tree_map(
                    lambda item: get_proxy_slot(item, self.tracer).proxy.node if isinstance(item, torch.Tensor) else item,
                    value)
                current, introduced = before.next, []
                while current is not graph._root:
                    introduced.append(current)
                    current = current.next
                operations[node] = tuple(introduced)
                return value

    metadata = [node.meta['val'] for node in module.graph.nodes if node.op == 'placeholder']
    mode = detect_fake_mode([node.meta['val'] for node in module.graph.nodes if 'val' in node.meta])
    with torch.no_grad(), mode, traceback.preserve_node_meta():
        Normalize(module, graph, decompositions).run(*metadata, enable_io_processing=False)
    normalized = torch.fx.GraphModule(module, graph)
    normalized.meta = dict(module.meta, mesh_source_results=_source_results(module, results),
                           mesh_source_operations=operations)
    normalized.recompile()
    return normalized
