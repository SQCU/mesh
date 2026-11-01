"""Compiler operands scoped to one graph realization, never runtime policy."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from . import _stream

_bound = ContextVar('mesh_compilation_operands', default=None)
_layouts = ContextVar('mesh_compilation_layouts', default=None)
_implementations = ContextVar('mesh_compilation_implementations', default=None)
_assignment = ContextVar('mesh_compilation_implementation_assignment', default=None)
_relations = ContextVar('mesh_compilation_relations', default=None)
_choices = ContextVar('mesh_compilation_choices', default=None)
_source_capture = ContextVar('mesh_compilation_source_capture', default=None)
_choice_member = ContextVar('mesh_compilation_choice_member', default=None)


def relations():
    """The explicit relation context of the function being prepared."""
    value = _relations.get()
    if value is None:
        raise ValueError('page relation construction requires a compilation context')
    return value


def operands():
    held = _bound.get()
    return _stream.compilation if held is None else held


def implementations():
    """Explicit available compiler implementations for this preparation."""
    import mesh_compute
    held = _implementations.get()
    return mesh_compute.implementations() if held is None else held


def implementation_assignment():
    """Implementation choices supplied to preparation, independent of storage."""
    import mesh_compute
    held = _assignment.get()
    return mesh_compute.implementation_assignment() if held is None else held


@contextmanager
def bind(configuration, *, algebra=None):
    values = dict(operands())
    previous_layouts = _layouts.get()
    preferences = () if previous_layouts is None else previous_layouts['preferences']
    available = implementations()
    assigned = implementation_assignment()
    relation_context = _relations.get() if algebra is None else algebra
    previous_choices = _choices.get()
    choice_preferences = () if previous_choices is None else previous_choices['preferences']
    if configuration is not None:
        for key, value in configuration.items():
            if key == 'source_preferences':
                choice_preferences = value
                continue
            if key == 'layout_preferences':
                preferences = value
                continue
            if key == 'implementations':
                available = value
                continue
            if key == 'implementation_assignment':
                assigned = value
                continue
            if key not in values:
                raise ValueError(f'unknown compiler operand {key}')
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key}: compiler operand must be a positive integer')
            values[key] = value
    token = _bound.set(values)
    layouts = _layouts.set(previous_layouts if configuration is None and previous_layouts is not None
                           else dict(preferences=preferences, choices=[]))
    candidates = _implementations.set(available)
    assignments = _assignment.set(assigned)
    relations_token = _relations.set(relation_context)
    choice_token = _choices.set(previous_choices if configuration is None and previous_choices is not None
        else dict(preferences=choice_preferences, records=[]))
    try:
        yield values
    finally:
        _choices.reset(choice_token)
        _relations.reset(relations_token)
        _assignment.reset(assignments)
        _implementations.reset(candidates)
        _layouts.reset(layouts)
        _bound.reset(token)


@contextmanager
def capture_choices(scope):
    """Source occurrences are preparation identity, never runtime dispatch."""
    token = _source_capture.set(dict(scope=scope, fold=0))
    try:
        yield
    finally:
        _source_capture.reset(token)


def fold_occurrence():
    capture = _source_capture.get()
    if capture is None:
        return None
    ordinal = capture['fold']
    capture['fold'] += 1
    return None if capture['scope'] is None else dict(scope=capture['scope'], occurrence=ordinal)


def source_choice(family, weights, *, grain, high=None, quantum=None):
    """Consume and retain the executed simplex through ordinary preparation."""
    import json
    from allocate import project, simplex
    family = json.loads(json.dumps(family, sort_keys=True, allow_nan=False))
    held = _choices.get()
    matches = [] if held is None else [record for record in held['preferences'] if record['family'] == family]
    if len(matches) > 1:
        raise ValueError('one source choice has multiple supplied operands')
    values = list(matches[0]['weights'] if matches else weights)
    high = [1] * len(values) if high is None else list(high)
    total = sum(values)
    values = [value / total for value in values]
    values = (project(values, 1, grain, [0] * len(values), high) if grain else
              simplex(values, 1, [0] * len(values), high))
    record = dict(family=family, weights=values, grain=grain, high=high)
    if quantum is not None:
        record['quantum'] = quantum
    if held is not None:
        held['records'].append(record)
    return record


def source_choices():
    from assignment import source_axes
    held = _choices.get()
    return source_axes(()) if held is None else source_axes(held['records'])


def source_preferences():
    held = _choices.get()
    return () if held is None else held['preferences']


def retain_source_choices(records):
    held = _choices.get()
    if held is not None:
        held['records'].extend(records)


@contextmanager
def choice_member(member):
    token = _choice_member.set(member)
    try:
        yield
    finally:
        _choice_member.reset(token)


def implementation_choice(required, assignment, *, source, region=False):
    """Realize configured backend alternatives before source/placement lowering.

    Each candidate is a complete invocation of this retained source domain.
    Its unit granularity is explicit; selected page domains can already divide
    a larger function into independently assigned invocations.
    """
    from mesh_compute import assignment_rule, eligibility
    rule = assignment_rule(required, assignment)
    options = rule.get('alternatives', [dict(implementation=rule['implementation'],
        **({'configuration': rule['configuration']} if 'configuration' in rule else {}))])
    catalog = implementations()
    high = []
    for option in options:
        candidate = catalog[option['implementation']]
        typed = dict(required, interfaces=['scalar']) if region and candidate['compiler'] in ('metal_scalar', 'cpp_scalar') else required
        high.append(int(not eligibility(typed, candidate)['missing']))
    initial = [int(option['implementation'] == rule['implementation']) for option in options]
    if not source:
        return options[initial.index(1)]
    record = source_choice(dict(kind='backend', member=_choice_member.get(), source=source, required=required,
        alternatives=options), initial, grain=1, high=high)
    return options[record['weights'].index(1)]


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
        for field in ('mesh_source_records', 'mesh_route_scope', 'mesh_implementation'):
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


def assign_regions(module, *, materializations=(), cuts=()):
    """Lower configured captured source regions before page placement.

    Regions are source operands, not runtime tasks. Collective boundaries and
    explicit source cuts delimit them. Selection uses retained source names and
    declared types/shapes; changing tensor/index/mask values are never read.
    The original captured graph (including its derivative) remains the source
    of every replacement and its returned-field identities.
    """
    rules = implementation_assignment().get('regions', ())
    if not rules or module.meta.get('mesh_regions_lowered'):
        return module
    import torch
    from torch.fx.node import map_arg
    from torch.utils import _pytree as pytree
    from mesh_compute import matches, select_implementation
    from . import host
    from .declarations import _bound, sections
    from .engines import graph_source
    from .algebra import decompositions
    nodes = list(module.graph.nodes)
    order = {node: i for i, node in enumerate(nodes)}
    names = {node.name: node for node in nodes}
    requested = tuple(dict.fromkeys((*materializations,
        *(node for node in nodes if 'mesh_ownership' in node.meta))))
    boundaries = {at for node, i in order.items() if _bound(node) or host.bound(node) for at in (i, i + 1)}
    boundaries.update(order[node] + (side == 'after') for node, side in cuts)
    for rule in rules:
        selected = rule.get('nodes')
        if selected is not None:
            missing = set(selected) - names.keys()
            if missing:
                raise ValueError(f'provider region names absent source nodes: {sorted(missing)}')
            selected = set(selected)
            prior = False
            for i, node in enumerate(nodes):
                inside = node.name in selected
                if inside != prior:
                    boundaries.add(i)
                prior = inside
            if prior:
                boundaries.add(len(nodes))
    graph, bound = torch.fx.Graph(), {}
    for node in nodes:
        if node.op == 'placeholder':
            made = graph.node_copy(node, bound.__getitem__);made.meta = node.meta.copy();bound[node] = made
    owners, decisions, implementation_boundaries = list(module.meta.get('mesh_provider_owners', ())), [], []
    operations = {}
    for fragment, inputs, outputs in sections(module, sorted(boundaries), requested_outputs=requested):
        source_nodes = tuple(node for node in fragment.graph.nodes if node.op not in ('placeholder', 'get_attr', 'output'))
        actual_names = {node.name for node in source_nodes}
        tensor_inputs = [value for node in inputs for value in pytree.tree_leaves(node.meta['val']) if isinstance(value, torch.Tensor)]
        tensor_outputs = [value for node in outputs for value in pytree.tree_leaves(node.meta['val']) if isinstance(value, torch.Tensor)]
        parameters = {'operations': len(source_nodes), 'inputs': len(inputs), 'outputs': len(outputs)}
        for side, values in (('input', tensor_inputs), ('output', tensor_outputs)):
            for index, value in enumerate(values):
                parameters[f'{side}{index}.rank'] = value.ndim
                parameters[f'{side}{index}.elements'] = value.numel()
                parameters.update((f'{side}{index}.dim{axis}', int(size)) for axis, size in enumerate(value.shape))
        required = dict(interfaces=['graph'],
            input_types=sorted({str(value.dtype).removeprefix('torch.') for value in tensor_inputs}),
            arithmetic_types=sorted({str(value.dtype).removeprefix('torch.') for node in source_nodes
                for value in pytree.tree_leaves(node.meta.get('val')) if isinstance(value, torch.Tensor)}),
            input_kinds=sorted({'tensor' if isinstance(node.meta['val'], torch.Tensor) else 'compound' for node in inputs}),
            output_kinds=sorted({'tensor' if isinstance(node.meta['val'], torch.Tensor) else 'compound' for node in outputs}),
            constants=[], source_nodes=sorted(actual_names),
            operations=sorted({str(node.target) for node in source_nodes}), parameters=parameters)
        choices = [] if any(_bound(node) or host.bound(node) for node in source_nodes) else [rule for rule in rules
            if ('nodes' not in rule or actual_names.issubset(rule['nodes'])) and matches(required, rule.get('when', {}))]
        if len(choices) > 1:
            raise ValueError(f'provider source region has overlapping assignments: {sorted(actual_names)}')
        replacement = fragment
        if choices:
            rule, = choices
            selected = implementation_choice(required, rule,
                source=tuple(record for node in source_nodes for record in source_records(node)), region=True)
            implementation = implementations()[selected['implementation']]
            if implementation['compiler'] in ('metal_scalar', 'cpp_scalar'):
                # The retained scalar compiler already consumes independently
                # assigned domains. Normalize products to their value equation
                # within this selected scalar region, then retain the choice.
                assignment = dict(implementation_assignment(), product='scalar_contraction', regions=[])
                with bind(dict(implementation_assignment=assignment)):
                    replacement = normalize_source(fragment, decompositions)
                for node in replacement.graph.nodes:
                    if node.op not in ('placeholder', 'get_attr', 'output'):
                        node.meta['mesh_implementation'] = implementation['id']
            else:
                implementation = select_implementation(required, implementations().values(), implementation['id'])
                replacement, owner, declaration = graph_source(fragment,
                    tuple(node.meta['val'] for node in inputs), implementation=implementation,
                    configuration=selected['configuration'])
                owners.append(owner)
                implementation = declaration['definition']['implementation']
            decisions.append(dict(required=required, implementation=implementation,
                source=tuple(record for node in source_nodes for record in source_records(node)),
                assignment=rule, selected=selected))
        copied = {}
        for node, original in zip((n for n in replacement.graph.nodes if n.op == 'placeholder'), inputs, strict=True):
            if original not in bound:
                made = graph.node_copy(original, bound.__getitem__);made.meta = original.meta.copy();bound[original] = made
            copied[node] = bound[original]
        before = graph._root.prev
        returned = graph.graph_copy(replacement.graph, copied)
        returned = tuple(returned) if isinstance(returned, (tuple, list)) else (returned,)
        introduced = []
        at = before.next
        while at is not graph._root:
            introduced.append(at);at = at.next
        for original, result in zip(outputs, returned, strict=True):
            bound[original] = result
            if isinstance(result, torch.fx.Node):
                value = result.meta.get('val', original.meta['val'])
                result.meta.update(original.meta, val=value)
                if choices and implementation['compiler'] in ('metal_scalar', 'cpp_scalar'):
                    result.meta['mesh_implementation'] = implementation['id']
            operations[original] = tuple(introduced)
            if choices and isinstance(result, torch.fx.Node):
                implementation_boundaries.append(result)
        # Compose the retained node map. Names select configuration regions;
        # they never establish value identity between compiler graphs.
        for original, tree in replacement.meta.get('mesh_source_results', {}).items():
            if original in order and all(node in copied for node in pytree.tree_leaves(tree) if isinstance(node, torch.fx.Node)):
                bound.setdefault(original, map_arg(tree, copied.__getitem__))
        for original, tree in fragment.meta.get('mesh_source_results', {}).items():
            if original in order and any(node in source_nodes for node in pytree.tree_leaves(tree) if isinstance(node, torch.fx.Node)):
                operations[original] = tuple(introduced)
    output = next(node for node in nodes if node.op == 'output')
    graph.output(map_arg(output.args[0], bound.__getitem__))
    graph.lint()
    made = torch.fx.GraphModule(module, graph)
    source_results = {source: map_arg(tree, bound.__getitem__) for source, tree in module.meta.get('mesh_source_results', {}).items()
                      if all(node in bound for node in pytree.tree_leaves(tree) if isinstance(node, torch.fx.Node))}
    source_results.update(bound)
    made.meta = dict(module.meta, mesh_source_results=source_results, mesh_source_operations=operations,
                     mesh_regions_lowered=True, mesh_provider_owners=tuple(owners), mesh_implementation_regions=tuple(decisions),
                     mesh_implementation_boundaries=tuple(implementation_boundaries))
    return made
