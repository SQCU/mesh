"""Compiler operands scoped to one graph realization, never runtime policy."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from . import _stream

_bound = ContextVar('mesh_compilation_operands', default=None)


def operands():
    held = _bound.get()
    return _stream.compilation if held is None else held


@contextmanager
def bind(configuration):
    values = dict(operands())
    if configuration is not None:
        for key, value in configuration.items():
            if key not in values:
                raise ValueError(f'unknown compiler operand {key}')
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key}: compiler operand must be a positive integer')
            values[key] = value
    token = _bound.set(values)
    try:
        yield values
    finally:
        _bound.reset(token)


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
    graph = torch.fx.Graph()
    graph.output(graph.graph_copy(module.graph, {}))
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
        if node is first:
            first = node.next
        graph.erase_node(node)
    lifted.recompile()
    return lifted, [*arguments, *values]


def lift_attributes(module, arguments):
    """Bind source state to page operands before splitting numerical spans."""
    import torch
    lifted, operands = lift_state(module, arguments)
    from torch.distributed.tensor import DTensor
    values = operands[len(arguments):]
    sources = [value._local_tensor if isinstance(value, DTensor) else value for value in values]
    imported = _stream.import_attributes(sources)
    # Each import takes its attribute's place (the module's own tensor now names the region pages), so the foreign
    # storage is freed, not held beside its region copy: one residency per attribute, aliases kept as views.
    with torch.no_grad():
        for value, made in zip(sources, imported):
            if made is not value:
                value.data = made
    return lifted, operands


def source_expression(function, *arguments):
    """Capture a pure value expression while retaining its tensor-state edges.

    Mutation in a value declaration is functionalized; no input writeback is
    added. The returned source graph still contains any declared collective or
    postprocessing. It has not been partitioned or lowered to native commands.
    """
    import torch
    from torch._subclasses.functional_tensor import dispatch_functionalize
    from torch.fx.experimental.proxy_tensor import make_fx
    with torch.no_grad():
        module = make_fx(dispatch_functionalize(function), decomposition_table={},
                         tracing_mode='fake', _allow_non_fake_inputs=True)(*arguments)
    return lift_state(module, arguments)


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


def normalize_source(module, decompositions):
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
    if not any(node.op == 'call_function' and node.target in decompositions for node in module.graph.nodes):
        return module
    graph, results = torch.fx.Graph(), {}

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
                value = super().run_node(node)
                created = []
                current = before.next
                while current is not graph._root:
                    created.append(current)
                    current = current.next
                origins = node.meta.get('mesh_source_nodes', (node,))
                for made in created:
                    made.meta['mesh_source_nodes'] = origins
                if node.op == 'output':
                    return value
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
                    provenance = made.meta.get('from_node')
                    metadata = node.meta.copy()
                    metadata['val'] = pytree.key_get(declared, path)
                    for field in ('mesh_ownership', 'mesh_local_source', 'mesh_gradient'):
                        if field in metadata and not isinstance(declared, torch.Tensor):
                            metadata[field] = pytree.key_get(metadata[field], path)
                    made.meta.update(metadata)
                    if provenance is not None:
                        made.meta['from_node'] = provenance
                    made.meta['mesh_source'] = node.meta.get('mesh_source', node.name)
                    made.meta['mesh_source_nodes'] = origins
                    made.meta['mesh_result_path'] = path
                    return item

                value = pytree.tree_map_with_path(retain, value)
                results[node] = pytree.tree_map(
                    lambda item: get_proxy_slot(item, self.tracer).proxy.node if isinstance(item, torch.Tensor) else item,
                    value)
                return value

    metadata = [node.meta['val'] for node in module.graph.nodes if node.op == 'placeholder']
    mode = detect_fake_mode([node.meta['val'] for node in module.graph.nodes if 'val' in node.meta])
    with torch.no_grad(), mode, traceback.preserve_node_meta():
        Normalize(module, graph, decompositions).run(*metadata, enable_io_processing=False)
    normalized = torch.fx.GraphModule(module, graph)
    normalized.meta = dict(module.meta, mesh_source_results=results)
    normalized.recompile()
    return normalized
