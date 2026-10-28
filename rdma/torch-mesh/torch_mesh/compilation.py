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


def lift_attributes(module, arguments):
    """Bind tensor attributes once, before a whole graph is split into spans.

    Imported contents remain ordinary page data. Keeping them as operands also
    preserves writes and aliases across numerical and collective boundaries.
    """
    import torch
    if not any(node.op == 'get_attr' for node in module.graph.nodes):
        return module, list(arguments)
    graph = torch.fx.Graph()
    graph.output(graph.graph_copy(module.graph, {}))
    lifted = torch.fx.GraphModule(module, graph)
    values, bindings = [], {}
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
            bindings[id(value)] = operand
            values.append(value)
        node.replace_all_uses_with(operand)
        if node is first:
            first = node.next
        graph.erase_node(node)
    lifted.recompile()
    return lifted, [*arguments, *_stream.import_attributes(values)]
