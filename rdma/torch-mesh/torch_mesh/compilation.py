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
def api_layouts(name):
    """Physical operand requirements of a declared native API, at preparation."""
    import json
    from pathlib import Path
    import mesh_compute
    path = Path(mesh_compute.__file__).resolve().parents[1] / 'etc/compute/metal-apis.json'
    return json.loads(path.read_text())[name]
