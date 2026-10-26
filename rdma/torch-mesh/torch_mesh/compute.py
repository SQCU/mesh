"""Resolve explicit backend manifests and live device limits before compilation."""
import json
from pathlib import Path

from mesh_compute import hardware
from . import _stream


def metal(directory):
    """A manifest directory is an operand, relative to the loaded mesh source.

    Metal supplies live resource limits; the family manifest supplies the matrix
    instruction extent and alignment. Neither source replaces a missing other.
    """
    import mesh_compute
    directory = Path(directory).expanduser()
    if not directory.is_absolute():
        directory = Path(mesh_compute.__file__).resolve().parents[1] / directory
    family, threads, memory = _stream.device_facts()
    path = directory / f'apple{family}.json'
    held = json.loads(path.read_text())
    fields = ('max_threads_per_threadgroup', 'threadgroup_memory_bytes',
              'threadgroup_memory_alignment_bytes', 'simdgroup_matrix_edge')
    if held['family'] != family or any(type(held[k]) is not int or held[k] <= 0 for k in fields):
        raise ValueError(f'{path}: invalid Metal capability fields')
    if threads <= 0 or memory <= 0:
        raise RuntimeError(f'{path}: Metal did not report thread or memory limits')
    return {**held, 'max_threads_per_threadgroup': min(threads, held['max_threads_per_threadgroup']),
            'threadgroup_memory_bytes': min(memory, held['threadgroup_memory_bytes']),
            'hardware': hardware()}
