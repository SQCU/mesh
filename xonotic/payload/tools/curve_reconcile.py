import ctypes
import dataclasses
import hashlib
import os
import platform
import subprocess
import tempfile

import numpy as np

@dataclasses.dataclass(frozen=True)
class CurveWeights:
    anchor: float
    strain: float
    bend: float
    cusp: float
    tangent_point: float
    thickness: float
    length_scale: float
    thickness_scale: float
    tangent_power: float
    thickness_power: float
    cusp_epsilon: float
    spatial_epsilon: float

class _CurveWeights(ctypes.Structure):
    _fields_ = [(field.name, ctypes.c_double) for field in dataclasses.fields(CurveWeights)]

class _CurveMeasures(ctypes.Structure):
    _fields_ = [
        ('anchor_energy', ctypes.c_double),
        ('strain_energy', ctypes.c_double),
        ('bend_energy', ctypes.c_double),
        ('cusp_energy', ctypes.c_double),
        ('tangent_point_energy', ctypes.c_double),
        ('thickness_energy', ctypes.c_double),
        ('total_energy', ctypes.c_double),
        ('minimum_turn_cosine', ctypes.c_double),
        ('minimum_nonneighbor_segment_distance', ctypes.c_double),
        ('turn_atom_mass', ctypes.c_size_t),
        ('directed_tangent_point_pair_mass', ctypes.c_size_t),
        ('nonneighbor_segment_pair_mass', ctypes.c_size_t),
    ]

_LIBRARY = None

def _library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(root, 'curve_reconcile.c')
    header = os.path.join(root, 'curve_reconcile.h')
    digest = hashlib.sha256(open(source, 'rb').read() + open(header, 'rb').read()).hexdigest()[:20]
    suffix = '.dylib' if platform.system() == 'Darwin' else '.so'
    output = os.path.join(tempfile.gettempdir(), 'mesh-curve-reconcile-' + digest + suffix)
    if not os.path.exists(output):
        temporary = output + '.%d' % os.getpid()
        command = [os.environ.get('CC', 'cc'), '-O3', '-std=c11']
        command += ['-dynamiclib'] if suffix == '.dylib' else ['-shared', '-fPIC']
        command += [source, '-lm', '-o', temporary]
        subprocess.run(command, check=True)
        os.replace(temporary, output)
    library = ctypes.CDLL(output)
    pointer = ctypes.POINTER(ctypes.c_double)
    library.MeshCurveAccumulate.argtypes = [
        pointer, pointer, ctypes.c_size_t, ctypes.c_int,
        ctypes.POINTER(_CurveWeights), pointer, ctypes.POINTER(_CurveMeasures),
    ]
    library.MeshCurveAccumulate.restype = None
    _LIBRARY = library
    return library

def accumulate(points, reference, weights, closed=False):
    current = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    initial = np.ascontiguousarray(reference, dtype=np.float64).reshape(current.shape)
    gradient = np.zeros_like(current)
    cweights = _CurveWeights(**dataclasses.asdict(weights))
    measures = _CurveMeasures()
    pointer = ctypes.POINTER(ctypes.c_double)
    _library().MeshCurveAccumulate(
        current.ctypes.data_as(pointer), initial.ctypes.data_as(pointer), len(current),
        int(bool(closed)), ctypes.byref(cweights), gradient.ctypes.data_as(pointer),
        ctypes.byref(measures),
    )
    return gradient, {
        name: getattr(measures, name)
        for name, _ in _CurveMeasures._fields_
    }
