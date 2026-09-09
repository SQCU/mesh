import json
import os

import numpy as np

from ..array_tree import pack_state, unpack_state, atomic_save


def matrix_factor(left, right, mass=1):
    denominator = max(mass, 1)
    row_square = np.sum(left * ((right @ right.T) @ left), axis=0) / denominator ** 2
    return {'representation': 'left_transpose_times_right_over_mass',
            'left': left, 'right': right, 'mass': mass, 'shape': [left.shape[1], right.shape[1]],
            'row_norm': np.sqrt(np.maximum(row_square, 0)),
            'frobenius_norm': float(np.sqrt(max(float(row_square.sum()), 0)))}


def scalar_report(report):
    def coordinate(value):
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": value.dtype.str, "array_length": len(value) if value.ndim else 1}
        if isinstance(value, dict):
            return {"mapping_entries": len(value)}
        if isinstance(value, (list, tuple)):
            return {"array_length": len(value)}
        return value.item() if isinstance(value, np.generic) else value
    return {section: {key: coordinate(value) for key, value in values.items()} if isinstance(values, dict) else coordinate(values) for section, values in report.items()}


def write_report(path, report):
    atomic_save(path, pack_state(report, '__report__'))


def read_report(path):
    with np.load(path, allow_pickle=False) as archive:
        if '__report__meta' in archive:
            return unpack_state(archive, '__report__')
        arrays = {}
        def resolve(value):
            if isinstance(value, dict):
                if set(value) == {"__ndarray__"}:
                    key = value["__ndarray__"]
                    if key not in arrays:
                        arrays[key] = archive[key]
                    return arrays[key]
                return {key: resolve(child) for key, child in value.items()}
            if isinstance(value, list):
                return [resolve(child) for child in value]
            return value

        return resolve(json.loads(archive["__manifest__"].tobytes())["report"])
