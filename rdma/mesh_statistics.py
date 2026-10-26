"""Whole-function statistics; execution and completion belong to the caller.

The C header supplies the clock and the paired latency/rate calculation. Late
completion observations can be collected without changing numerical execution.
"""
import json
import ctypes
from pathlib import Path

class _Value(ctypes.Structure):
    _fields_ = [('seconds', ctypes.c_double), ('useful_gigabytes_per_second', ctypes.c_double),
                ('units_per_second', ctypes.c_double)]


_lib = ctypes.CDLL(str(Path(__file__).with_name('libmesh-statistics.dylib')))
_lib.mesh_statistics_now.restype = ctypes.c_double
_lib.mesh_statistics_measure.argtypes = [ctypes.c_double] * 4
_lib.mesh_statistics_measure.restype = _Value
now = _lib.mesh_statistics_now


def measure(started, observed, useful_bytes, units=0):
    value = _lib.mesh_statistics_measure(started, observed, useful_bytes, units)
    return {'seconds': value.seconds,
            'useful_gigabytes_per_second': value.useful_gigabytes_per_second,
            'units_per_second': value.units_per_second}


def emit(record, *, file=None):
    records = record if isinstance(record, list) else [record]
    print('\n'.join(json.dumps(row, separators=(',', ':'), sort_keys=True) for row in records), file=file, flush=True)


def completed(record, seconds):
    measured = measure(0, seconds, record['useful_work_bytes'], record['report'].get('steps', 0))
    result = {**record['report'], 'rank': record['rank'], 'size': record['size'],
              'seconds': measured['seconds'], 'useful_output_bytes': record['useful_work_bytes'],
              'useful_gigabytes_per_second': measured['useful_gigabytes_per_second'],
              'timing_method': 'host-clock-observation-of-ordinary-completion'}
    if 'steps' in result:
        result.update(step_microseconds=seconds / result['steps'] * 1e6,
                      calls_per_second=measured['units_per_second'])
        if 'rows' in result:
            result['aggregate_rows_per_second'] = measured['units_per_second'] * result['rows']
    if 'calls' in result:
        result['microseconds_per_call'] = seconds * 1e6 / result['calls']
        result['useful_output_gigabytes_per_second_rank'] = measured['useful_gigabytes_per_second']
    return result


def aggregate(ranks):
    """Whole-workload intervals and useful delivery rates from actual rank records."""
    if not ranks:
        return []
    rows = {}
    for rank in ranks.values():
        for value in rank['measurements']:
            key = (value.get('what'), value.get('requested_bytes'))
            if 'seconds' in value or 'microseconds_per_call' in value:
                rows.setdefault(key, []).append(value)
    result = []
    for (operation, requested), values in rows.items():
        if len(values) != len(ranks):
            continue
        if operation is not None:
            interval = max(value['microseconds_per_call'] for value in values) * 1e-6
            useful = sum(value['useful_output_bytes_per_call_rank'] for value in values)
            result.append({'operation': operation, 'requested_bytes': requested,
                           'amortized_seconds_per_call': interval,
                           'useful_bytes_per_call': useful,
                           'useful_gigabytes_per_second': measure(0, interval, useful)['useful_gigabytes_per_second']})
        else:
            seconds = max(value['seconds'] for value in values)
            useful = values[0].get('useful_output_bytes')
            result.append({'function_seconds': seconds,
                           'useful_output_bytes': useful,
                           'useful_gigabytes_per_second': measure(0, seconds, useful)['useful_gigabytes_per_second'] if useful is not None else None})
            if all('rows' in value and 'steps' in value for value in values):
                result[-1].update(completed_rows=values[0]['rows'] * values[0]['steps'],
                                  aggregate_rows_per_second=measure(0, seconds, 0, values[0]['rows'] * values[0]['steps'])['units_per_second'])
    return result

