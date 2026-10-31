"""Whole-function statistics; execution and completion belong to the caller.

The C header supplies the clock and the paired latency/rate calculation. Late
completion observations can be collected without changing numerical execution.
"""
import json
import ctypes
import contextlib
import copy
from pathlib import Path


EXECUTED = ('configuration', 'episode', 'proposal', 'parts', 'domains', 'operands', 'graphs', 'layouts', 'workload', 'implementation', 'family', 'importance')

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


@contextlib.contextmanager
def recorded_values(records):
    """Paired values already recorded by the canonical hourly, without running code."""
    return [{'commits': record['commits'], 'unix': record['unix'], 'workload': name, 'configuration_index': index,
             'seconds': pair[0] * 1e-6, 'useful_gigabytes_per_second': pair[1],
             **({'aggregate_rows_per_second': pair[2]} if len(pair) > 2 else {})}
            for record in records for name, pairs in record['us,GB/s'].items() for index, pair in enumerate(pairs)]


def completed(record, ended):
    """One submitted function's latency and useful rate over that same interval.

    Other submissions may overlap it. Neither their completion order nor their
    end timestamps establish when this function began executing.
    """
    measured = measure(record['started'], ended, record['useful_work_bytes'], record['report'].get('steps', 0))
    seconds = measured['seconds']
    result = {**record['report'], **{key: record[key] for key in EXECUTED if key in record},
              'rank': record['rank'], 'size': record['size'], 'scope': 'function', 'complete': True,
              'submitted_at': record['started'], 'completed_at': ended,
              'seconds': seconds, 'latency_seconds': seconds, 'useful_output_bytes': record['useful_work_bytes'],
              'useful_gigabytes_per_second': measured['useful_gigabytes_per_second'],
              'timing_scope': 'submission to whole-function completion',
              'timing_method': 'caller submission clock to latest existing GPU-part or host-task completion; useful rate covers the same interval, including overlap and queue delay'}
    if 'steps' in result:
        result.update(step_microseconds=seconds / result['steps'] * 1e6,
                      calls_per_second=measured['units_per_second'])
        if 'rows' in result:
            result['aggregate_rows_per_second'] = measured['units_per_second'] * result['rows']
    if 'calls' in result:
        result['microseconds_per_call'] = seconds * 1e6 / result['calls']
        result['useful_output_gigabytes_per_second_rank'] = measured['useful_gigabytes_per_second']
    return result


def cohorts(records):
    """Join completed rank reports by their executed episode, retaining late members.

    This is preparation data: no completion is inspected and no collective is
    issued here. The caller supplies every available report, including pending
    rows from its previous preparation.
    """
    grouped = []
    reports = [row for record in records for row in record.get('ranks', [record])]
    for record in reports:
        if record.get('scope') != 'function' or record.get('complete') is not True:
            raise ValueError('an observation must describe a completed function')
        for key in ('episode', 'proposal', 'parts', 'domains', 'graphs', 'workload', 'implementation', 'family'):
            if key not in record:
                raise ValueError(f'an executed observation needs {key}')
        episode = record['episode']
        members = next((members for family, configuration, key, members in grouped
                        if family == record['family'] and configuration == record.get('configuration') and key == episode), None)
        if members is None:
            members = {}
            grouped.append((record['family'], record.get('configuration'), episode, members))
        rank, size = record['rank'], record['size']
        if type(rank) is not int or type(size) is not int or size <= 0 or not 0 <= rank < size:
            raise ValueError(f'{episode}: invalid observation rank or size')
        if members:
            previous = next(iter(members.values()))
            for key in (*EXECUTED, 'size'):
                if previous.get(key) != record.get(key):
                    raise ValueError(f'{episode}: ranks disagree on executed {key}')
        if rank in members and members[rank] != record:
            raise ValueError(f'{episode}: conflicting completion for rank {rank}')
        members[rank] = record
    complete, pending = [], []
    for _, _, episode, members in grouped:
        first = next(iter(members.values()))
        if len(members) != first['size']:
            pending.extend(members.values())
            continue
        seconds = max(row['seconds'] for row in members.values())
        complete.append({**{key: copy.deepcopy(first[key]) for key in EXECUTED if key in first},
                         'scope': 'function', 'complete': True, 'size': first['size'],
                         'seconds': seconds, 'latency_seconds': seconds,
                         'timing_scope': 'maximum rank submission-to-completion latency',
                         'ranks': [members[rank] for rank in range(first['size'])]})
    return complete, pending


def rehydrate(records, declarations):
    """Resolve compact completion references using declared preparation metadata.

    The coordinator owns ``declarations`` in memory. This function only folds
    supplied records; it reads no streams, polls no completions, and plans no
    work. A late metric retains the configuration under which it executed.
    """
    declarations = dict(declarations)
    complete = []
    for record in records:
        if record.get('phase') == 'observation_configuration':
            key = record['configuration'], record['rank']
            value = {k: v for k, v in record.items() if k != 'phase'}
            if key in declarations and declarations[key] != value:
                raise ValueError(f'{key}: conflicting prepared observation declaration')
            declarations[key] = value
        elif record.get('scope') == 'function' and record.get('complete') is True:
            if 'configuration' in record:
                key = record['configuration'], record['rank']
                if key not in declarations:
                    raise ValueError(f'{key}: completion has no prepared declaration')
                declared = declarations[key]
                if any(key in record and record[key] != value for key, value in declared.items()):
                    raise ValueError(f'{key}: completion changed its prepared operands')
                complete.append({**declared, **record})
            else:
                complete.append(record)
    return complete, declarations


def next_configuration(declaration, observations, *, family):
    """The next caller input from a prepared result and already supplied reports.

    ``family`` is the caller's semantic declaration, distinct from the executed
    family's member/routing/layout basis. No future completion is implied and
    no proposal is solved here; the next ordinary preparation does that.
    """
    return {'placement': {key: declaration[key] for key in ('parts', 'operands', 'graphs', 'layouts')},
            'observations': list(observations), 'proposal': declaration['proposal'] + 1, 'family': family}


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



class Observations:
    """Retain each episode with its own declaration and submission timestamp.

    ``put`` and ``poll`` publish already-completed episodes without waiting for
    earlier submissions. Completion order does not imply execution order. The
    existing GPU/host end timestamp measures each call's own latency; a new
    configuration does not reset or inherit a fictitious service-time clock.
    ``close`` drains only at the caller's administrative shutdown.
    """

    def __init__(self, declaration=None):
        self.submitted, self.records = [], []
        self.declaration = copy.deepcopy(declaration or {})
        if self.declaration:
            emit({'phase': 'observation_configuration', **self.declaration})

    def put(self, completion, record):
        self.submitted.append((completion, record))
        self._emit(False)

    def close(self):
        self._emit(True)

    def poll(self):
        """Ready reports for the caller's next preparation; never wait."""
        self._emit(False)
        records, self.records = self.records, []
        return records

    def _emit(self, waiting):
        from torch_mesh import _stream
        index = 0
        while index < len(self.submitted):
            completion, record = self.submitted[index]
            if not waiting and not _stream.completion_ended(completion):
                index += 1
                continue
            end = _stream.completion_end(completion)
            self.submitted.pop(index)
            result = completed(record, end)
            self.records.append(result)
            # The full executed operands stay on the preparation channel. A
            # completion line references the declaration emitted once above.
            retained = {key: value for key, value in result.items()
                        if key in ('configuration', 'episode', 'proposal', 'rank', 'size')
                        or key not in self.declaration or value != self.declaration[key]}
            emit([retained])
