"""Whole-function statistics; execution and completion belong to the caller.

The C header supplies the clock and the paired latency/rate calculation. Late
completion observations can be collected without changing numerical execution.
"""
import json
import ctypes
import contextlib
import copy
import threading
from pathlib import Path


EXECUTED = ('configuration', 'episode', 'proposal', 'parts', 'domains', 'operands', 'graphs', 'layouts', 'choices', 'workload', 'implementation', 'family', 'importance')

class _Value(ctypes.Structure):
    _fields_ = [('seconds', ctypes.c_double), ('useful_gigabytes_per_second', ctypes.c_double),
                ('units_per_second', ctypes.c_double)]


_lib = ctypes.CDLL(str(Path(__file__).with_name('libmesh-statistics.dylib')))
_lib.mesh_statistics_now.restype = ctypes.c_double
_lib.mesh_statistics_measure.argtypes = [ctypes.c_double] * 4
_lib.mesh_statistics_measure.restype = _Value
now = _lib.mesh_statistics_now
_lib.mesh_statistics_streams.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_ulong, ctypes.POINTER(ctypes.c_int)]
_lib.mesh_statistics_streams.restype = ctypes.c_long


class StreamWords:
    """this host's stream words beside a run, `with StreamWords() as words: <the run>`, then words.chains(): the C
    spin (mesh-statistics.h mesh_statistics_streams) on a thread of its own, which ctypes runs without the GIL, so the
    run's own thread is not held; it stops when the block ends. `capacity` moves kept at most (32 bytes each)."""

    def __init__(self, capacity=1 << 18):
        self.capacity, self.count = capacity, None
        self.buffer = (ctypes.c_ulonglong * (4 * capacity))()
        self.stop = ctypes.c_int(0)
        self.thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        self.count = _lib.mesh_statistics_streams(self.buffer, self.capacity, ctypes.byref(self.stop))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.value = 1
        self.thread.join(2.0)

    def moves(self):
        """(ns, stream, word, value) each kept move (word 0 owner, 1 ready, 2 sent, 3 landed)"""
        if self.count is None or self.count < 0:
            raise RuntimeError(f'no stream words (mesh_statistics_streams: {self.count})')
        b = self.buffer
        return [(b[4 * k], b[4 * k + 1], b[4 * k + 2], b[4 * k + 3]) for k in range(self.count)]

    def chains(self):
        """each tick's chain on this host, microseconds from its last landing (every held stream's `landed` at that
        tick's count) to its next release (`ready`'s next rise on the stream releasing least often, a stream carrying
        a forward as well rising twice a tick): the rank's compute in the run itself, as tools/mesh/tick_words.py
        reads it from mesh-stat (metal-microbench)"""
        owner, series = {}, {}
        for ns, t, w, v in self.moves():
            if w == 0:
                owner[t] = v
            elif v < 1 << 62 and 0 < owner.get(t, 0) < 1 << 63:
                series.setdefault((t, w), []).append((ns, v))
        rises = {t: [ns for (a, b) in zip(s, s[1:]) for ns in [b[0]] if b[1] > a[1]]
                 for (t, w), s in series.items() if w == 1}
        rises = {t: r for t, r in rises.items() if len(r) > 2}
        if not rises:
            return []
        releases = min(rises.values(), key=len)
        ticks = len(releases)
        arrivals = []
        for (t, w), s in series.items():
            if w != 3 or s[-1][1] <= s[0][1]:
                continue
            base, step = s[0][1], (s[-1][1] - s[0][1]) / ticks
            arrivals.append([next((ns for ns, v in s if v >= base + round((j + 1) * step)), None) for j in range(ticks)])
        out = []
        for j in range(ticks - 1):
            last = [a[j] for a in arrivals]
            if last and None not in last and releases[j + 1] > max(last):
                out.append((releases[j + 1] - max(last)) / 1e3)
        return out


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

    A caller that submits several functions back to back measures each with its
    actual queue residence and contention. Observation adds no start barrier or
    completion gate between submissions.
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
        # Callers report the logical output of the complete function, not wire
        # amplification or a per-rank byte sum. It is fixed by this workload.
        useful = first['useful_output_bytes']
        if any(row['useful_output_bytes'] != useful for row in members.values()):
            raise ValueError(f'{episode}: ranks disagree on whole-function useful work')
        complete.append({**{key: copy.deepcopy(first[key]) for key in EXECUTED if key in first},
                         'scope': 'function', 'complete': True, 'size': first['size'],
                         'seconds': seconds, 'latency_seconds': seconds,
                         'useful_output_bytes': useful,
                         'useful_gigabytes_per_second': measure(0, seconds, useful)['useful_gigabytes_per_second'],
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
    return {'placement': {key: declaration[key] for key in ('parts', 'operands', 'graphs', 'layouts', 'choices')},
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
    ``close`` makes the same nonblocking observation and reports whether all
    submitted episodes are complete. The application owns shutdown.
    """

    def __init__(self, declaration=None):
        self.submitted, self.records = [], []
        self.declaration = declaration or {}  # the preparation's own record, which nothing writes after (configuration.bind)
        # The declaration's line (megabytes: its graphs and layouts) is written at the first report, after the
        # program it declares has started, never between a prepared program and its start: there it took 1 to 6 s
        # of a pipe's time a rank (chain service, 2026-10-31), longer than a peer's GPU wait is bounded, so the ranks
        # that started first ended their waits.  rehydrate reads it before any completion that refers to it.
        self.declared = not self.declaration

    def _declare(self):
        if not self.declared:
            self.declared = True
            emit({'phase': 'observation_configuration', **self.declaration})

    def put(self, completion, record):
        self.submitted.append((completion, record))
        self._emit()

    def close(self):
        self._declare()
        self._emit()
        return not self.submitted

    def poll(self):
        """Ready reports for the caller's next preparation; never wait."""
        self._emit()
        records, self.records = self.records, []
        return records

    def _emit(self):
        from torch_mesh import _stream
        index = 0
        while index < len(self.submitted):
            completion, record = self.submitted[index]
            end = _stream.completion_observe(completion)
            if end is None:
                index += 1
                continue
            self.submitted.pop(index)
            result = completed(record, end)
            self.records.append(result)
            self._declare()
            # The full executed operands stay on the preparation channel. A
            # completion line references the declaration emitted once above.
            retained = {key: value for key, value in result.items()
                        if key in ('configuration', 'episode', 'proposal', 'rank', 'size')
                        or key not in self.declaration or value != self.declaration[key]}
            emit([retained])
