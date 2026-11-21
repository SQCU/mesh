"""The trainer's telemetry table (telemetry.h) read where it lies: the shared memory object is mapped read-only and sampled; a sample's
record is the counters' rates since the last sample, the gauges, and the phases' seconds since the start. Nothing is
written by the trainer to report them: the table is the store."""
import mmap
import os
import time

import numpy as np

from .native import alias, const, dtype, names

TABLE = dtype('struct route_tel_file')
NAMES = names('telemetry.h', 'route_tel_names')
GAUGE, PHASE = alias('telemetry.h', 'RT_FIRST_GAUGE'), alias('telemetry.h', 'RT_FIRST_PHASE')
MAGIC = const('ROUTE_TEL_MAGIC')


def now():
    return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)   # telemetry.c's clock


class Table:
    def __init__(self, path):
        import _posixshmem   # the table is a POSIX shared memory object (telemetry.c: shm_open), never a file
        fd = _posixshmem.shm_open(path, os.O_RDONLY, 0)
        try:
            self.map = mmap.mmap(fd, TABLE.itemsize, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        self.t = np.frombuffer(self.map, TABLE, 1)[0]
        self.last = None

    def sample(self):
        """the record since the last sample (None for the first, or before the trainer has mapped the table)"""
        if int(self.t['magic']) != MAGIC:
            return None
        v, at = self.t['v'].astype(np.int64).copy(), now()
        rec = None
        if self.last is not None:
            dt = (at - self.last[1]) / 1e9; start = v[NAMES.index('phase_start')]
            rec = {'record': 'telemetry', 'rank': int(self.t['rank']), 't': round((at - start) / 1e9, 2) if start else 0.0}
            rec.update({NAMES[i]: float(v[i] - self.last[0][i]) / dt for i in range(GAUGE)})
            rec.update({NAMES[i]: int(v[i]) for i in range(GAUGE, PHASE)})
            rec.update({NAMES[i]: round((v[i] - start) / 1e9, 3) for i in range(PHASE + 1, len(NAMES)) if v[i]})
            rec['totals'] = {k: int(v[NAMES.index(k)]) for k in ('steps', 'optimizer_steps', 'transactions')}
        self.last = (v, at)
        return rec
