# design/measurement-regressions-2026-10-30.md#the-observer
"""A program's completion observations, on its own rank alone: each completion stamped as soon as it is seen, its own
interval from when it began or from the completion queued ahead of it, whichever is later (one queue runs one call at a
time, so a call submitted behind another starts when that one ends), and emitted as this rank's record. No store and
no peer: ranks' records are joined from their logs afterwards, so no control-plane round trip clocks a measurement."""
import queue
import threading
import time
from mesh_statistics import now, completed, emit

# the pause between looks at a pending completion: a stamp's resolution, far below any call this observes
PAUSE = 50e-6


def start():
    """An observer: put (completion, record) pairs, in submission order, with record['started'] the clock (now) as the
    call was submitted; close it once the program's calls are submitted."""
    outgoing, stopping = queue.SimpleQueue(), threading.Event()
    worker = threading.Thread(target=_observe, args=(outgoing, stopping), name='mesh-observations', daemon=True)
    worker.start()
    return outgoing, stopping, worker


def close(observer):
    """Every observation put before this emitted, then the observer ended."""
    outgoing, stopping, worker = observer
    stopping.set()
    worker.join()


def _observe(outgoing, stopping):
    from torch_mesh import _stream
    waiting, ended = [], 0.0
    while True:
        while True:
            try:
                waiting.append(outgoing.get_nowait())
            except queue.Empty:
                break
        if not waiting:
            if stopping.is_set():
                return
            time.sleep(PAUSE)
            continue
        completion, record = waiting[0]
        if not _stream.completed(completion):
            time.sleep(PAUSE)
            continue
        observed = now()
        began = max(record['started'], ended)
        ended = observed
        record['report'] = {**record['report'], 'queued_seconds': began - record['started']}
        emit([completed(record, observed - began)])
        waiting.pop(0)
