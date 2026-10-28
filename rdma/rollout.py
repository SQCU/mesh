# design/whole-function.md#observations
import json
from concurrent.futures import Future
import queue
import threading
from mesh_statistics import now, completed, emit

_history = {}


def observations(key):
    return _history.setdefault(str(key), [])


def start(store, key, rank, ranks, history):
    outgoing, stopping = queue.SimpleQueue(), threading.Event()
    future = Future()
    def collect():
        try:
            future.set_result(_collect(store, key, rank, ranks, history, outgoing, stopping))
        except BaseException as error:
            future.set_exception(error)
    worker = threading.Thread(target=collect, name='mesh-observations', daemon=True)
    worker.start()
    return outgoing, stopping, worker, future


def close(observer):
    """The collection's records, or its error: it ends within its deadline of the last observation (_collect)."""
    outgoing, stopping, worker, future = observer
    stopping.set()
    emit([{'observations': future.result()}])


def _collect(store, key, rank, ranks, history, outgoing, stopping):
    import mesh
    sent, received, pending = 0, [0] * ranks, {}
    waiting, idle = [], threading.Event()
    announced = False
    records = []
    # Once stopping, each observation (a completion here, a peer's record) within MESH_PLATFORM_NS of the last or of the
    # stop, else the collection ends naming what it waited for: a peer that died never reports.
    deadline, progressed, stopped = mesh.lib.MESH_PLATFORM_NS / 1e9, now(), None
    while True:
        while True:
            try:
                waiting.append(outgoing.get_nowait())
            except queue.Empty:
                break
        remaining = []
        from torch_mesh import _stream
        for completion, record in waiting:
            pending_at = now()
            if not _stream.completed(completion):
                record['last_pending'] = pending_at
                remaining.append((completion, record))
                continue
            observed = now()
            seconds = observed - record['started']
            record['seconds'] = seconds
            record['completion_seconds_bounds'] = [record.get('last_pending', record['started']) - record['started'], seconds]
            emit([completed(record, seconds)])
            progressed = observed
            if rank == 0:
                pending.setdefault(record['episode'], {})[0] = record
                received[0] += 1
            else:
                store.set(f'{key}/{rank}/{sent}', json.dumps(record))
            sent += 1
        waiting = remaining
        if stopping.is_set() and not waiting and not announced:
            if rank:
                store.set(f'{key}/finished/{rank}', str(sent))
            announced = True
        if rank == 0:
            for peer in range(1, ranks):
                address = f'{key}/{peer}/{received[peer]}'
                while store.check([address]):
                    record = json.loads(store.get(address))
                    pending.setdefault(record['episode'], {})[peer] = record
                    received[peer] += 1
                    address = f'{key}/{peer}/{received[peer]}'
                    progressed = now()
            for episode, peers in list(pending.items()):
                if len(peers) == ranks:
                    row = {**peers[0], 'seconds': max(p['seconds'] for p in peers.values()),
                           'rank_seconds': [peers[p]['seconds'] for p in range(ranks)],
                           'completion_seconds_bounds': [max(p['completion_seconds_bounds'][0] for p in peers.values()),
                                                         max(p['completion_seconds_bounds'][1] for p in peers.values())]}
                    records.append(row)
                    held = history if isinstance(history, list) else observations(history)
                    held.append(row)
                    from assignment import window
                    held[:] = window(held)
                    del pending[episode]
            finished = [f'{key}/finished/{peer}' for peer in range(1, ranks)]
            if announced and (not finished or store.check(finished)) and all(received[p] == int(store.get(k)) for p, k in enumerate(finished, 1)):
                break
        elif announced:
            break
        if stopping.is_set():
            stopped = stopped or now()
            if now() - max(progressed, stopped) > deadline:
                quiet = [p for p in range(1, ranks) if rank == 0 and not (store.check([f'{key}/finished/{p}']) and
                         received[p] == int(store.get(f'{key}/finished/{p}')))]
                raise TimeoutError(f'rollout {key}: {len(waiting)} completions here and the records of ranks {quiet}: '
                                   f'past their deadline of {deadline:.3f} s; the collection ends')
        idle.wait(.005)
    return records
