# design/whole-function.md#observations
import json
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
from pathlib import Path


def start(store, key, rank, ranks, path):
    outgoing, stopping = queue.SimpleQueue(), threading.Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mesh-observations')
    future = executor.submit(_collect, store, key, rank, ranks, Path(path), outgoing, stopping)
    return outgoing, stopping, executor, future


def close(observer):
    outgoing, stopping, executor, future = observer
    stopping.set()
    try:
        future.result()
    finally:
        executor.shutdown()


def _collect(store, key, rank, ranks, path, outgoing, stopping):
    sent, received, pending = 0, [0] * ranks, {}
    waiting, idle = [], threading.Event()
    announced = False
    while True:
        while True:
            try:
                waiting.append(outgoing.get_nowait())
            except queue.Empty:
                break
        remaining = []
        from torch_mesh import _stream
        for completion, record in waiting:
            observed = time.monotonic()
            if not _stream.completed(completion):
                record['last_pending'] = observed
                remaining.append((completion, record))
                continue
            seconds = observed - record['started']
            record['seconds'] = seconds
            record['completion_seconds_bounds'] = [record.get('last_pending', record['started']) - record['started'], seconds]
            from torch_mesh.benchmarks import completed, emit
            emit(completed(record, seconds))
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
            for episode, peers in list(pending.items()):
                if len(peers) == ranks:
                    row = {**peers[0], 'seconds': max(p['seconds'] for p in peers.values()),
                           'rank_seconds': [peers[p]['seconds'] for p in range(ranks)]}
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('a') as out:
                        out.write(json.dumps(row, sort_keys=True) + '\n')
                    del pending[episode]
            finished = [f'{key}/finished/{peer}' for peer in range(1, ranks)]
            if announced and (not finished or store.check(finished)) and all(received[p] == int(store.get(k)) for p, k in enumerate(finished, 1)):
                break
        elif announced:
            break
        idle.wait(.005)
