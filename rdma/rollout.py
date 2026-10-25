"""Asynchronous whole-function evidence, separate from payload progress.

Data-path methods only enqueue immutable records or drain a local inbox. The
worker owns control-store I/O and learning; no caller waits for another rank.
"""
import copy
import json
import queue
import threading
from pathlib import Path


class RolloutInbox:
    def __init__(self, ranks):
        self.ranks, self.pending, self.consumed = ranks, {}, set()

    def receive(self, rank, record):
        if not 0 <= rank < self.ranks:
            raise ValueError('rollout rank is outside its communicator')
        episode = record['episode']
        if episode in self.consumed:
            raise ValueError('duplicate completed episode')
        peers = self.pending.setdefault(episode, {})
        if rank in peers:
            raise ValueError('duplicate rank observation')
        if peers:
            previous = next(iter(peers.values()))
            for field in ('proposal', 'workload', 'behavior_probability_ticks', 'target_probability_ticks'):
                if previous.get(field) != record.get(field):
                    raise ValueError(f'episode ranks disagree about {field}')
        peers[rank] = copy.deepcopy(record)

    def drain(self):
        completed = []
        for episode in sorted(self.pending):
            peers = self.pending[episode]
            if len(peers) != self.ranks:
                continue
            seconds = [peers[rank]['seconds'] for rank in range(self.ranks)]
            completed.append({**peers[0], 'seconds': max(seconds), 'rank_seconds': seconds})
            self.consumed.add(episode)
        for record in completed:
            del self.pending[record['episode']]
        return completed


class RolloutStats:
    """In-process stats worker. No new daemon/service and no payload collective.

    Only rank zero fits the common model. Published states are proposals; this
    object cannot install routes or change a live generation's page bindings.
    Store operations, including readiness queries, run only on the worker.
    """
    def __init__(self, store, key, rank, ranks, controller, state_path=None):
        from controller import WholeFunctionController
        self.store, self.key, self.rank, self.ranks = store, key, rank, ranks
        self.controller = WholeFunctionController.of(controller.state()) if rank == 0 else None
        self.state_path = Path(state_path) if state_path is not None and rank == 0 else None
        self.outgoing, self.incoming = queue.SimpleQueue(), queue.SimpleQueue()
        self.stopped = threading.Event()
        self.wake = threading.Event()
        self.pending, self.submitted = 0, 0
        self.thread = threading.Thread(target=self._run, name='mesh-rollout-stats', daemon=True)
        self.thread.start()

    def submit(self, records):
        self.outgoing.put((copy.deepcopy(records), None))
        self.wake.set()

    def observe(self, record, timer):
        """Transfer observation ownership; even the final timer drains on the worker."""
        self.pending += 1
        self.outgoing.put(([copy.deepcopy(record)], timer))
        self.wake.set()

    def poll(self):
        records = []
        while True:
            try:
                record = self.incoming.get_nowait()
            except queue.Empty:
                return records
            if isinstance(record, Exception):
                raise RuntimeError('asynchronous rollout evidence failed') from record
            records.append(record)

    def close(self):
        """Drain statistics at process shutdown, after all numerical work has ended.

        No invocation calls this. Nonroots only publish their own final batch;
        the store owner remains alive to consume those publications and persist
        the model used by the next service invocation.
        """
        self.stopped.set()
        self.wake.set()
        self.thread.join()
        self.poll()

    def _run(self):
        sent, published, received, observed = 0, 0, [0] * self.ranks, 0
        inbox = RolloutInbox(self.ranks) if self.rank == 0 else None
        waiting, announced = [], False
        try:
            while True:
                while True:
                    try:
                        records, timer = self.outgoing.get_nowait()
                    except queue.Empty:
                        break
                    waiting.append((records, timer))
                remaining = []
                for records, timer in waiting:
                    if timer is not None:
                        if not timer.ready():
                            remaining.append((records, timer))
                            continue
                        start, end = timer.read()
                        records[0]['seconds'] = end - start
                        self.pending -= 1
                    self.store.set(f'{self.key}/rank/{self.rank}/{sent}', json.dumps(records))
                    sent += 1
                    self.submitted += len(records)
                waiting = remaining
                if self.stopped.is_set() and not waiting and not announced:
                    self.store.set(f'{self.key}/finished/{self.rank}', str(sent))
                    announced = True
                if self.rank == 0:
                    for rank in range(self.ranks):
                        key = f'{self.key}/rank/{rank}/{received[rank]}'
                        while self.store.check([key]):
                            for record in json.loads(self.store.get(key)):
                                inbox.receive(rank, record)
                            received[rank] += 1
                            key = f'{self.key}/rank/{rank}/{received[rank]}'
                    completed = inbox.drain()
                    if completed:
                        self.controller.completed_rollouts(completed)
                        observed += len(completed)
                        state = {'controller': self.controller.state(), 'observed': observed,
                                 'episodes': [row['episode'] for row in completed]}
                        self.store.set(f'{self.key}/state/{published}', json.dumps(state))
                        if self.state_path is not None:
                            self.state_path.parent.mkdir(parents=True, exist_ok=True)
                            temporary = self.state_path.with_suffix(self.state_path.suffix + '.tmp')
                            temporary.write_text(json.dumps(state['controller'], sort_keys=True) + '\n')
                            temporary.replace(self.state_path)
                        published += 1
                        self.incoming.put(state)
                    final_keys = [f'{self.key}/finished/{rank}' for rank in range(self.ranks)]
                    if announced and self.store.check(final_keys):
                        if all(received[r] == int(self.store.get(key)) for r, key in enumerate(final_keys)):
                            if inbox.pending:
                                raise ValueError('terminated rollouts have incomplete rank evidence')
                            break
                else:
                    if announced:
                        break
                    key = f'{self.key}/state/{published}'
                    while self.store.check([key]):
                        self.incoming.put(json.loads(self.store.get(key)))
                        published += 1
                        key = f'{self.key}/state/{published}'
                # A shutdown Event is already set during final draining; it must
                # not turn this observer into a busy loop.
                self.wake.wait(.005)
                self.wake.clear()
        except Exception as exc:
            self.incoming.put(exc)
