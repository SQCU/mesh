"""Asynchronous whole-function evidence, separate from payload progress.

Data-path methods only enqueue immutable records or drain a local inbox. The
worker owns control-store I/O and learning; no caller waits for another rank.
"""
import copy
import json
import queue
import threading


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
    def __init__(self, store, key, rank, ranks, controller):
        from controller import WholeFunctionController
        self.store, self.key, self.rank, self.ranks = store, key, rank, ranks
        self.controller = WholeFunctionController.of(controller.state()) if rank == 0 else None
        self.outgoing, self.incoming = queue.SimpleQueue(), queue.SimpleQueue()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name='mesh-rollout-stats', daemon=True)
        self.thread.start()

    def submit(self, records):
        self.outgoing.put(copy.deepcopy(records))

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
        self.stopped.set()

    def _run(self):
        sent, published, received, observed = 0, 0, [0] * self.ranks, 0
        inbox = RolloutInbox(self.ranks) if self.rank == 0 else None
        try:
            while not self.stopped.is_set():
                while True:
                    try:
                        records = self.outgoing.get_nowait()
                    except queue.Empty:
                        break
                    self.store.set(f'{self.key}/rank/{self.rank}/{sent}', json.dumps(records))
                    sent += 1
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
                        published += 1
                        self.incoming.put(state)
                else:
                    key = f'{self.key}/state/{published}'
                    while self.store.check([key]):
                        self.incoming.put(json.loads(self.store.get(key)))
                        published += 1
                        key = f'{self.key}/state/{published}'
                self.stopped.wait(.005)
        except Exception as exc:
            self.incoming.put(exc)
