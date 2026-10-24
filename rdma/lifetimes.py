"""Compile page ownership from the existing data-dependency graph.

No edge, message, copy, or wait is inserted. A returned data publication may
prove a remote final read. Without that proof the allocation stays live. The
bridge may instead retire a source after its *local final DMA read*: that is a
different obligation from consuming the destination and is not a network ACK.

Events and page ranges are invocation/version specific. Channel names include
the connection generation; prefixes mean contiguous completed/supplied bytes,
never merely the largest out-of-order completion. See design/whole-function.md.
"""
from dataclasses import dataclass
import heapq


@dataclass(frozen=True)
class Event:
    name: str
    node: int
    kind: str                         # write_begin, publish, land, read_done, source_done
    after: tuple = ()                 # existing local data dependencies only
    channel: str | None = None
    prefix: int | None = None


@dataclass(frozen=True)
class Lease:
    name: str
    node: int
    pages: int
    born: str                         # publication/landing creating this value
    start: str | None = None           # first write; absent means no alias into it


@dataclass(frozen=True)
class Read:
    lease: str
    done: str                         # final read, or its dependent publication


@dataclass(frozen=True)
class Transfer:
    source: str
    publish: str
    land: str
    source_done: str | None = None     # bridge-owned final DMA read of this range
    begin: str | None = None           # first destination write, not final landing


@dataclass(frozen=True)
class Allocation:
    node: int
    page: int
    pages: int


@dataclass(frozen=True)
class Retirement:
    lease: str
    node: int
    required: int
    evidence: tuple                   # (local event name, obligations it proves)
    unproved: tuple                   # obligations with no local natural proof


@dataclass(frozen=True)
class GatherCertificate:
    participants: tuple
    paths: int
    slots: int
    proofs: tuple                     # (old lifetime obligation, future first write)


class Tracker:
    """Allocator-side observation of existing events, never a data-path wait.

    Compile once; each observed event performs only the affected lease updates.
    Calling observe asserts the actual event happened. It does not poll, read
    driver state, send a receipt, or itself establish a device memory fence.
    """

    def __init__(self, rules):
        self._rules = {rule.lease: rule for rule in rules}
        self._seen = {rule.lease: 0 for rule in rules}
        self._by_event = {}
        self._released = set()
        for rule in rules:
            for event, mask in rule.evidence:
                self._by_event.setdefault(event, []).append((rule.lease, mask))

    def observe(self, event):
        released = []
        for lease, mask in self._by_event.get(event, ()):
            if lease in self._released:
                continue
            self._seen[lease] |= mask
            if self._seen[lease] == self._rules[lease].required:
                self._released.add(lease)
                released.append(lease)
        return tuple(released)

    @property
    def retained(self):
        return tuple(name for name in self._rules if name not in self._released)


class LifetimePlan:
    def __init__(self, events, leases, ancestors, obligations, allocations):
        self.events, self.leases = events, leases
        self.ancestors, self.obligations = ancestors, obligations
        self.allocations = allocations

    def precedes(self, before, after):
        return before in self.ancestors[after]

    def rules(self, node, observations=None, *, bridge=False):
        """Local natural proofs; source_done is visible only to bridge ownership.

        Each reader is an AND obligation. A transfer's final DMA read has two
        possible witnesses: local source_done OR a causally downstream landing.
        Parallel fragments remain separate obligations, even to the same peer.
        """
        allowed = set(self.events if observations is None else observations)
        unknown = allowed - self.events.keys()
        if unknown:
            raise ValueError(f'unknown observations: {sorted(unknown)}')
        observed = [event for event in self.events.values() if event.name in allowed
                    and event.node == node and (bridge or event.kind != 'source_done')]
        rules = []
        for lease in self.leases.values():
            if lease.node != node:
                continue
            requirements = self.obligations[lease.name]
            evidence = []
            covered = 0
            for event in observed:
                mask = sum(1 << i for i, (_, witnesses) in enumerate(requirements)
                           if any(w in self.ancestors[event.name] for w in witnesses))
                if mask:
                    evidence.append((event.name, mask))
                    covered |= mask
            rules.append(Retirement(lease.name, node, (1 << len(requirements)) - 1,
                                    tuple(evidence), tuple(label for i, (label, _) in enumerate(requirements)
                                                           if not covered & (1 << i))))
        return tuple(rules)

    def tracker(self, node, observations=None, *, bridge=False):
        return Tracker(self.rules(node, observations, bridge=bridge))


def compile_lifetimes(events, leases, reads=(), transfers=()):
    """Validate a typed DAG and allocate compatible page intervals ahead of time.

    First-fit aliasing is legal only where the *existing* graph already orders
    every old read before the new writer. It is not a minimum-memory theorem.
    Split a large lease into page/range leases for finer retirement. Every local
    read and outgoing physical fragment must be declared, including forwards.
    The graph and its prefixes must cover the full presentation, including DMA
    reads not posted yet; an early SEND completion is not a final-read event.
    """
    event_list, lease_list = tuple(events), tuple(leases)
    events, leases = {e.name: e for e in event_list}, {p.name: p for p in lease_list}
    reads, transfers = tuple(reads), tuple(transfers)
    if len(events) != len(event_list) or len(leases) != len(lease_list):
        raise ValueError('event and lease names must be unique')
    parents = {name: set(event.after) for name, event in events.items()}
    channels = {}
    for event in events.values():
        if event.kind not in ('write_begin', 'publish', 'land', 'read_done', 'source_done'):
            raise ValueError(f'{event.name}: invalid event kind')
        if (event.channel is None) != (event.prefix is None):
            raise ValueError(f'{event.name}: channel and contiguous prefix are paired')
        if event.channel is not None:
            if event.kind not in ('publish', 'land', 'source_done') or event.prefix < 0:
                raise ValueError(f'{event.name}: invalid prefix evidence')
            channels.setdefault((event.node, event.kind, event.channel), []).append(event)
        for parent in event.after:
            if parent not in events or events[parent].node != event.node:
                raise ValueError(f'{event.name}: only a Transfer may introduce a remote dependency')
            if events[parent].kind == 'source_done' and event.kind != 'source_done':
                raise ValueError(f'{event.name}: data execution cannot wait on bridge source completion')
    for ordered in channels.values():
        ordered.sort(key=lambda e: e.prefix)
        for previous, event in zip(ordered, ordered[1:]):
            if previous.prefix == event.prefix:
                raise ValueError('one event names each channel prefix')
            parents[event.name].add(previous.name)
    for lease in leases.values():
        if lease.pages <= 0 or int(lease.pages) != lease.pages:
            raise ValueError(f'{lease.name}: positive integral page extent required')
        if lease.born not in events or events[lease.born].node != lease.node:
            raise ValueError(f'{lease.name}: birth must be a local event')
        if events[lease.born].kind not in ('publish', 'land'):
            raise ValueError(f'{lease.name}: birth must write the value')
    for transfer in transfers:
        if transfer.source not in leases or transfer.publish not in events or transfer.land not in events:
            raise ValueError('transfer names an unknown lease/event')
        source = leases[transfer.source]
        publish, land = events[transfer.publish], events[transfer.land]
        if publish.kind != 'publish' or land.kind != 'land' or publish.node != source.node or land.node == source.node:
            raise ValueError('transfer must publish source data into a remote landing')
        parents[land.name].add(publish.name)
        if transfer.begin is not None:
            begin = events.get(transfer.begin)
            if begin is None or begin.node != land.node or begin.kind != 'write_begin':
                raise ValueError('first destination write belongs to the landing node')
            parents[begin.name].add(publish.name)
            parents[land.name].add(begin.name)
        if transfer.source_done is not None:
            done = events.get(transfer.source_done)
            if done is None or done.kind != 'source_done' or done.node != source.node:
                raise ValueError('source completion belongs to the source bridge')
            parents[done.name].add(publish.name)

    children = {name: [] for name in events}
    remaining = {name: len(deps) for name, deps in parents.items()}
    for name, deps in parents.items():
        for parent in deps:
            children[parent].append(name)
    ready = [name for name, n in remaining.items() if not n]
    heapq.heapify(ready)
    ancestors, order = {}, []
    while ready:
        name = heapq.heappop(ready)
        ancestors[name] = {name}.union(*(ancestors[p] for p in parents[name]))
        order.append(name)
        for child in children[name]:
            remaining[child] -= 1
            if not remaining[child]:
                heapq.heappush(ready, child)
    if len(order) != len(events):
        raise ValueError('data dependencies and channel prefixes form a cycle')

    for lease in leases.values():
        if lease.start is not None and (
            lease.start not in events or events[lease.start].kind != 'write_begin'
            or events[lease.start].node != lease.node or lease.start not in ancestors[lease.born]
        ):
            raise ValueError(f'{lease.name}: first write must precede its local publication')

    obligations = {p.name: [('born', (p.born,))] for p in leases.values()}
    for read in reads:
        if read.lease not in leases or read.done not in events:
            raise ValueError('read names an unknown lease/event')
        lease, done = leases[read.lease], events[read.done]
        if done.node != lease.node or done.kind not in ('read_done', 'publish') or lease.born not in ancestors[done.name]:
            raise ValueError('read completion must follow the value on its own node')
        obligations[lease.name].append((f'read:{done.name}', (done.name,)))
    for transfer in transfers:
        source = leases[transfer.source]
        if source.born not in ancestors[transfer.publish]:
            raise ValueError('transfer publication must depend on its source value')
        witnesses = (transfer.land,) + ((transfer.source_done,) if transfer.source_done else ())
        obligations[source.name].append((f'transfer:{transfer.land}', witnesses))

    # Existing reachability is sufficient; never create a reuse dependency to
    # force two otherwise concurrent values onto the same storage.
    allocations, slots, high = {}, {}, {}
    ordinal = {name: i for i, name in enumerate(order)}
    for lease in sorted(leases.values(), key=lambda p: (ordinal[p.start or p.born], p.name)):
        available = slots.setdefault(lease.node, [])
        slot = None
        for candidate in available:
            previous = leases[candidate['last']]
            if lease.start is not None and candidate['pages'] >= lease.pages and all(
                any(w in ancestors[lease.start] for w in witnesses)
                for _, witnesses in obligations[previous.name]
            ):
                slot = candidate
                break
        if slot is None:
            slot = {'page': high.get(lease.node, 0), 'pages': lease.pages}
            high[lease.node] = slot['page'] + slot['pages']
            available.append(slot)
        slot['last'] = lease.name
        allocations[lease.name] = Allocation(lease.node, slot['page'], lease.pages)
    return LifetimePlan(events, leases, ancestors, obligations, allocations)


def prove_recurrent_gather(participants, paths=None):
    """Prove two-slot source, landing and relay reuse at realization.

    Every participant's next publication must follow its previous consumption
    of *all* participants' pieces. The executor must realize those data edges.
    This is not a certificate for arbitrary send/recv or asymmetric migration.
    ``paths[(source, destination)]`` lists node paths, one per physical fragment;
    parallel cables can appear as distinct paths with the same node sequence.
    Absent paths means direct paths. No proof is cached across topology changes.
    """
    nodes = tuple(range(participants)) if isinstance(participants, int) else tuple(participants)
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError('gather participants must be nonempty and unique')
    pairs = {(a, b) for a in nodes for b in nodes if a != b}
    paths = ({pair: (pair,) for pair in pairs} if paths is None else
             {pair: tuple(tuple(p) for p in value) for pair, value in paths.items()})
    if set(paths) != pairs:
        raise ValueError('gather needs the actual paths for every source/destination pair')
    for pair, routes in paths.items():
        if not routes or any(len(p) < 2 or (p[0], p[-1]) != pair or len(set(p)) != len(p) for p in routes):
            raise ValueError('gather paths must be nonempty simple paths with matching endpoints')

    events, leases, reads, transfers, starts = [], [], [], [], {}
    arrivals = {(node, i): [] for node in nodes for i in range(3)}
    for i in range(3):
        for node in nodes:
            start, pub = f'start:{node}:{i}', f'publish:{node}:{i}'
            deps = (f'consume:{node}:{i - 1}',) if i else ()
            events.extend((Event(start, node, 'write_begin', deps), Event(pub, node, 'publish', (start,))))
            lease = f'source:{node}:{i}'
            leases.append(Lease(lease, node, 1, pub, start))
            reads.append(Read(lease, f'consume:{node}:{i}'))
            starts[f'source:{node}', i] = start
        for pair, routes in sorted(paths.items()):
            for fragment, path in enumerate(routes):
                source, pub = f'source:{pair[0]}:{i}', f'publish:{pair[0]}:{i}'
                for hop, target in enumerate(path[1:]):
                    stem = f'path:{pair[0]}:{pair[1]}:{fragment}:{hop}'
                    lease, start, land = f'{stem}:{i}', f'begin:{stem}:{i}', f'land:{stem}:{i}'
                    events.extend((Event(start, target, 'write_begin'), Event(land, target, 'land')))
                    leases.append(Lease(lease, target, 1, land, start))
                    transfers.append(Transfer(source, pub, land, begin=start))
                    starts[stem, i] = start
                    if hop == len(path) - 2:
                        arrivals[target, i].append(land)
                        reads.append(Read(lease, f'consume:{target}:{i}'))
                    else:
                        pub = f'forward:{stem}:{i}'
                        events.append(Event(pub, target, 'publish', (land,)))
                        source = lease
        for node in nodes:
            events.append(Event(f'consume:{node}:{i}', node, 'read_done',
                                tuple([f'publish:{node}:{i}'] + arrivals[node, i])))

    plan = compile_lifetimes(events, leases, reads, transfers)
    proofs = []
    for stem, i in starts:
        if i:
            continue
        lease, future = f'{stem}:0', starts[stem, 2]
        for label, witnesses in plan.obligations[lease]:
            if not any(plan.precedes(witness, future) for witness in witnesses):
                raise ValueError(f'two-slot reuse has no proof: {lease}/{label} before {future}')
            proofs.append((f'{lease}/{label}', future))
    return GatherCertificate(nodes, sum(map(len, paths.values())), 2, tuple(proofs))
