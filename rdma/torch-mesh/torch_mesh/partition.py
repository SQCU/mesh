"""A partition operand (design/heterogeneity.md §4): a mesh dimension's extents split among its ranks by parts that
vary between calls, a program's own operand on raw tensors.

A mesh dimension carries a capacity c_r >= 1 per coordinate and parts p_r (1 <= p_r <= c_r) of P = sum(p) units,
attached with the mesh: attach(mesh, tp=((15, 15), (4, 12))), the same on every rank.  The parts are a tensor operand
(operand(mesh, "tp").parts), written between calls by write(mesh, tp=(5, 11)).  An extent N that P divides (any other
raises ValueError, on every rank alike) is split N p_r / P per coordinate in rank order, as the logical tensor reads:
split(mesh, "tp", N), the blocks the list collectives take (dist.all_gather into a list of unequal outputs,
dist.reduce_scatter of a list of unequal inputs: MPI's v-collectives in torch's spelling, the backend's Allgatherv and
Reduce_scatter).  A rank that holds its share in a buffer of its capacity, c_r N / P, its valid elements first, keeps
its shapes while the parts move within the capacities and masks its own padding (R14).

rebalance(mesh, name, every) moves the parts by the evidence of the program's steps (allocate.Balancer)."""
import torch
import torch.distributed as dist

_OPERANDS = {}
_BALANCING = {}


class _Operand:
    """A partitioned dimension's capacities (fixed) and parts (a CPU tensor, written in place between calls)."""

    def __init__(self, capacity, parts):
        self.capacity, self.units = tuple(int(c) for c in capacity), sum(int(p) for p in parts)
        self.parts = torch.zeros(len(self.capacity), dtype=torch.int64)
        self.write(parts)

    def write(self, parts):
        parts = tuple(int(p) for p in parts)
        if len(parts) != len(self.capacity) or sum(parts) != self.units or not all(1 <= p <= c for p, c in zip(parts, self.capacity)):
            raise ValueError(f'mesh: parts {parts} of {self.units} units within the capacities {self.capacity}: one a '
                             f'coordinate, each at least 1')
        self.parts.copy_(torch.tensor(parts))


def attach(mesh, **dims):
    """Partition mesh dimensions by name: attach(mesh, tp=(capacity, parts)), a capacity and a part a coordinate."""
    for name, (capacity, parts) in dims.items():
        if name not in (mesh.mesh_dim_names or ()) or len(capacity) != mesh.size(mesh.mesh_dim_names.index(name)):
            raise ValueError(f'mesh: partition {name}: {len(capacity)} capacities, one a coordinate of a mesh dimension of '
                             f'{mesh.mesh_dim_names}')
        _OPERANDS[mesh, name] = _Operand(capacity, parts)


def operand(mesh, name):
    """The partition of mesh dimension `name`: .capacity, .units, .parts (the tensor)."""
    return _OPERANDS[mesh, name]


def write(mesh, **dims):
    """New parts for partitioned dimensions, between calls: write(mesh, tp=(5, 11))."""
    for name, parts in dims.items():
        operand(mesh, name).write(parts)


def split(mesh, name, N):
    """The blocks of an extent N along partitioned mesh dimension `name` under the current parts, in rank order."""
    o = operand(mesh, name)
    if N % o.units:
        raise ValueError(f'mesh: an extent of {N} split over {o.units} units is no multiple of them')
    return [N // o.units * p for p in o.parts.tolist()]


def rebalance(mesh, name='tp', every=8):
    """The parts of partitioned mesh dimension `name` that the evidence of its last `every` steps gives
    (design/heterogeneity.md R3, R5, R10, R11): call it once a step, after the step is issued, on every rank of the
    mesh alike.  The first call watches the dimension's group (torch_mesh.evidence: each rank's own work between its
    collectives there, one stretch a collective); every `every` steps after it the ranks' stretches are gathered to
    every rank and one step of the mesh repo's rdma/allocate.py Balancer (the same on every rank: R4) gives the parts,
    each at least 1, at most the units less one a peer, minimising the steps' time: the sum over their stretches of
    the slowest rank's, every stretch scaling with this dimension's parts.

    A rank that computes its whole capacity buffer does work that follows its capacity: the evidence measures parts
    where they are the capacities, and parts that move within capacities set only what is valid.  Returns (parts,
    written): written where the parts moved and fit the capacities (then written here); parts that moved past a
    capacity are the program's to attach again (capacities and parts both: its buffers' shapes change, the one cost
    R11 weighs), whose dimension's group keeps this balancer.  The parts stand once the evidence promises no gain past
    its resolution."""
    import allocate
    from . import evidence
    o = operand(mesh, name)
    group = mesh.get_group(name)
    if mesh.device_type == 'mps' and not getattr(group, 'handle', 0):
        raise ValueError(f'mesh: rebalance {name}: its group is not the mesh backend\'s (no communicator to watch)')
    where = (id(group), name)
    held = _BALANCING.get(where)
    n = len(o.capacity)
    if held is None:
        _BALANCING[where] = {'balancer': allocate.Balancer({name: {'parts': o.parts.tolist(), 'low': [1] * n,
                                                                   'high': [o.units - n + 1] * n}}), 'steps': 0}
        evidence.watch(group, mesh.device_type)
        return tuple(o.parts.tolist()), False
    evidence.step(group)
    held['steps'] += 1
    if held['steps'] < every:
        return tuple(held['balancer'].parts[name]), False
    held['steps'] = 0
    everyone = [None] * dist.get_world_size(group)
    dist.all_gather_object(everyone, evidence.work(group), group=group)
    held['work'] = [sorted(sum(step) for step in steps)[len(steps) // 2] if steps else None for steps in everyone]
    positions = max((len(step) for steps in everyone if steps for step in steps), default=0)
    stretches = {f'step:{k}': {'scales': {name: [1.0] * n},
                               'times': [[step[k] for step in steps if len(step) > k] if steps else None for steps in everyone]}
                 for k in range(positions)}
    held['balancer'].decisions[name]['parts'] = o.parts.tolist()
    parts = held['balancer'].observe(stretches)[name]
    written = parts != o.parts.tolist() and all(p <= c for p, c in zip(parts, o.capacity))
    if written:
        o.write(parts)
    return tuple(parts), written


def balancing(mesh, name='tp'):
    """What rebalance last saw of partitioned dimension `name`: {'parts', 'stands', 'work': each rank's median
    step work (seconds, its stretches summed) in the last window}, None before its first window."""
    held = _BALANCING.get((id(mesh.get_group(name)), name))
    if not held or 'work' not in held:
        return None
    return {'parts': held['balancer'].parts[name], 'stands': held['balancer'].stands(name), 'work': held['work']}
