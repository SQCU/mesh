"""Each rank's own work between its collectives on a process group, measured where those collectives run
(design/heterogeneity.md R2, R3, R11: a rank's time never holds its collectives; the evidence partition.rebalance
balances on).

While a group is watched (`watch`), every collective it runs is marked as it is issued and once it has completed. On
MPS tensors the marks are timing events of torch's pool recorded on its stream by the backend's C++ path
(torch_mesh/_stream.mm: DTensor's functional collectives and the group's direct calls), so the mark at issue is
where the GPU finished the work before the collective and the mark at completion where the GPU has passed its wait;
on host tensors the monotonic clock around the group's call (a host collective completes in the call).  A step's
work on the rank is the sum, over its collectives, of the time from the mark after the previous collective (the
previous step's last, or the step's boundary for the first) to the mark at this one's issue: the stretch the
bridges' ledgers time for a prepared call (rdma/ledger.py).  `step` closes a step; `work`, taken at a step's
boundary, returns each closed step's stretches (seconds, one a collective in issue order) and forgets them.  Two marks with no GPU work between them cannot be
timed apart; that interval counts zero.  An MPS collective issued through a coalesced group (start_coalescing) is
not marked.

An unwatched program pays nothing in Python and one empty-map test a collective in C++; a watched group two marks a
collective."""
import time

_WATCHED = {}


class _Watch:
    def __init__(self, device, handle):
        self.device, self.handle, self.marks, self.steps, self.last = device, handle, [], [], None


def _stream():
    from . import _stream as s
    return s


_METHODS = ('allreduce', 'allreduce_coalesced', 'reduce', 'broadcast', 'all_gather_single', 'all_gather_single_coalesced',
            'allgather', 'reduce_scatter_single', 'reduce_scatter_single_coalesced', 'reduce_scatter', 'alltoall',
            'all_to_all_single', 'gather', 'scatter', 'send', 'recv')


def _install(cls):
    """The backend's host collectives marked (once, as a host group is first watched: an unwatched program's calls
    pass through nothing)."""
    if getattr(cls, '_evidence', False):
        return
    for name in _METHODS:
        setattr(cls, name, measured(getattr(cls, name)))
    cls.end_coalescing = measured(cls.end_coalescing, issues=True)
    cls.allgather_into_tensor_coalesced = cls.all_gather_single_coalesced
    cls._evidence = True


def watch(group, device):
    """Marks `group`'s collectives from now on (device 'mps' or 'cpu': where its tensors live); the first step
    begins here."""
    if id(group) in _WATCHED:
        return
    w = _WATCHED[id(group)] = _Watch(device, getattr(group, 'handle', 0))
    if device == 'mps':
        _stream().watch(w.handle)
        _stream().mark(w.handle, 2)
    else:
        _install(type(group))
        w.last = time.monotonic()


def unwatch(group):
    w = _WATCHED.pop(id(group), None)
    if w is not None and w.device == 'mps':
        _stream().unwatch(w.handle)


def mark(group, before):
    w = _WATCHED.get(id(group))
    if w is not None and w.device != 'mps':
        w.marks.append((0 if before else 1, time.monotonic()))


def step(group):
    """Closes the group's current step."""
    w = _WATCHED.get(id(group))
    if w is None:
        return
    if w.device == 'mps':
        _stream().mark(w.handle, 2)
    else:
        w.steps.append(w.marks + [(2, time.monotonic())])
        w.marks = []


def work(group):
    """The rank's stretches in each closed step of `group` ([seconds, one a collective in issue order] a step), the
    steps then forgotten."""
    w = _WATCHED.get(id(group))
    if w is None:
        return []
    if w.device != 'mps':
        out = []
        for marks in w.steps:
            each = []
            for kind, t in marks:
                if kind == 0:
                    each.append(t - w.last)
                w.last = t
            out.append(each)
        w.steps = []
        return out
    import torch
    torch.mps.synchronize()
    out, each, done = [], [], []
    for kind, event in _stream().take(w.handle):
        if w.last is None:
            w.last = event
            continue
        if kind == 0:
            try:
                each.append(max(0.0, torch._C._mps_elapsedTimeOfEvents(w.last, event) / 1e3))
            except RuntimeError:
                each.append(0.0)
        done.append(w.last)
        w.last = event
        if kind == 2:
            out.append(each)
            each = []
    for event in done:
        torch._C._mps_releaseEvent(event)
    return out


def measured(method, issues=False):
    """`method` of a process group, marked on the host while the group is watched on host tensors (the MPS path is
    marked in _stream.mm); a call inside a coalesced group is marked where the coalesced group is issued (`issues`:
    end_coalescing)."""
    def call(self, *args, **kwargs):
        w = _WATCHED.get(id(self))
        if w is None or w.device == 'mps' or (not issues and getattr(self, '_pending', None) is not None):
            return method(self, *args, **kwargs)
        mark(self, True)
        try:
            return method(self, *args, **kwargs)
        finally:
            mark(self, False)
    call.__name__, call.__doc__, call.__wrapped__ = method.__name__, method.__doc__, method
    return call
