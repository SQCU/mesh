"""Each rank's own work between its collectives on a process group, measured where those collectives run
(design/heterogeneity.md R2, R3, R11: a rank's time never holds its collectives; the evidence partition.rebalance
balances on).

While a group is watched (`watch`), every collective it runs is marked as it is issued and once it has completed, by
`mark` alone: the backend's group (ProcessGroupMesh._group, every coalesced group among them) takes it around its call,
and the C++ path (torch_mesh/_stream.mm: DTensor's functional collectives and the group's direct calls) takes the same
mark on the same stream.  On MPS tensors a mark is a timing event of torch's pool recorded on its stream, so the mark
at issue is where the GPU finished the work before the collective and the mark at completion where the GPU has passed
its wait; on host tensors the monotonic clock (a host collective completes in the call).  A step's work on the rank is
the sum, over its collectives, of the time from the mark after the previous collective (the previous step's last, or
the step's boundary for the first) to the mark at this one's issue: the stretch the ranks' ledgers time for a prepared
call (rdma/ledger.py).  `step` closes a step; `work`, taken at a step's boundary, returns each closed step's stretches
(seconds, one a collective in issue order) and forgets them.  Two marks with no GPU work between them cannot be timed
apart; that interval counts zero.

An unwatched program pays one dictionary lookup a group call in Python and one empty-map test a collective in C++;
a watched group two marks a collective."""
import time

_WATCHED = {}


class _Watch:
    def __init__(self, device, handle):
        self.device, self.handle, self.marks, self.steps, self.last = device, handle, [], [], None


def _stream():
    from . import _stream as s
    return s


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
        w.last = time.monotonic()


def mark(group, before):
    """The group's mark at a collective's issue (`before`) or completion."""
    w = _WATCHED.get(id(group))
    if w is None:
        return
    if w.device == 'mps':
        _stream().mark(w.handle, 0 if before else 1)
    else:
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
