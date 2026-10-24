"""Each rank's own work between its collectives on a process group, measured where those collectives run
(design/heterogeneity.md R2, R3, R11: a rank's time never holds its collectives; the evidence partition.rebalance
balances on).

While a group is watched (`watch`), every collective it runs is marked as it is issued and once it has completed, by
`mark` alone: every collective takes the one path (ProcessGroupMesh._issue into torch_mesh/_stream.mm issue, torch's
functional collectives and the library's own calls alike), which marks a host-path group around its call and a Metal-path
group on torch's stream.  On MPS tensors a mark is a timing event of torch's pool recorded on its stream, so the mark
at issue is where the GPU finished the work before the collective and the mark at completion where the GPU has passed
its wait, and an issued collective's (an async op's) also where its issue is encoded and where its wait begins; on host
tensors the monotonic clock (a host collective completes in the call).  A step's work on the rank is the sum of its
stretches, each from a mark where the GPU passed a collective (or the step's start) to the next where the rank's work
ends: the work before each collective, the work an issued one overlaps, and the step's tail.  `step` closes a step;
`work` returns the stretches (seconds in order, the last the tail) of each closed step the GPU has passed and forgets
them, waiting for nothing: on MPS a step whose closing mark the GPU has not reached stays for a later `work` (torch's
pool events complete in the stream's order, so the latest completed closing mark bounds a prefix of complete
marks).  Two marks with no GPU work between them cannot be timed apart; that interval counts zero.

An unwatched program pays one dictionary lookup a group call in Python and one empty-map test a collective in C++;
a watched group two marks a collective (an issued one four)."""
import time

_WATCHED = {}


class _Watch:
    def __init__(self, device, handle):
        self.device, self.handle, self.marks, self.steps, self.last, self.pending = device, handle, [], [], None, []


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
    """The rank's stretches in each closed step of `group` the GPU has passed ([seconds in order, the last the
    step's tail] a step), the steps then forgotten; nothing is waited for."""
    w = _WATCHED.get(id(group))
    if w is None:
        return []
    if w.device != 'mps':
        out = []
        for marks in w.steps:
            each = []
            for kind, t in marks:
                if kind != 1:
                    each.append(t - w.last)
                w.last = t
            out.append(each)
        w.steps = []
        return out
    import torch
    w.pending.extend(_stream().take(w.handle))
    closed = [k for k, (kind, event) in enumerate(w.pending) if kind == 2]
    passed = next((k + 1 for k in reversed(closed) if torch._C._mps_queryEvent(w.pending[k][1])), 0)
    ready, w.pending = w.pending[:passed], w.pending[passed:]
    out, each, done = [], [], []
    for kind, event in ready:
        if w.last is None:
            w.last = event
            continue
        if kind != 1:
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
