"""Each rank's own work between its collectives on a process group, measured where those collectives run
(design/heterogeneity.md R2, R3, R11: a rank's time never holds its collectives; the evidence partition.rebalance
balances on).

While a group is watched (`watch`), every collective it runs is marked just before it is issued and just after: on
torch's MPS stream a timing event (torch.mps.Event: it records where the GPU reaches it, so the mark before a
collective is when the GPU finished the work before it, and the mark after is when the GPU has passed its wait), on
the host the monotonic clock (a host collective completes in the call).  A step's work on the rank is the sum, over
its collectives, of the time from the mark after the previous collective (the previous step's last, or the step's
boundary for the first) to the mark before this one: the same stretch the engine's ledgers time (metal-microbench
tools/mesh/rates.py).  `step` closes a step; `work` returns the closed steps' seconds and forgets them.  Two marks with
no GPU work between them cannot be timed apart (the event pool orders them as one); that interval counts zero.

Unwatched groups pay one dictionary lookup a collective; a watched group two marks."""
import time

_WATCHED = {}


class _Watch:
    def __init__(self, device):
        self.device, self.marks, self.steps, self.last = device, [], [], None

    def mark(self):
        if self.device == 'mps':
            import torch
            event = torch.mps.Event(enable_timing=True)
            event.record()
            return event
        return time.monotonic()

    def seconds(self, start, end):
        if self.device != 'mps':
            return end - start
        try:
            return max(0.0, start.elapsed_time(end) / 1e3)
        except RuntimeError:
            return 0.0


def watch(group, device):
    """Marks `group`'s collectives from now on (device 'mps' or 'cpu': where its tensors live); the first step
    begins here."""
    w = _WATCHED.setdefault(id(group), _Watch(device))
    if w.last is None:
        w.last = w.mark()


def unwatch(group):
    _WATCHED.pop(id(group), None)


def mark(group, before):
    w = _WATCHED.get(id(group))
    if w is None:
        return
    w.marks.append((before, w.mark()))


def step(group):
    """Closes the group's current step (its marks so far)."""
    w = _WATCHED.get(id(group))
    if w is not None:
        w.steps.append(w.marks)
        w.marks = []
        boundary = w.mark()
        w.steps[-1].append((None, boundary))


def work(group):
    """The rank's work in each closed step of `group` (seconds), the steps then forgotten."""
    w = _WATCHED.get(id(group))
    if w is None:
        return []
    if w.device == 'mps':
        import torch
        torch.mps.synchronize()
    out = []
    for marks in w.steps:
        total, last = 0.0, w.last
        for before, m in marks:
            if before:
                total += w.seconds(last, m)
            last = m
        w.last = last
        out.append(total)
    w.steps = []
    return out


def measured(method, issues=False):
    """`method` of a process group, marked while the group is watched; a call inside a coalesced group is marked
    where the coalesced group is issued (`issues`: end_coalescing)."""
    def call(self, *args, **kwargs):
        if id(self) not in _WATCHED or (not issues and getattr(self, '_pending', None) is not None):
            return method(self, *args, **kwargs)
        mark(self, True)
        try:
            return method(self, *args, **kwargs)
        finally:
            mark(self, False)
    call.__name__, call.__doc__, call.__wrapped__ = method.__name__, method.__doc__, method
    return call
