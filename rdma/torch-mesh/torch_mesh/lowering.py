"""design/page-maps-2026-11-02.md: the final lowering, the one step from values to storage.

Choosers hold TensorValues: layouts over virtual spaces, page identities, no storage. Here, once per occurrence, each
value becomes a TensorView over real storage: the run's arena lease where the page assignment packed it, the
caller's tensor where it is pinned or lease-backed. Downstream (page effects' native bind, the recorder) consumes
those views and holds pointers, as effectors may. No chooser imports this module, and no decision is made here: it
evaluates the assignment it is given.
"""
import torch

from .operand import TensorValue, TensorView

# the caller's tensor behind each lease-backed space, by its serial, and behind each state's pins: a storage handle,
# the lowering's to hold, never a chooser object's
_LEASES, _PINS = {}, {}


def register_lease(space, tensor):
    """type: a lease-backed ValueSpace and the caller's tensor whose lease it is -> None (kept for the lowering)."""
    _LEASES[space.serial] = tensor


def register_pins(state, tensor):
    """type: a state's identity and the caller's tensor its first and last versions are pinned to -> None."""
    _PINS[state] = tensor


def _view(storage, value, first_byte):
    """a TensorView of `storage` holding `value`'s elements from byte `first_byte` of its space"""
    width = value.dtype.itemsize
    if first_byte % width:
        raise ValueError('a value placed off its element width (design/page-maps-2026-11-02.md)')
    typed = storage.view(value.dtype) if storage.dtype != value.dtype else storage
    return TensorView(typed, value.layout.shift(first_byte // width))


def _whole(tensor):
    """a uint8 tensor over the whole storage of `tensor`, from its first byte: a bin's positions count from there"""
    storage = tensor.untyped_storage()
    return torch.empty((0,), dtype=torch.uint8, device=tensor.device).set_(storage, 0, (storage.nbytes(),))


def lower_value(value, invocation, assignment, arena):
    """type: (a TensorValue, its occurrence's run invocation, the run's page assignment, the run's arena tensor) -> a
    TensorView over real storage: its item's position reindexed through its bin, the arena or a state's lease. A
    TensorView passes unchanged."""
    if not isinstance(value, TensorValue):
        return value
    space = value.space
    if space.lease:
        return _view(_LEASES[space.serial].view(torch.uint8).reshape(-1), value, 0)
    key = assignment['items'].get((space.serial, invocation))
    if key is None:
        raise ValueError('a value with no page assignment at its invocation (design/page-maps-2026-11-02.md)')
    runs = assignment['places'][key]
    if len(runs) != 1:
        # a TensorView is one storage and an affine layout: the contiguous operand type. A value on several runs is
        # reached through its page table, the paged operand type of design/page-maps-2026-11-02.md, "Kernel addressing
        # from a value's pieces"; no command here takes it yet
        raise ValueError('a paged value reached a contiguous operand (design/page-maps-2026-11-02.md)')
    place, _ = runs[0]
    for first, pages, owner in assignment['bins']:
        if first <= place < first + pages:
            storage = arena if owner == 'arena' else _whole(_PINS[owner[1]])
            return _view(storage, value, (place - first) * space.page_bytes)
    raise ValueError('a value placed in no bin (design/page-maps-2026-11-02.md)')


def lower_programs(programs, assignment, invocation_of, page_bytes, device):
    """type: the run's programs (their commands' operands values or views), its page assignment -> the same programs,
    every operand a TensorView over storage at its occurrence; the arena leased once for the run."""
    from . import empty
    arena = empty((max(1, assignment['arena']) * page_bytes,), dtype=torch.uint8, device=device)
    for program in programs:
        invocation = invocation_of(program['order'])
        for command in program.get('commands', ()):
            if 'operands' in command:
                command['operands'] = [lower_value(operand, invocation, assignment, arena)
                                       for operand in command['operands']]
    return programs, arena
