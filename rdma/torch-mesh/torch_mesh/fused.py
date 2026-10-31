"""Physical one-sided write bindings on the shared native Metal schedule.

All callers use the compiler's indexed source/destination writes. A kernel's
constant block contains its words padded to an even count, then physical
landing bindings and publication bindings. Their extent is the realized
channel count, independently of the number of ranks: parallel cables and
relay paths have the same representation for every solver.

Generic gather and sum lower to these same prepared writes, with exact rank
segments and separately owned relay ranges. They do not select a separate
collective transport or pad each contribution to a page.

The producer stores through mesh_step_write and releases each publication
once its compiled source range is ready. A consumer observes the landing
prefixes its data dependency names, then reads through mesh_step_read. An
independent ready dispatch can observe these prefixes once before parallel
readers on the serial MPS encoder. Bindings with no work are null records.

Receive destinations are fixed before execution and preposted independently
of the receiver's producer. Source, relay and destination leases are governed
by the compiled read/write dependency graph. A later unrelated collective or
an outgoing publication is not evidence of a remote read. Reuse requires an
actual descendant observation of every last reader, or deferred retirement by
the owner of the transfer. There is no peer-sized exchange interface.
"""
from collections import namedtuple

from . import _stream

# the checkout this package is installed from (its editable install: README "Install"), whose library it links
STEPS = _stream.metal_prelude() + '\n'
Crossing = namedtuple('Crossing', 'publish land')
_NONE = (0,) * 6


def steps():
    """The Metal text a kernel that runs a fused crossing's steps is compiled after (module docstring)."""
    return STEPS


def _words(addresses):
    return [w for a in addresses for w in (a & 0xFFFFFFFF, a >> 32)]


def declare(send, landings, pairs, group, polls=0, *, repeats=1, stride=0, ready=None, ready_value=0, tile_bytes=0):
    """Local typed views and producer records for the common communicator plan."""
    declaration = _stream.write_operands(send, list(landings), list(pairs), group.handle,
        int(polls), int(repeats), int(stride), ready, int(ready_value), int(tile_bytes))
    return declaration, calls(declaration.declarations())


def calls(declarations):
    import torch
    from .backend import Call, WRITE
    return [Call(WRITE, dtype=torch.uint8, **record) for record in declarations]


def writes(schedule, declaration):
    """Encode the next declared group and expose its compiled publication records."""
    publish, land = schedule.writes(declaration)
    return Crossing(tuple(map(tuple, publish)), tuple(map(tuple, land)))


def _encode_words(constants, channels, land, publish):
    words = list(constants) + [0] * (len(constants) % 2)
    for crossing, bindings in ((land, 'land'), (publish, 'publish')):
        records = getattr(crossing, bindings) if crossing else ()
        if len(records) > channels:
            raise ValueError('kernel binding extent is smaller than its physical crossing')
        words += [w for k in range(channels) for w in _words(records[k] if k < len(records) else _NONE)]
    return words
