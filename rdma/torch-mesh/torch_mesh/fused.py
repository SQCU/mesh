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


def writes(schedule, send, landings, pairs, group, polls=0, *, repeats=1, stride=0, ready=None, ready_value=0, tile_bytes=0):
    """Immutable routed writes, one landing lease per (source, destination).

    The producer releases every returned publication after making its operand.
    The consumer awaits every returned landing before reading its destinations.
    Bindings follow physical channels, not peers. Source and relay leases remain
    unchanged through the whole function's downstream uses. Previous users of
    the disjoint landing leases must finish before encoding this call; receives
    are posted immediately, without waiting for the local producer.

    ``repeats`` preposts a train of equal compiled frames once. Tensor arguments
    name the first frame inside backing storage covering all frames; ``stride``
    is their local byte displacement. Kernels release/await frame ``j``
    using the binding's repeat index. Publication must advance in frame order;
    it never certifies an unfinished earlier frame. A zero stride requires a
    separate lifetime proof before overwriting any prior frame's live bytes.

    ``ready`` and ``ready_value`` bind an independent producer word instead of
    the channel prefix. The producer calls mesh_source_release after this
    tile's stores. Distinct tiles own distinct words; no completion ordering
    between them is assumed. Relay readiness still follows incoming pages.
    With ``tile_bytes``, one WRITE family covers the operand. ``ready`` has one
    word per tile; ``land`` contains tile-major physical-channel bindings,
    padded with null records. Contributions to the same tile share each
    channel's last required prefix.
    """
    publish, land = schedule.writes(send, list(landings), list(pairs), group.handle,
                                    int(polls), int(repeats), int(stride), ready, int(ready_value), int(tile_bytes))
    return Crossing(tuple(map(tuple, publish)), tuple(map(tuple, land)))



def write_rows(schedule, send, landings, pairs, group):
    """One exact WRITE of a view (rows of it at a rank-local pitch, whatever its layout): the library lowers each rank's
    own view's layout, so only its bytes cross links, with no packing; publication and landing use repeat index zero."""
    publish, land = schedule.writes(send, list(landings), list(pairs), group.handle, 0, 1, 0, None, 0, 0)
    return Crossing(tuple(map(tuple, publish)), tuple(map(tuple, land)))

def _encode_words(constants, channels, land, publish):
    words = list(constants) + [0] * (len(constants) % 2)
    for crossing, bindings in ((land, 'land'), (publish, 'publish')):
        records = getattr(crossing, bindings) if crossing else ()
        if len(records) > channels:
            raise ValueError('kernel binding extent is smaller than its physical crossing')
        words += [w for k in range(channels) for w in _words(records[k] if k < len(records) else _NONE)]
    return words
