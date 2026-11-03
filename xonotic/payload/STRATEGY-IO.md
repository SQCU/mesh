# Strategy I/O

The current OBS width is 84. Slots 30–38 and 43–46 carry the twelve realized
Havocbot configuration traits and global skill; slot 83 is BOT_CONFIG_SCHEMA.
STATE type-tag bit 21 marks read-only skill words. Their input values are retained,
but their STRATEGY rate and residual must be zero. The native adapter enforces this
independently of the sender. See [the policy program](../../design/POLICY-PROGRAM.md).

The state-rate interface and its native page layout are documented in
[Policy rates over Havocbot's state views](../../design/POLICY-STATE-STEERING.md).
The operator's clarification in [SPECIFICATION §19](../../design/SPECIFICATION.md)
requires residuals on bot reads, with authoritative game state retained by the
engine. The old target/goal/spawn decoder has been deleted.

OBSERVATION, CART, TEAM and STATE join by engine session and snapshot tick; EVENT
is optional and does not block that join.
STATE adds typed 256-word pages of the bot's global and entity views. STRATEGY returns
residual/rate pages to the native view consumer. Actual bot execution acknowledges
the sequence; observation slots 39–42 expose receipt/source/application timing.
Slots 26–27 report view faults and their policy sequence; slot 28 counts coalesced
snapshot opportunities under transport backpressure. Slot 29 is literal native
player occupancy, populated for every client slot; absent slots are structural
padding. Slots 30–38 and 43–46 are
reserved. `strategy_io_schema.py` names the observation
columns; `state_steering.py` owns state-page packing and labels. Generic wire framing
is shared through `rdma/xonwire.def`.

STATE has 1,551 float32 columns: 15 headers and six 256-word arrays. The headers
include the original owner/schema/export-time/applied-sequence/address/generation/
session fields, followed by command source time, duration, relaxation time,
received sequence and page forcing. The arrays carry last-read source words,
current residuals, type/bit metadata, last-read times, held command velocities and
command-anchor residuals. A source word is a witnessed read, not a refreshed world
snapshot; its timestamp retains that distinction. STRATEGY remains 524 columns.

Response installation resolves each page against its current owner and entity
generation. A departed owner or replaced entity excludes only the affected pages
and reports their count; valid pages in the same response still apply. Replaced
pages are reset by the native generation lookup, so an old residual cannot reach
a new entity through a reused slot. Framing and numeric validation precede page
installation.

EVENT team zero denotes public, unowned context. The native topology producer has
neither a team nor a player observer; its complete rows reach every spatially
supported active observation destination. Positive team IDs share context within
that team. The same spatial and post-embedding age contraction applies to both.
Live waypoint CELL_LINK rows are retained: their coarse runtime graph can differ
from the separately supplied offline navigation realization.

EVENT has 18 columns. Column17 is the literal episode number at event deposit
time; the earlier 17 columns retain their order. Delayed events retain their
producer episode even when transport delivers them after a later round begins.

OUTCOME kind 7 repeatedly publishes five-column rows: engine session high/low,
episode, winner and engine time. These come from the engine's durable outcome
journal and survive loss of EVENT frames. The learner consumes episode identities
once; see [training continuation](../../design/TRAINING-CONTINUATION.md).
