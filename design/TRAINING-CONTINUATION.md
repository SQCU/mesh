# Training continuation and execution attribution

The policy emits full-state rates for each Havocbot view, with the integration
defined in [POLICY-STATE-STEERING.md](POLICY-STATE-STEERING.md). An emitted vector
becomes an actor example only when the engine reports its application sequence.
Critic intervals may use other policies' observed behavior. Sent, observed and
first applied are separate quantities in both viewers.

## Data ownership

| Owner | Durable state or operation |
| --- | --- |
| `mesh_ipc.c` | Complete native response staging; session identity; append-only engine `mesh-outcomes.tsv`; repeated OUTCOME frames |
| `xonwire.py::RuntimeFrames` | Session/tick joins for OBS, CART, TEAM and STATE; optional EVENT; duplicate watermarks and independent outcomes |
| `action_history.py::ActionHistory` | Issued frames, rates, behavior likelihoods and versions; accumulated returns; per-player application consumption |
| `online.py::OnlineLearner` | Model, optimizer, RNG, replay, current and suspended episode pools, idempotent outcome labels |
| `array_tree.py`, `checkpoint_state.py`, `journal.py` | Shared lossless array/string encoding, checkpoint schema, atomic replacement and checksummed write-ahead records |
| `strat_responder.py` | One transaction boundary joining these records; recovery before new publication |

The authoritative file is `<online-checkpoint>.runstate.npz`. It embeds every
trained learner plus runtime history, complete source frames, schema, RNG, cursors, episode identity
and journal position. Individual policy NPZ paths contain small relative-path
and learner-namespace references to that authoritative bundle, rather than
duplicated optimizer and replay arrays. All policy readers resolve the reference;
inference reads only its required arrays and continuation reads the complete
learner subtree. `--append-telemetry` restores the runtime bundle and loads each
learner once from its already loaded state. Standalone checkpoints remain readable.

Before publishing an action, the responder appends and fsyncs the complete step,
including its sampled vector and behavior likelihood. Recovery replays complete
post-checkpoint records through the learner using those exact samples. It does not
resend obsolete vectors or duplicate telemetry. Checksums and lengths identify an
incomplete last record; recovery reports and preserves its bytes separately before
repairing the tail. A completed checkpoint switches to a new empty journal, then
removes the old journal. An interrupted step cannot save a half-updated bundle.

This establishes process-interruption recovery. Power loss, disk failure and
filesystem durability beyond the completed fsync/rename operations have not been
fault-injected by the current validation.

## Delayed execution and terminal outcomes

Pending requests are independent of the telemetry row window. A request remains
until each observed participant has moved past its sequence or departed. When
memory exceeds the history budget, inactive request payloads spill into
`<online-checkpoint>.actions/`; attribution metadata stays resident. A delayed
acknowledgement restores the exact source frame, vector and likelihood. Referenced
spill files remain until a newer authoritative checkpoint no longer needs them.
Current and applied source records may exceed the soft memory budget; that amount
is reported. Disk use follows outstanding execution, rather than silently losing
training sources.

An episode is `(engine session high, engine session low, episode number)`. The
engine records its actual winner or tie in `mesh-outcomes.tsv` before repeatedly
publishing OUTCOME kind 7: five floats for that identity, winner and engine time.
Each publication includes the newest outcome and cycles older entries. Engine
startup reloads the journal. The learner labels each identity once, including a
suspended prior episode. Rounded scores and transient EVENT packets are not the
terminal authority. `outcome.json` acknowledges closure only after checkpointing.

A completed snapshot whose TEAM rows mark the episode finished joins its OUTCOME
before consumption. The pending window retains one such snapshot per episode and
consumes it before later snapshots. An early outcome cannot close the active
episode or label a preterminal transition; a late outcome releases the retained
finished snapshot. Attribution and optimization run before episode closure. The
idle path resolves only suspended prior episodes. Pending snapshots and outcomes
share the existing checkpoint representation.

Native receive fragments accumulate separately from the last completed response.
Session, request and tick identify a group; only completion swaps the buffers.
The engine coalesces snapshot opportunities while the preceding transport group
drains, retaining events and sampling current state next. Bot and game execution
continue, and OBS 28 reports the coalesced opportunities. This prevents an
ever-growing queue of obsolete state snapshots.

## Reporting and failures found by the actual match

`/policy` and `/j` expose optimizer updates, pending episodes, applied-source
matches, duplicate applications, unmatched applications, history bytes, journal
bytes and view faults. They retain measured results across producer restart.
State/rate covariance is stored exactly as `left.T @ right / mass` with centered
sample factors, avoiding quadratic storage in the number of state words. The
state-to-policy-representation covariance is explicit. Observed word addresses
label reports; unused padding is excluded from measurement.

The complete match exposed failures that isolated gradient checks missed: a
baseline member named `values` collided with the model container; clean game
directories never registered payload cvars; raw VM magnitudes overwhelmed fresh
readouts; invalid viewed references aborted the game; large covariance reports
blocked shutdown; partial frames overwrote completed responses; and telemetry
eviction destroyed still-pending actor sources on the slower delivery path.
The source fixes and [two-machine evidence](../measurements/training-continuation-20260906/README.md)
cover these reproductions. Uniform input conditioning and rate units preserve
unbounded full-state output; no replacement action vocabulary was introduced.

## Historical observations and remaining questions

The recorded matches used the actual native engine, full game code, bots and MLX
optimizer. Their isolated shared-memory/Unix-datagram transport opened no verbs
device. They exercised handled responder restart, abrupt process exit after a durable action,
recovery, a controlled engine tie, repeated terminal recovery and disk-outcome
replay after engine restart. The original fixture dropped EVENT frames to isolate
the durable outcome channel; the later fixture delivered actual events and
navigation, including deliberately late event delivery and an ownerless snapshot.
Those harnesses were removed at the operator's instruction. The current complete
policy flow is [POLICY-PROGRAM.md](POLICY-PROGRAM.md).

Those records describe the measured training plumbing, attribution and arithmetic. Learned
playing strength, sustained many-team throughput, worst-case bot cost, full native
side-effect containment and recovery during a real RDMA interruption require
separate evidence. The view fault handler unwinds VM-reported errors and retries
stock Havocbot; it does not undo previously issued game actions.

The subsequent [resident-bridge RDMA run](../measurements/rdma-training-20260906/README.md)
used four teams, eight bots and both 128-wide learners on the Mini, with the engine
on the laptop. It exposed a borrowed receive-buffer lifetime bug hidden by the
socket fixture. Direct iterator consumption repaired it. Both policies trained,
resumed and received the engine outcome; all 112 observed applications had exact
source attribution. The successful window preserved bridge PIDs and capacity with
no new bridge errors. An earlier bridge stall/replacement remains separately
documented; successful application recovery does not establish its cause.

## Full-row representation follow-up

The [September 6 representation repair](STATE-REDUCTION-RCA.md) retains all state
pages and native event rows through learned encoding. Large repeated coordinate
label lists exposed a checkpoint metadata overflow during its first RDMA attempt.
`checkpoint_state.py` now stores an interned string table, array-backed label
indices and UTF-8 metadata bytes, preserving labels and list/tuple identity without
expanding every repeated label in JSON. Legacy metadata remains readable;
incompatible model/optimizer versions start fresh with provenance. The successful
rerun and the original application failure are both retained in the
[representation evidence](../measurements/state-representation-20260906/README.md).

A later combined-update run exposed the same repeated-label expansion in the J
artifact's separate serializer. Checkpoints and reports now share `array_tree.py`.
The report's binary columnar viewer sidecar is paged into exact requested HTTP
coordinate slices; the full artifact retains all coordinates and numerical arrays.
Legacy artifacts remain readable. The reporting failure and rerun are retained in
the same evidence directory, independently of graph-reuse/allocation qualification.


## Raw observation history continuation

The [feature-integration follow-up](FEATURE-GRAM-PROVENANCE.md) preserves raw
EVENT history before learned embedding. Complete EVENT frames are not subject
to the snapshot queue's capacity limit or its latest-snapshot watermark. A
separate `(session,tick,request)` set deduplicates actual retransmissions while
allowing the first arrival of an older observation. Identical payloads in different
native frames remain distinct observations.

`ObservationMemory` appends literal rows and retains immutable prefix views for
past source frames. Episode changes reset this history. Ownerless snapshots add
events before output production resumes. Runstate saves history, coverage status,
pending complete event frames and their receipt identities. Each durable action
journal record contains only its new event rows and consumed frame identities;
replay consumes the corresponding pending frames without duplicating history.
Realized-event reports consume only fresh rows.

Legacy runstate without history recovers the latest available saved raw frame,
or an empty history when none exists, and explicitly reports incomplete coverage.
That status survives another save/restore; missing history is not invented.
The historical [native runtime report](../measurements/policy-feature-integration-20260906/native-runtime.json)
records event-prefix recovery, late delivery and ownerless input alongside
actual optimizer updates, bot application attribution and engine outcome recovery.
