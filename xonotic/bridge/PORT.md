# PORT.md — the Xonotic engine bridge on the sealed mesh

The bridge in `engine/mesh_shm.h` + `engine/mesh_ipc.c` talks a bespoke seqlocked
SHM protocol to a local `fakesolver`. That protocol is dead. The fabric already
carries pages between the two nodes and `rdma/mesh.h` + `rdma/mesh-client.c` is the
only interface to it. This document is the port: what the engine keeps, what the
wire looks like, how it builds, what the mini runs, and what happens when a page is
lost.

Node 0 is the MBP, running `darkplaces-dedicated` as the sole mesh client.
Node 1 is the mini, running the solver as its sole mesh client.
Neither side starts, stops, or restarts `mesh-flow`; both attach to a running pair.

## 0. What the sealed API gives us

```
void  *mesh_open(size_t *nslots, size_t *stride, size_t *usable);
size_t mesh_write(const void *p, size_t nbytes, int node);
size_t mesh_read(void **p, int *from);
```

Live values on this fabric (`rdma/mesh-stat`): `pgsz` 4096, `stride` 4096,
`usable` 4090 (`pgsz - sizeof(struct wire)`), `arena` 762492 slots.

Four properties the port is built on:

- `mesh_write` derives its start page from the pointer's offset into the arena base
  and chops the byte count into `usable`-sized descriptors, one per consecutive
  slot. A multi-slot payload must therefore be laid out at `stride`, not packed —
  `usable` bytes at `base + i*stride`, next chunk at `base + (i+1)*stride`.
- It returns the bytes it actually queued. A short return is a full submit ring, not
  an error, and there is no completion event to wait on.
- `mesh_read` returns one arrived slot; the pointer is valid only until the next
  `mesh_read`. Everything is copied out before the next call.
- The arena base is `mesh_data()`, i.e. page base + 6 bytes of `struct wire`. Slot
  payloads are **not** 4-byte aligned. Nothing casts a slot pointer to `float *`;
  every field crosses by `memcpy`. (`planner/plan.py` already lives with this via
  numpy views.)

There is no acknowledgement, no retransmit, and no ordering promise across slots.
That is the promise the failure posture in §5 is written against.

## 1. Builtins: six live, four die

Numbers stay where they are so QC declarations do not shuffle. The four dead slots
become `NULL` in both `vm_sv_builtins` and `vm_cl_builtins`.

| # | builtin | fate |
|---|---|---|
| 644 | `mesh_open` | **lives**, new signature |
| 645 | `mesh_close` | **dies** |
| 646 | `mesh_set` | **dies** |
| 647 | `mesh_get` | **dies** |
| 648 | `mesh_gather` | **lives**, column semantics |
| 649 | `mesh_scatter` | **lives**, column semantics |
| 650 | `mesh_publish` | **lives**, new signature |
| 651 | `mesh_poll` | **lives** |
| 652 | `mesh_wait` | **dies** |
| 653 | `mesh_stat` | **lives**, new selectors |

Why the four die:

- `mesh_close` unmapped a region the engine had created. The arena belongs to the
  running bridge; `mesh-client.c` has one process-global context and no detach. An
  engine that tears it down is tearing down someone else's fabric.
- `mesh_set` / `mesh_get` are per-element scalar accessors. `mesh_gather` and
  `mesh_scatter` already move a whole column of edict fields per call; a scalar path
  exists only to be used by accident on the hot path.
- `mesh_wait` spun until a sequence arrived, bounded by a deadline. On a transport
  with no delivery guarantee a single lost response makes every frame pay the whole
  deadline — 4 ms a frame, forever, for nothing. §5 replaces it with "no new plan
  this tick".

`mesh_gather` and `mesh_scatter` stay because they are the reason this is cheap:
a `.float` parameter arrives as a raw field offset and `prog->edictsfields` is a
flat array of stride `prog->entityfields`, so one call strides one feature across
every bot instead of `nbots` interpreted QC operations.

### The surviving QC interface

```
float(float node, float width, float maxrows)                  mesh_open    = #644;
void (float h, float col, .float fld, float first, float n)    mesh_gather  = #648;
void (float h, float col, .float fld, float first, float n)    mesh_scatter = #649;
float(float h, float tick, float nrows)                        mesh_publish = #650;
float(float h)                                                 mesh_poll    = #651;
float(float h, float sel)                                      mesh_stat    = #653;
```

- `mesh_open(node, width, maxrows)` attaches the process to the fabric and allocates
  the handle's scratch. Returns a handle `>= 0`, or `-1` when the bridge is not up.
  It is a **level-load call, never a frame call**: `mesh_attach` inside
  `mesh-client.c` retries for up to 30 s when the region is absent, and that time is
  charged to the caller. QC re-calls it on a slow timer when the previous call
  returned `-1`; a detached handle does not exist, so no frame ever pays for one.
- `mesh_gather(h, col, fld, first, n)` writes `req[row*width + col]` for
  `row in [0, n)` from edict `first + row`. `col < width`.
- `mesh_publish(h, tick, nrows)` frames and submits the request block. Returns the
  `req_id` it used, or 0 if the transport queued nothing.
- `mesh_poll(h)` drains arrived slots into the response reassembler and returns the
  highest `req_id` for which a complete response block is held. 0 means none ever.
- `mesh_scatter(h, col, fld, first, n)` writes edict `first + row`'s field from
  `resp[row*respwidth + col]` of the last complete block.
- `mesh_stat(h, sel)`, `sel & 15`:

| sel | value | sel | value |
|---|---|---|---|
| 0 | last published `req_id` | 8 | tx slots in the window |
| 1 | last complete `req_id` | 9 | short-write publishes |
| 2 | published − complete | 10 | rows in the last complete block |
| 3 | request width | 11 | response chunks received |
| 4 | maxrows | 12 | last complete tick |
| 5 | attached (1/0) | 13 | arena slots |
| 6 | dropped incomplete blocks | 14 | usable bytes per slot |
| 7 | request rows per slot | 15 | peer node |

### The per-tick shape

```
mesh_gather(h, 0, botid,   first, n);
mesh_gather(h, 1, health,  first, n);
...
mesh_publish(h, tick, n);
if (mesh_poll(h) > lastplan)
{
    lastplan = mesh_poll(h);
    mesh_scatter(h, 1, objective, first, n);
}
```

Nothing in that block can block, and the `if` is the whole failure handler.

## 2. Wire framing

Every slot is self-describing: 32-byte header, then rows. There is no shared
sequence state between the nodes and no header page. A slot that arrives out of
order, late, or alone carries everything needed to place or discard it.

```
offset size field
  0     4   magic     0x584d5348  'XMSH'
  4     2   version   1
  6     2   kind      1 = request, 2 = response
  8     4   req_id    monotonic, engine-assigned, never reused
 12     4   tick      server tick that produced the block, echoed in the response
 16     2   width     floats per row in this block
 18     2   rows      rows in this chunk
 20     2   chunk     0-based chunk index
 22     2   chunks    chunks in this block
 24     4   rows_total rows across the whole block
 28     4   flags     0
```

Little-endian, packed, copied field-by-field or as a 32-byte `memcpy` of a
`_Static_assert`ed struct. Payload follows immediately: `rows * width` float32,
row-major, no padding between rows.

Derived sizes at `usable = 4090`:

```
payload bytes   = usable - 32               = 4058
rows per slot   = 4058 / (width * 4)
request  width  16 -> 64 B/row  -> 63 rows/slot
response width   8 -> 32 B/row  -> 126 rows/slot
chunks          = ceil(rows_total / rows_per_slot)
```

480 bots is 8 request chunks and 4 response chunks per tick. The cap is
`MESH_XON_MAXCHUNK` 64 chunks, i.e. 4032 bots at width 16.

### Request row, width 16

`0` bot id, `1` team, `2` health, `3` armor, `4` ammo fraction, `5..7` origin/1024,
`8..10` velocity/1024, `11` distance to the payload, `12` payload progress,
`13` friends within radius, `14` enemies within radius, `15` current objective.

### Response row, width 8

`0` bot id (echoed), `1` chosen objective, `2..6` per-objective score, `7` epoch.

The engine has no opinion about any of it. Widths are the contract between
`qc/` and the solver; `mesh_ipc.c` only sees `width`.

### Multi-slot publish

The handle owns a contiguous TX window of `MESH_XON_TXSLOTS` = 4096 arena slots and
a cursor. `mesh_publish`:

1. `req_id++`; `chunks = ceil(nrows / rows_per_slot)`.
2. If `cursor + chunks > MESH_XON_TXSLOTS`, `cursor = 0`. A block is always
   contiguous, because `mesh_write` walks consecutive pages from one pointer.
3. For each chunk: `memcpy` the header to `arena + (cursor+c)*stride`, then the
   rows to `+32`.
4. One `mesh_write(arena + cursor*stride, chunks*usable, node)`.
5. `cursor += chunks`. If the return is short, the submit ring was full: count it in
   stat 9 and move on. The solver drops the incomplete block. No retry.

At 8 chunks per tick and 60 Hz, the window recycles every 512 ticks (8.5 s), which
is the reuse lag against in-flight pages.

### Reassembly

`mesh_poll` drains `mesh_read` to empty. Per slot: validate magic, version,
`kind == 2`, `width == respwidth`, `chunk < chunks <= MESH_XON_MAXCHUNK`, and that
the rows fit. Then:

- `req_id > inflight_id`: start a new reassembly — zero the 64-bit chunk mask, and
  if the previous one was incomplete, count it in stat 6.
- `req_id < inflight_id`: drop. Freshest wins; a straggler from an older block is
  never worth the frame it would take to finish.
- Set the chunk bit, `memcpy` the rows into `resp + chunk*rows_per_slot*respwidth`.
- Mask complete: `done_id = req_id`, `done_rows = rows_total`, `done_tick = tick`.

Every byte is copied out before the next `mesh_read`, per §0.

## 3. Building it into darkplaces

The shape from the existing patch is kept exactly: `prvm_cmds.c` ends with

```
#include "mesh_ipc.c"
```

and `mesh_ipc.c` pulls the sealed client into that same translation unit:

```
#include "mesh.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
#include "mesh-client.c"
#pragma GCC diagnostic pop
```

`apply.sh` copies `rdma/mesh.h`, `rdma/mesh-client.c` and `bridge/engine/mesh_ipc.c`
into the darkplaces tree and no longer copies `mesh_shm.h`, which is deleted with
`solver/fakesolver.c` and `solver/mesh_attach.h`.

Consequences, all checked against the tree in `xonotic/darkplaces-work`:

- **No makefile change.** No new object, no new link flag. `mesh-client.c` needs
  only `shm_open`/`mmap`, which darkplaces already links on this platform. Nothing
  in the engine links `-lrdma`; verbs stay inside `mesh-flow`.
- **No header collision.** `mesh.h` does not exist in the darkplaces tree
  (`meshqueue.h` does). A whole-word scan of every `.c`/`.h` for `FREE`, `RECV`,
  `SEND`, `APP`, `NOWN`, `SUB`, `CMP`, `REL`, `ACK`, `NRING`, `RINGS`, `hdr`,
  `desc`, `ring`, `wire`, `push`, `pop`, `slot` finds one hit — `slot`, in a comment
  and in a QC prototype string. Zero real collisions.
- **Warnings.** The strict set is live (`makefile:272`:
  `-Wall -Wold-style-definition -Wstrict-prototypes -Wsign-compare
  -Wdeclaration-after-statement -Wmissing-prototypes`). `mesh-client.c` mixes
  declarations and statements throughout; the pragma above is scoped to the include
  and nothing else is suppressed. Its non-static functions are all prototyped in
  `mesh.h`, so `-Wmissing-prototypes` and `-Wstrict-prototypes` stay quiet.
- **One attach per process.** `mesh_open`/`mesh_write`/`mesh_read` share the single
  static `CTX0`. All handles in the engine share one attachment and one arena; the
  handle table only partitions the TX window. In a listen server SVQC and CSQC share
  it too, and a handle number passed across VMs reaches the other VM's window.
  Documented, not guarded.
- **Region name.** `mesh_attach(NULL)` honours `$MESH_REGION`, then `/mesh0`. The
  builtin passes NULL; a second instance is separated by the environment, not by an
  engine argument.
- **One client per node.** Nothing else may be attached while the server runs. Check
  `rdma/mesh-stat`, field `client`, is 0 before launching.

## 4. The solver worker on the mini

`xonotic/solver/xon_solve.py`, node 1, one process, sole mesh client.

```
m = Mesh()
while True:
    for buf, src in m.read(dtype=np.uint8):
        place the chunk
    for each block completed this pass:
        pick, G = plan(X)
        frame the response and m.write(...) back to src
```

Contract:

- **Read.** `m.read(dtype=np.uint8)` yields one arrived slot as a byte view. Parse
  the 32-byte header, check magic/version/`kind == 1`, and copy the rows into a
  staging `(rows_total, width)` float32 array with
  `np.frombuffer(buf[32:32+rows*width*4].tobytes(), np.float32)`. The `.tobytes()`
  is the alignment answer from §0. Track a chunk mask per `req_id`; a `req_id` lower
  than the newest one seen is dropped on arrival.
- **Compute.** `planner/plan.py`'s `solve()` with `D = width`, `EXPERTS = 8`,
  `FF = 2048`, `TEAMS = 5`, `SEED = 20260828`: route each row to an expert by
  `argmax(X @ R)`, apply that expert grouped by expert (never gathered per row),
  score with `Y @ O`, `pick = argmax(G, axis=1)`.
- **Deterministic fallback**, used when mlx is unavailable, and still a function of
  every input feature: with the same seeded rng,
  `G = tanh(X @ A) @ O`, `A` of shape `(width, 64)`, `O` of shape `(64, TEAMS)`,
  `pick = argmax(G, axis=1)`. Constant picks, id-hash picks, or anything that
  ignores `X` are not an acceptable simplification — the point of the bridge is that
  behaviour is conditioned on a solve.
- **Write.** Rows `[bot_id, pick, G[0..4], epoch]`, 126 rows per slot, chunked with
  the same header and `kind = 2`, `req_id`/`tick`/`rows_total` echoed. Lay the
  chunks out at `stride` in a rotating TX window and `m.write(first, chunks, src)`;
  a short return is a partial block that the engine will never complete, and the
  worker moves to the next request rather than retrying.
- **No state across requests** except the model weights. No queue, no backlog, no
  catch-up: if requests arrive faster than the solve, the worker plans the newest
  complete block it holds and discards the rest. A solver slower than the tick costs
  responses, never frame time.

## 5. Failure posture

The sealed promise is that a page may not arrive. The whole handling of that is:

**A lost request or response is a tick with no new plan. The QC keeps its
last-known objectives.**

Concretely:

- The engine never retries, never resends, never waits, and never times out. There
  is no path in `mesh_ipc.c` that spins on a response.
- `mesh_publish` runs unconditionally every tick with a fresh `req_id`. A lost
  request is superseded ~16 ms later by a newer, better-conditioned one; a resend
  would deliver stale features.
- An incomplete response block is never rendered. `mesh_poll` returns the previous
  `done_id`, the QC's `mesh_poll(h) > lastplan` test fails, `mesh_scatter` is not
  called, and every bot's `.objective` field keeps the value it already had.
- A dead or restarted solver is silence. `mesh_poll` keeps returning the last
  `done_id` and the engine keeps publishing into it; when the solver comes back it
  starts answering the current `req_id` with no handshake, because every slot is
  self-describing. There is no epoch latch to go stale, which is what the old
  protocol needed `epoch` for.
- A bridge that is not up when the map loads yields `mesh_open == -1`. The handle
  stays detached, `mesh_publish`/`mesh_poll` on it return 0, the game runs with
  bot objectives chosen by the ordinary QC path, and QC retries `mesh_open` on its
  own slow timer. The capability is never withheld, and it is never charged to a
  frame.
- Losses are counted, not hidden: stat 6 (dropped incomplete blocks), 9 (short
  writes), 2 (in flight). A quiet bridge is visible in `mesh_stat`, not inferred
  from bot behaviour.
- Nothing in this bridge sends a signal, kills a process, or stops a bridge. The
  engine attaches and detaches by exiting.
