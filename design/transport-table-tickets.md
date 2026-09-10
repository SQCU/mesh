# Transport-and-table work tickets

Ongoing work for `rdma/` after the September 10, 2026 review. Scope is the
page transport and the page table: `mesh.h`, `mesh-dataflow.c/.h`,
`mesh-flow.c`, `mesh-links.h`, `mesh-transport.c/.h`, `mesh-metal.m`,
`mesh-tensor.m`, `mesh-client.c`, and the numerical caller's use of that API
in `metal-microbench/reduce_scatter.swift` and `tools/mesh/participants.py`.

The governing statements are [pages-and-functions.md](pages-and-functions.md)
and [distributed-reduce.md](distributed-reduce.md): the page table is the only
state; a row holds a page, a stamp and a use count; a function runs when its
input rows carry stamp k; release at zero use; the bridge routes pages and
holds no schedule; no message, gate, token or wait stands in for a stamped row.
Every ticket below deletes something the specification does not require.
Each names what is deleted, what replaces it, and what evidence closes it.
Line references are to the tree at canonical `cf867e2`.

Order: A7 first, so every later deletion is measured against a transport
yardstick that exists. Then A1+A2 as one change. Then A4–A6. Then B, C, D.
Tickets marked *done* or *withdrawn* stay for the record.

The operator's subsequent clarification in
[pages-and-functions.md](pages-and-functions.md#backing-memory-and-logical-values)
supersedes A1's compulsory page prefixes and A2's transport interpretation of
logical rows. Metadata occupies configured byte extents; physical pages have no
numerical semantics. These tickets retain their historical prescriptions below
so that they are not mistaken for completed implementations.

## September 10 connected rewrite status

[Committed source and operational evidence](transport-deletions-2026-09-10.md)
records the current implementation and exact measured revisions. This section
supersedes the historical open/half-done labels below.

- A4–A6 and B1–B7: the named excluded implementations are deleted; matching
  builds, three full one-GiB streams, and both real-input FFN configurations pass.
- B8: the cancellation/sort/EBUSY implementations are deleted. The operator's
  later index-mask instruction supersedes the ticket's QP-error/detach prescription.
  Table invalidation is local and does not tear down the link.
- C1–C2: the numerical caller uses cached registered-region buffers and resolved,
  command-specific extent addresses. Genuine scatter-to-contiguous result aliases
  remain for NumPy/MLX APIs; the operator explicitly permits these zero-copy views.
- C3: the generic tensor runtime is deleted and its live consumers migrated.
  Their application execution evidence remains outstanding.
- D1: canonical ctypes mirrors are replaced by header-generated declarations.
  `peers.py`, `workload.py`, and `xonwire.def` implement live application data,
  rather than redundant canonical ABI mirrors; their deletion prescription was
  incorrect and is withdrawn.
- D2: repeated stop/layout/start/ready cycles between numerical cases are removed.
  Configuration registers once for the batch's maximum declared storage.
- A1/A2 remain superseded as described above. The registered metadata/payload
  transfer pair remains explicit; no claim of one WR per payload is made.

## Corrections carried into these tickets

- Under TN3205 SEND/RECV on one QP, receives are matched to sends in posting
  order; `wr_id` is not transmitted; there is no immediate data. A receiver
  therefore cannot pre-post a RECV into a specific destination row's page
  when sends stream in arbitrary order. The landing page is anonymous by
  construction. The association from a transfer to its row is the header
  written inside the transmitted page. This withdraws the "receive into the
  row's own page" proposal and merges it into A1/A2.
- TN3205 provides hardware credit flow control: a SEND is not processed until
  the peer has posted a matching RECV. A late re-post delays; it never loses a
  page. No software credit, acknowledgement or receive-count mirror is needed.
- Queue capacity is counted in 4 KiB frames (TN3205). A 16 KiB page plus a
  separate 64-byte header costs five frames; a header inside the page costs
  four. Message length and posted receive length must match in frames.

## A. Transport

### A7. Page-echo transport measurement — open, first

Implementation and initial measurements are in
[transport-page-echo-2026-09-10.md](transport-page-echo-2026-09-10.md).
Exact echo and lifecycle checks pass. Stable repeated latency/rate measurement
and attribution of the instrumented-run variation remain open.

The operator superseded the short echo with full 1 GiB one-way streams.
[The matched direct-versus-mesh record](transport-one-gib-2026-09-10.md)
contains the accepted bandwidth yardstick, exact payload/completion validation,
and the fixed receive-preposting bug. Repeat mesh throughput is within 0.53%
of the direct path; first-use overhead remains open. A1–A6 must preserve that
repeat rate and distinguish first-use cost from link serialization.

Delete: nothing. Add: one program (~100 lines) that sends N pages one way
through the actual path (`SUB` ring → bridge → QP → peer bridge → row table)
and has the peer echo them, with no numerical function. Report one-way page
latency, sustained pages/s and bytes/s for the configured page size, the
bridge pass period, and `ibv_poll_cq` cost per call. Record the port's
`active_width × active_speed` beside the result.
Why: the only existing transport rate is an inherited 3.53 GB/s estimate;
no isolated measurement of this link, this page size or this bridge exists.
Every other ticket's effect is otherwise unattributable.
Done when: `design/` holds the record with commits on both machines, and the
bench is the acceptance yardstick cited by A1–A6.

### A1 (+A3). Header inside the page — superseded

Delete: the separate header region (`hdr.headers_off`, `MESH_HEADER_STRIDE`,
`mesh_header`), `struct mesh_send` records with their `next/previous/owner`
pending list (`link_submit`, `link_release`, `link->pending`), the two-WR post
per page, the `1<<62` header-completion identity and the `header_completion`
branch in `mesh_progress`, the `frames = 5` accounting.
Replace: `struct mesh_page_header` at byte 0 of the transmitted page, payload
at byte 64; one SGE, one SEND, one signaled completion, four frames. The
receiver lands the page in any free page, reads the header, and publishes the
row (A2). The free list of landing pages remains; nothing else of the pool's
state machine does (`owner[]`, `pool_link[]`, `counts[NOWN]`, `MOVE`,
`RELEASE`, `receive_share`, `mesh_received_pages`, the `receive ? UInt32.max`
case in `MeshValue.map`, the `MESH_RECEIVE_PAGES` lifecycle parameter).
Caller ripple: payload capacity becomes page size minus 64 while page stride
is unchanged. Kernels currently use one constant `g[0]` for both `byte/g[0]`
and `byte%g[0]`; they need a stride and a payload constant. The existing
`mesh_rows.offset` field, currently forced to 0 by `mesh_rows_validate`, is
the payload offset. Channel-layout region pages that are exactly full today
(1920 × 128 × 2 bytes = 30 pages) gain one page of padding per region.
Why: TN3205 `max_sge = 1` limits one request; it does not require two.
Nothing in the algorithm requires a page to spend a frame on a struct.
Done when: one `ibv_post_send` and one completion per page; A7 shows the
per-page cost change; the FFN comparison still passes at exact peer agreement.

### A2. Bridge writes rows directly — superseded

Delete: `CMP` and `ACK` rings, `ring_select`/`ring_erase` swap-removal and
its cursor, the `continue`-and-rescan path in `mesh_rows_poll`, the whole top
of `mesh_rows_poll` (ACK drain, CMP drain), `mesh_context_row`,
`mesh_context_metadata`, `mesh_context_consume`.
Replace: on receive completion the bridge reads `{table, row, stamp}` from
the in-page header and stores `{page, uses, stamp}` into that row of the
table, which is already in the shared region it maps. On send completion it
decrements the source row's `uses`. Transport errors are written into the
page header and the port record and are never stamped.
Why: the bridge can read and write memory it maps. A descriptor ring that
carries a completion to a client which then performs the same 16-byte write
is a message standing in for a row.
Done when: `mesh_rows_poll` no longer exists; arrival is one atomic row
write in the bridge; A7 latency drops by the removed ring hop.

### A4. Delete zeroing — open

Delete: the `memset` per released page in the `mesh-flow.c` main loop, the
`memset` of output and indices pages in `mesh_rows_realize`, the
`MESH_ROW_WRITING` compare-and-swap protocol in `mesh_release_page`, and the
`REL` round trip for arena pages.
Replace: an arena row whose uses reach zero is flipped to `ABSENT` by the
releaser; a landing-page row whose uses reach zero pushes the page number on
one plain SPSC FIFO so the bridge re-posts it. That FIFO is the only ring
left after A2.
Why: validity is the stamp. Producers write every byte consumers read; the
NIC fills the whole receive page. The bridge currently clears about 300 MB
per 4096-row FFN generation, one page per progress pass, and the next
generation's selection is gated on it (`mesh_rows_select` requires
`page == ABSENT`).
Done when: no `memset` in the transport path; one-in-flight FFN latency and
two-in-flight throughput are re-measured and recorded.

### A5. One queue pair, one thread, fail-stop — open

Delete: `link_worker` and its `LINK_SETUP/ACTIVE/RELEASED/RETIRED/ACKNOWLEDGED`
ownership state machine with `usleep` handshakes, `reset_request`,
`mesh_link_reset`, `retire_device`, the reconnection loop, the nonce self
check, `system("ping6 ...")`, the multi-link `--link/--route` tables.
Replace: pair once through the out-of-band TCP exchange TN3205 requires. A
verbs error is recorded in the port metadata and the process exits; the data
layer relaunches. One peer, one QP.
Why: recovery inside the transport is the protocol the specification
excludes. Failures are result values; relaunch happens at the data layer.
Done when: `mesh-flow.c` has no thread besides `main` and no state beyond
paired/faulted.

### A6. The service loop carries only verbs — open

Delete: per-pass `now()`, the 0.25 s telemetry publication, `flight_begin`/
`flight_end` `TRACE` wrappers on every verbs call, `heartbeat`, `mean/sd`
publication, `flight_status` operation stamps.
Replace: counters the bridge increments and `mesh-stat` reads. The flight
log becomes an opt-in build, not a per-call cost.
Done when: a pass with no completions executes only a CQ poll and a FIFO
check; A7 pass period is recorded before and after.

### A8. `mesh-transport.c/.h` (758766b) — done

Deleted: five one-line pass-throughs to `ibv_post_send`,
`ibv_post_recv`, `ibv_poll_cq`, plus two constructors that copy an SGE's
fields back into the same SGE; a third dylib and a Makefile target. Mechanism
unchanged.
Resolution: existing owners construct descriptors and call verbs directly;
the wrapper sources, dylib target and linkage are removed.

## B. Table

### B1. Send at publication, not by scan — open

Delete: `mesh_rows_send`, which reads stamp and page of every row of every
send binding on every poll.
Replace: an output row's binding membership is static. `mesh_rows_complete`
stamps the row and, if the row is outbound, pushes it to the bridge FIFO
there. Cost is proportional to pages published.

### B2. Release is the event; delete retirement — half done

Done (cf867e2): the recursive walk through receive headers in
`mesh_rows_return`.
Open: `mesh_rows_retire` and `mesh_rows_retire_range` still scan every
function's indices pages and every return range per poll; `mesh_rows_return`
and `mesh_release_page` still implement a stamp CAS protocol.
Replace: `mesh_row_release` reaching zero performs the release itself (A4).
Nothing scans for zero.

### B3. Selection buffers are host memory, not rows — open

Delete: the per-function `indices` value (pages reserved per occurrence set),
the `width` computation repeated in `mesh_rows_issue`, `mesh_rows_complete`
and `mesh_rows_retire`, the availability scan over reservations, retirement
of indices pages, `values[0] = 0`, and the `indices` clauses of
`mesh_rows_validate` and `mesh_rows_close`.
Replace: a per-function ring of small Metal-visible host buffers, owned by the
caller, holding the selected occurrence list for one command buffer.
Why: the list is ephemeral host data used to encode a dispatch; it has no
stamp, use count or lifetime beyond the command buffer.

### B4. Static use counts — open

Delete: the `delta[]` counting pass in `mesh_rows_realize`, the per-(row,
occurrence) `uses[]` arrays attached to every output map and binding, the
copy-per-claim in `mesh_rows_select`.
Replace: one initial use count per map, computed once at configure, stored in
the row at claim.

### B5. Delete the common-input heuristic — open

Delete: the post-pass that sets `map->stride` from range inequality, the
`common_checked` branch in `mesh_rows_select`, the
`fetch_sub(uses, values[0])` batch release in `mesh_rows_complete`.
Replace: an input is a range per occurrence. Equal ranges are equal checks.

### B6. Configure-time rows are stamped once — open

Delete: the `.source` functions the caller issues and completes every
generation to re-stamp tokens and the initial tensor, the `immutable` flag
through validate/select/retire/close, the `uses <= 1` special case.
Replace: `present(row, g)` treats a configure-time row as present for every
generation.

### B7. Configuration is host memory — open

Delete: the second half of `row_layout` that copies functions, maps,
bindings and returns into arena pages through `row_storage`, the dry-run
`mesh_rows_configuration_pages`, `mesh_region_table_pages`, and in the
caller the `arenaPages` fixed-point loop plus the separate `LM_MESH_LAYOUT`
planner process before every run.
Keep: the one-time validation (bounds, overlap). It is correct and runs once.

### B8. Invalidate is detach — open

Delete: `mesh_rows_invalidate` (held-return permutation and sort, per-binding
cancellation), `mesh_rows_held`, the `EBUSY` ladder in `mesh_rows_close`, the
caller's `precondition(status == EBUSY)` spin.
Replace: close moves the QP to error, drains the CQ, unmaps. "The calling
process may invalidate the page table whenever it chooses" is satisfied by
exit.

### B9. Bindings have no inputs — done (cf867e2)

## C. GPU view

### C1. One Metal buffer over the region — open

Delete: `mesh_memory_view`/`mach_vm_remap`, `mesh_memory_release`,
`mesh_metal_page_span`, the 1 GiB atlas table in `mesh_metal_regions`,
`mapping_pinned`, per-value `MatrixView` buffers, the `accumulatorViews`
tracking list.
Replace: `newBufferWithBytesNoCopy` on the mapped region once (per 1 GiB only
if `maxBufferLength` forces it); every kernel receives that buffer.

### C2. Address per occurrence, not per element — open

Delete: the `uint4` row-table buffer and `MeshRegion` argument from every
kernel; `number()`/`payload()` currently perform a table lookup and a region
lookup per scalar.
Replace: the issuing host writes each selected occurrence's page base offsets
into the selection buffer (B3); kernels index `base[i] + offset`.
Depends on: A1 (fixed payload offset), B3.

### C3. Delete `mesh-tensor.m/.h` — open

A second generic tensor program API with its own queue, view/dimension/
argument pages and residency lists. The NFE never executes it; it has no
numerical validation. Same disposition as the deleted protocolized dataflow.

## D. Tooling and lifecycle

### D1. Python struct mirrors — open

`mesh.py`, `peers.py`, `workload.py`, `mesh-flight.py`, `xonwire.def`,
`vendor/dlpack.h`: hand-maintained ctypes copies of every struct, paid for
again in cf867e2. Delete with their structs; regenerate one shim from the
header if `participants.py` still needs `--ready`.

### D2. Bridge lifecycle per run — open

`participants.py` performs stop → layout → start → ready through launchd,
with `MESH_ARENA_PAGES`/`MESH_RECEIVE_PAGES` and a `wire_check` sysctl before
every pair. Replace: the bridge registers a fixed region once; the client
sub-allocates. `participants.py` launches both clients and collects.

## Caller obligation outside this scope

Publication granularity is the page. Once `mesh_rows_complete` stamps at
page granularity and A1–B2 remove per-page overhead, the numerical caller's
per-owner, per-call publication on the MPS participant is what lets the
gather stream behind the reduce with no phase boundary. That change lives in
`metal-microbench/reduce_scatter.swift` and is tracked there.

## What remains after A–D

`mesh.h`: region header, row, one SPSC FIFO, port record. `mesh-dataflow.c`:
create, map, validate, issue, complete, release. `mesh-flow.c`: pair, post,
poll, write rows. `mesh-metal.m`: one buffer, kernel source. The page table is
the only state; the bridge routes pages and touches the table directly;
nothing polls the table except the firing rule.
