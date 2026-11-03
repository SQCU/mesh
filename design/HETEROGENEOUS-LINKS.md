# Links between devices that hold different numbers of queue pairs

A link's two bridges used to make ten queue pairs each, whatever their devices held. A device that holds fewer
could not pair at all. This document covers the failure that showed it (2026-10-21), what probes of the new machine
showed, what the first remedial change does, and what remains. Nodes are named here by their bridge's node number
(drawn from the host UUID) and their hardware. The operator, 2026-10-21: "from this point onwards, we never rely on
something like a localhostname as a name or identifier for something which we are programming."

| node | hardware | macOS | its link |
|---|---|---|---|
| 521750041 | Mac17,15, M5 Ultra, 96 GB; serial WP6Q97X9Y3, host UUID E8F9C574-257D-5425-8501-E5BF6981BEA6, fabric fd6d:6573:68:577f:d1f4:93fc:f466:51a6 | 27.0 (26A425) | rdma_en3, cabled to node 1974797816's en2 |
| 1974797816 | Mac16,11, M4 Pro | 26.5.1 | en2 to 521750041; another port to 1980260065 |
| 1980260065 | Mac17,6, M5 Max | 26.3.2 | to 1974797816 (this link works) |

## 1. The failure

Measured by the operator on 2026-10-21 with a read-only `ibv_open_device` and `ibv_query_device` (nothing allocated,
no close):

| device | max_qp | max_qp_wr | max_cq | max_cqe | max_mr | max_pd | max_sge |
|---|---|---|---|---|---|---|---|
| 521750041 rdma_en3 (M5 Ultra, 27.0) | 3 | 4095 | 6 | 4096 | 100 | 6 | 1 |
| 1980260065 rdma_en1 (M5 Max, 26.3.2) | 11 | 4095 | 11 | 4096 | 100 | 11 | 1 |
| 1974797816 (M4 Pro, 26.5.1) | not queried; transport.md's figure is 11, 10 usable | | | | | | |

Node 521750041's bridge, built from the same source and MESH_VERSION (131) as node 1974797816's, logs every 5 s:

```
link 2 (en3): its child N spawned
rdma_en3: region of 92341796864 bytes (86 extents) registered in 0.165 s
link 2 (en3): paired with node 1974797816
link 2 (en3): lanes toward node 1974797816 not made: No space left on device; paired again with the port's next child
link 2 (en3): node 1974797816 lost: code 28
```

and node 1974797816's end logs `node 521750041 lost: code 52` (ENETRESET, the control socket closed under it). The
cause: `rdma/mesh-bridge.h` fixed `MESH_LINK_LANES` at 10, `rdma/mesh-verbs.h` `device_up` set a device's usable
queue pairs to `max_qp - 1` (2 here), `pair_up` returned ENOSPC when `pairs + count > max_pairs`, and `link_make`
asked `pair_up(p, MESH_LINK_LANES)`. The check comes before any verbs object is made, so each 5 s cycle registers the
whole region (86 one-GiB extents), pairs the control socket, fails the make, and recycles the port's child.

## 4. What the probes showed

All probes on node 521750041 only, one process at a time, each under `perl -e 'alarm 20; exec @ARGV'`, compiled
there from source sent over ssh to `/tmp` and removed after; no release call of any kind (the process's exit releases
everything); only the link-down devices rdma_en2, rdma_en13, rdma_en4, rdma_en5, rdma_en6 (the bridge holds
rdma_en3). The node had rebooted cleanly at 23:45:39Z (its bridge stopped by SIGTERM). The allocation probes ran
after its bridge had run 10 minutes and cleared its boot ledger. Every probe exited at once, and the bridge's log
shows nothing but its rdma_en3 cycle throughout. Node 1980260065 (this checkout's machine) was only read passively:
`ioreg` and `kmutil showloaded`, no verbs call.

**Passive (no device opened).**
- The kexts differ between the two OS builds: `com.apple.iokit.IORDMAFamily (1.0)` is 0x9316 bytes on 27.0 against
  0x6e24 on 26.3.2, and `com.apple.driver.AppleThunderboltRDMA (0.0.1)` is 0x29ba5 against 0x26cb5. Both keep the
  same version strings.
- The 27 SDK adds `/usr/lib/rdma/librdmacm.dylib` (the RDMA connection manager: `rdma_create_id`, `rdma_connect`,
  `rdma_listen`, ...) to `librdma.tbd`, and renames `IBV_NODE_TBT` / `IBV_TRANSPORT_TBT` to `IBV_NODE_THUNDERBOLT` /
  `IBV_TRANSPORT_THUNDERBOLT` in `infiniband/verbs.h`. The bridge uses neither name. Nothing else in `verbs.h` changed.
- The IORegistry on 27 holds, under every `AppleThunderboltRDMAInterface` (each listed twice in a tree walk, commit
  dabb136f), from boot and before any process opens the device: one `AppleThunderboltRDMAProtectionDomain`, one
  `AppleThunderboltRDMAKernelCQ` and one `AppleThunderboltRDMAKernelQP`, none with properties. There is one
  `IORDMACM` in the whole tree. Each open context is an `IORDMAFamilyUC` user client (the bridge's children held one
  on every port), and a user PD is another `AppleThunderboltRDMAProtectionDomain` under the interface (rdma_en3 shows
  the child's).
- On 26 (node 1980260065, passive) there are no kernel PD, CQ or QP objects, no IORDMACM, and each user queue pair is
  an `AppleThunderboltRDMAQueuePair` under its device's interface (ten on the device whose link is up).

**Read-only query of the five link-down devices** (`ibv_open_device`, `ibv_query_device`, `ibv_query_port`): every
one reports what rdma_en3 does: max_qp 3, max_qp_wr 4095, max_cq 6, max_cqe 4096, max_mr 100, max_pd 6, max_sge 1,
max_srq 0, max_mw 0, atomic_cap 0, max_ah 32768, max_pkeys 1, device_cap_flags 0, max_mr_size 16384000 (not
enforced: the bridge registers 86 one-GiB extents on rdma_en3), page_size_cap 0x4000. The ports report state DOWN,
gid_tbl_len 1024 and max_msg_sz 16 MiB. The figures do not depend on the link being up, and all six devices of the
machine report the same.

**Allocation on the link-down devices.** A context's `ibv_alloc_pd` was refused (errno ESRCH in one run, 0 in the
next) and its `ibv_create_cq` was refused with EBUSY: in a first and a second context on rdma_en2 and on rdma_en5, and
in a context on rdma_en4 and on rdma_en6. On 27 a device whose port is down makes no PD and no CQ, so no queue pair
could be made, and the user queue pair count was not measured on these devices. The bridge is not
affected: `device_up` reads the port active before it allocates.

**The operator's questions, as far as the evidence goes.**
- *Is max_qp per context or per device?* Not measured on 27. On 26 each user queue pair is an IORegistry object under
  its device's interface, and transport.md section 14's history (an old child's queue pairs blocking a new child's
  make) points to a per-device count. On 27 the reported figures are identical in every context of every device.
- *Does the driver's kernel QP change what a user QP is?* The kernel QP is a separate object, made at boot, one per
  device, beside a kernel PD and a kernel CQ. 27 also ships the connection manager (librdmacm, IORDMACM). The
  likeliest reading, not established, is a QP the driver reserves for connection management, as InfiniBand reserves
  QP1. If it counts against max_qp as 26's unexplained "11, 10 usable" did, a 27 device holds 2 user queue pairs.
  Nothing queried suggests a user QP is otherwise different: the same 4095 work requests, one SGE, 16 MiB messages, no
  SRQ, no atomics, no memory windows.
- *Do max_cq 6 and max_pd 6 constrain the design?* Not for one context a device: a port's child takes one PD and two
  CQs, and the kernel's one PD and one CQ leave at least five of each. They would bound a design that spreads one
  link over several contexts of a device (section 8, P3): at most two more contexts' CQ pairs.
- *Can a QP on 27 carry what several did on 26?* No. On this transport a queue pair carries one sequence of receives,
  because a SEND fills the oldest posted receive and must cover its frame count exactly (TN3205). Several streams on
  one queue pair need either a fixed interleave, where an idle holder blocks the rest, or receives into staging with a
  header and a copy, which gives up zero copy. 27 adds no SRQ and no immediate data that would change this, and its QP's
  capacities equal 26's.
- *macOS 27 or M5 Ultra?* Not separable with this data. Node 521750041 is both the only macOS 27 machine and the only
  M5 Ultra. The kernel objects and librdmacm are visibly the OS's (a driver and SDK change), but the count 3 could
  come from either. What would separate them: the same read-only query on an M5 Ultra running 26, or on an M4 Pro or
  M5 Max running 27. The second Studio (serial CJ473V7FJV) was not queried; its macOS version is not known here.

## 5. The change: each end states its budget, the link makes what both hold

- **Budget.** A port's child reads its device's usable queue pairs at its opening, `max_qp - 1` as before
  (`mesh_device.max_pairs`), and tells the parent in CHILD_OPENED (`lanes`). The parent keeps it as the port's
  `budget`.
- **Stated in the HELLO.** The HELLO a bridge sends on a link's control socket carries `count` = 1 + its budget.
  `hello_budget(count)` reads it back, and a `count` of 0 is a bridge before budgets, which makes all
  `MESH_LINK_LANES`. The multicast discovery hellos are unchanged.
- **The plan.** At the greeting each end computes `plan = link_plan(min(own budget, peer's budget))`, a bit a lane of
  the existing layout. Both ends compute it from the same two numbers. It goes to the child with CHILD_MAKE (`lanes`).
  The child makes one queue pair per lane of the plan (`pair_up`), tells them in CHILD_MADE, connects only those, and
  serves `served = plan`. The LINK names the plan's lanes in `count` and is checked against this end's plan.

| budget (the fewer of the two ends') | plan | lanes | streams | lane 0 |
|---|---|---|---|---|
| 0, 1 | 0 | 0 | 0 | not made: the link pairs, records flow, no stream is live |
| 2 | 0xc | 2 | 1 | not made |
| 3, 4 | 0xd | 3 | 1 | made |
| 5, 6 | 0x3d | 5 | 2 | made |
| 7, 8 | 0xfd | 7 | 3 | made |
| 9 | 0x3fd | 9 | 4 | made (no reserved lane 1) |
| 10 and more | 0x3ff | 10 | 4 | made, and lane 1 |

Lane 0 comes first once a stream fits beside it (`rdma/mesh-bridge.h` `link_plan`: bit 0 where the budget is three or
more). The reserved lane 1 is made only in the full plan, so a full plan is lane for lane the layout of MESH_VERSION
131. The plan of a smaller budget is always a subset of a larger one's.

Since version 133 there is no keepalive (section 10; design/transport.md section 5: "a link carries no keepalive"), and
nothing is posted on lane 0 or lane 1. The plans stay as in this table, so the two ends' LINKs still match.

- **Streams.** Streams beyond the plan's are never opened: not live, not free, never offered to a client
  (`mesh_streams_free`), and never in the node's reach (`mesh_nodes_reached`).
- **Completion queues.** Each stream lane's share of the two completion queues is `cqe` over the plan's stream lanes
  (mesh-link.c:584), so a two-lane link's lanes get 2048 frames each, against 512 on a ten-lane link's eight stream
  lanes. A stream lane's receive window is its share counted in pages, at most the queue pair's receive capacity
  (mesh-link.c:693-697). Lanes 0 and 1 have no window: nothing is posted on them.

**What a two-lane link costs.** Node 521750041's link to node 1974797816 gets the 0xc plan: one stream and no
lane 0.
- One communicator at a time between the two nodes. A second concurrent one is refused at its init with EBUSY as a
  value ("rank 0 could not hold its streams at its bridge"). The first runs exact.
- No keepalive frames, as on every link since version 133 (section 10), so a controller that slows after a quiet
  stretch (the Mac mini's SEND 6.2 us after 40-50 us of nothing, section 10) is not kept warm. This costs latency on
  the first frames after a pause, not correctness.

## 6. Versions: bridges of different MESH_VERSION or macOS in one map

**macOS versions.** With budgets, nothing in the bridge depends on the OS version itself. What differs between 26 and
27 and what the bridge does about each:
- queue pair, CQ and PD limits: the budget above;
- PD and CQ refused on a link-down device (27): `device_up` allocates only once the port reads active;
- `IBV_NODE_TBT` renamed: unused;
- a GID table of 1024 entries on 27: `device_gid` reads at most 256, taking the first nonzero;
- the SDK: each node builds its own binaries with its own SDK, and the Makefile keeps deprecations warnings, not
  errors, for that reason.

A binary built on 26 and run on 27 is untested. Every node builds its own with `make -C rdma all install`.

**MESH_VERSION.** `XMAGIC` is fixed at the value version 131 sends, and no version enters it, so bridges of any version
pair (the operator, 2026-10-21: "rejection filter removal. categorical. all of them without sanction in rubric"). For
versions to join one map:
1. *Extend in place* (done here). New information goes in a field the older version sends as 0 and never reads, with 0
   meaning the older behavior. The older version's exchanges keep their meaning, so MESH_VERSION does not change.
   That covers additive changes like this one.
2. *A version-stating hello* (done in part, 7d236bbb). The header is version-independent: the fixed magic, and the
   sender's MESH_VERSION in `version` (0 from a bridge before it). No reader tests `version`, so a bridge hears a
   peer of any version and pairs with it. Not done: logging which version a peer runs.
3. *Records appended.* A field added at the end of `struct mesh_record`, the region's header or a slot leaves the
   fields before it where they were, so a newer bridge floods records an older one can still read up to its own
   length.
4. *The region stays apart from the wire* (already so: MESH_LAYOUT), so a wire change never expires a region.

## 8. What remains

- **P1.** Open: hardware-and-discipline.md:26, "An M5 Ultra device holds 2 usable queue pairs a link" (under
  "Measured on these machines"), against HETEROGENEOUS-LINKS.md:232, "What a 27 device holds: not measured. The
  budget assumes `max_qp - 1`." A process that opens rdma_en3 with its link up, allocates a PD and two CQs and makes
  UC queue pairs until refused would read the count; a second context of the same device would tell a per-context
  count from a per-device one, and a second device with its link up (the second Studio, cabled) whether the count is
  pooled across a machine's devices. If 3 are usable, `max_pairs`
  for this driver is `max_qp`, and the plan gains lane 0 (0xd).
- **P2. Making until refused (not done).** Each end could make its plan's
  queue pairs in the order 2, 3, 0, 4, 5, 6, 7, 8, 9, 1, so that the first c made always contain `link_plan(c)`.
  It would stop at the first refusal and name in LINK how many it made, and both ends would connect
  `link_plan(min(made, peer's made))`. Queue pairs made past that stay in INIT until the child's exit. Two 131-
  compatible ends both make ten, so the wire stays compatible. This removes the reliance on `max_qp - 1` for any
  hardware.
- **P3. More streams on few queue pairs.** If P1 finds a per-context count, a child can open more than one context on
  its device and spread a link's lanes over them, each context with its own two CQs (max_cq 6, one the kernel's,
  bounds it to two contexts). The progress threads would poll both pairs of CQs. Otherwise, more streams on two queue
  pairs would need a staging ring and a copy (section 4); the operator, 10-21: "zero copy methods must always be used
  over access by copy".
- **P4. The plan is not published.** `struct mesh_link_info` (the bridge's observation pages) has no room for the plan without a
  MESH_LAYOUT bump, so mesh-stat cannot show it yet; the log line does. A record's `reach` could carry each link's
  stream count so a planner knows how many communicators a node pair takes at once.
- **P7. Host names in the bridge's area** (the debts listed in AGENTS.md at e6d406ad). `nccl-mesh.c`
  `ncclGetUniqueId` falls back to `gethostname()` for the bootstrap's listen host when the node holds no fabric address.
  The operator, 2026-10-21: "we should be able to use any kind of virtualization of 'names' such that we never rely on
  anything like, again, that localhostname thing, which makes no promises of selectivity or unique identifiability or
  anything of the sort".

## 10. The keepalive on every link

**Since version 133 (2026-10-22) there is no keepalive.** The operator: "we don't introduce race conditions or use
timing anywhere. why? because using timing is always slower than not using timing, and we require the lowest
attainble overhead on all transactions in bridge/collective". A stream's lanes carry its holders' entries' pages and
nothing else (design/transport.md section 14), so no frame on a timer can fill a receive a holding's unit is owed,
and no spare receive waits between holdings for one.
- On a link of three or more queue pairs, lane 0 now carries nothing, as lane 1 never did. `link_plan` is unchanged:
  it still makes lane 0 where both ends hold three or more, so two bridges plan alike, and a later change may give
  the queue pair to a stream.
- The cost: a Thunderbolt controller that has been quiet is slow on its next frame (the Mac mini's SEND completed in
  6.2 us after 40-50 us of nothing, 1.9 us with a frame every 5 us, metal-microbench docs/measurement.md:3184-3197),
  now at a holding's first SEND and after every quiet stretch inside a holding (a layer's 40-50 us between crossings),
  on every plan.
- The operator's 5 s deadline (design/transport.md section 5, 2026-10-18) bounds the bridge's lifecycle, not a link's
  traffic: since 2026-10-22 no clock decides a transaction (design/transport.md section 5).
