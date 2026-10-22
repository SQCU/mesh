# Links between devices that hold different numbers of queue pairs

A link's two bridges used to make ten queue pairs each, whatever their devices held. A device that holds fewer
could not pair at all. This document covers the failure that showed it (2026-10-21), what
[transport.md](transport.md) assumed, every place the code depends on the count, what probes of the new machine
showed, what the first remedial change does, and what remains. Nodes are named by their bridge's node number (drawn
from the host UUID) and their hardware, never by host name ([AGENTS.md](../AGENTS.md), "A node is its hardware;
names are virtual").

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
link 2 (en3): its agent N spawned
rdma_en3: region of 92341796864 bytes (86 extents) registered in 0.165 s
link 2 (en3): paired with node 1974797816
link 2 (en3): lanes toward node 1974797816 not made: No space left on device; paired again with the port's next agent
link 2 (en3): node 1974797816 lost: code 28
```

and node 1974797816's end logs `node 521750041 lost: code 52` (ENETRESET, the control socket closed under it). The
cause: `rdma/mesh-bridge.h` fixed `MESH_LINK_LANES` at 10, `rdma/mesh-verbs.h` `device_up` set a device's usable
queue pairs to `max_qp - 1` (2 here), `pair_up` returned ENOSPC when `pairs + count > max_pairs`, and `link_make`
asked `pair_up(p, MESH_LINK_LANES)`. The check comes before any verbs object is made, so each 5 s cycle registers the
whole region (86 one-GiB extents), pairs the control socket, fails the make, and recycles the port's agent.

## 2. What transport.md assumed

- Section 1, "A link": "ten unreliable-connected queue pairs (index 0 carries the keepalive frame)".
- Section 2: "ten queue pairs per link shared by every client", and in the list of private state, "Ten queue pairs a
  link".
- Section 14, the stream: "Each link carries `MESH_LINK_STREAMS` (4) streams ... each on a lane pair of its own, since
  a queue pair carries one ring".
- Section 14, "A process a port": "makes and connects the link's ten queue pairs" and "A device holds ten usable queue
  pairs and a link takes all ten, so a returning peer's lanes cannot exist beside the old ones, and the old agent's
  exit is their one release".
- README.md, from TN3205: "10 UC queue pairs per device"; `rdma/mesh-verbs.h`: "max_qp: 11 (10 usable)";
  `rdma/loopverbs.c`: `LOOP_DEVICE_QPS 10`.

Each statement was true of every device the mesh had. Nothing in the code read the count from the peer: both ends
assumed that both devices held ten. This change edits those passages to say a link takes what both ends hold.

## 3. Every place the code depends on the lane count

Before this change, in `rdma/`:

| where | what depends on it | after this change |
|---|---|---|
| `mesh-bridge.h` `MESH_LINK_LANES` 10, `MESH_LANE_KEEPALIVE` 0, `MESH_LANE_FIRST` 2, `MESH_LANE_PAIRS` 4, `MESH_STREAM_PAIRS` 1, the static asserts, `lane_class`, `lane_stream`, `lane_stripe` | the lane layout: keepalive lane 0, lane 1 reserved and unused (the stripes lane deleted at 125, kept so indices did not move), stream s's large lane 2 + 2s and small lane 3 + 2s | unchanged; the layout is the index space, and a link makes the subset its plan names (`link_plan`) |
| `mesh-bridge.h` `struct message` `pair[MESH_LINK_LANES][2]` and `count` | the wire's LINK: every lane's queue pair number and first PSN; `count` was always 10 | the same size; `count` is the plan's lanes, 0 in `pair` for a lane the plan leaves out; a HELLO's `count` carries the budget (section 5) |
| `mesh-bridge.h` `struct link` `peer`, `qpn`, `gate`, `lane` arrays; `struct agent_message` `qpn`, `peer` | per-lane state, sized by the layout | unchanged in size; `plan` added to the link, `lanes` to the agent message, `budget` to the port |
| `mesh-observe.h` `MESH_LINK_STREAMS` 4, `MESH_PORTS` = `MESH_STREAMS` / 4 = 8 | each port owns 4 stream slots of the region | unchanged: the slots stay, those beyond a link's plan are never live (no region layout change, `MESH_LAYOUT` 133 kept) |
| `mesh-verbs.h` `device_up` `max_pairs = max_qp - 1` | a device's usable queue pairs | unchanged; it is now what the port states (its budget) |
| `mesh-verbs.h` `pair_up(provider, count)` | made queue pairs 0..count-1, ENOSPC past `max_pairs` | `pair_up(provider, plan)`: one queue pair per lane of the plan, queue q lane q, the same ENOSPC check against the plan's count; the completion queues' poll entry kept in the provider |
| `mesh-link.c` `link_make` | `pair_up(p, MESH_LINK_LANES)`; each lane's share of the completion queues `cqe / MESH_LINK_LANES` | the plan's lanes; the share divided among them |
| `mesh-link.c` `link_connect` | RTR, RTS, `qpn`, gate of all ten; the keepalive's receives; `lane_reset` of lanes 2..9; four streams' `wire` | the plan's lanes alone; `qpn` UINT32_MAX for a lane left out (no queue pair has that number, `link_lane`); the keepalive's receives only where the plan has its lane; `wire` of the plan's streams |
| `mesh-link.c` `link_send`, `link_receive` | polled through `queues[0].poll`, lane 0's | through the provider's `poll` (a plan without lane 0 has no `queues[0]` pair) |
| `mesh-link.c` `link_tell`, `link_heard` | LINK `count` 10, checked to be 10 | the plan's lanes; a mismatch logged with its cause, then EPROTO as before |
| `mesh-link.c` `agent_jobs` | `qpn` of all ten in AGENT_MADE; `served` = all ten bits | the plan's; `served` = the plan |
| `mesh-link.c` `lanes_room` | request rooms for all eight data lanes | the plan's data lanes |
| `mesh-link.c` `streams_open` | opened all four streams of the port | the plan's streams; the others stay not live and not free, so `mesh_streams_free` never offers them |
| `mesh-link.c` `stream_turn`, `link_held`, `streams_cancel`, `streams_let_go`, `stream_rings`; `mesh-records.c` `busy_tell` | iterate the port's four stream slots | unchanged: a stream that is not live returns at once, and its rings stay leased |
| `mesh-link.c` `streams_open` bandwidth `bandwidth / MESH_LINK_STREAMS` | each stream's share of the link, which `mesh_map` takes as the link's figure | unchanged, so a one-stream link and a four-stream link are compared on the same footing |
| `mesh-link.c` `port_pass` | loops lanes 2..9 checking `served` | unchanged |
| `loopverbs.c` `LOOP_DEVICE_QPS` 10 | the loop devices' max_qp 11 | still the default; `LOOPVERBS_MAX_QP=rdma_loop<i>=<max_qp>,...` sets a device's (section 7) |
| `mesh-call.c`, `nccl-mesh.c` | a communicator holds one stream toward each other member's node; concurrency between two nodes = the link's live streams | unchanged; on a one-stream link a second concurrent communicator between the same two nodes is refused at init with EBUSY, as a value (section 7) |

Nothing outside `rdma/` reads the lane count.

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
  `IORDMACM` in the whole tree. Each open context is an `IORDMAFamilyUC` user client (the bridge's agents held one
  on every port), and a user PD is another `AppleThunderboltRDMAProtectionDomain` under the interface (rdma_en3 shows
  the agent's).
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
could be made, and the user queue pair count could not be measured within the probing rules. The bridge is not
affected: `device_up` reads the port active before it allocates.

**The operator's questions, as far as the evidence goes.**
- *Is max_qp per context or per device?* Not measured on 27. On 26 each user queue pair is an IORegistry object under
  its device's interface, and transport.md section 14's history (an old agent's queue pairs blocking a new agent's
  make) points to a per-device count. On 27 the reported figures are identical in every context of every device.
- *Does the driver's kernel QP change what a user QP is?* The kernel QP is a separate object, made at boot, one per
  device, beside a kernel PD and a kernel CQ. 27 also ships the connection manager (librdmacm, IORDMACM). The
  likeliest reading, not established, is a QP the driver reserves for connection management, as InfiniBand reserves
  QP1. If it counts against max_qp as 26's unexplained "11, 10 usable" did, a 27 device holds 2 user queue pairs.
  Nothing queried suggests a user QP is otherwise different: the same 4095 work requests, one SGE, 16 MiB messages, no
  SRQ, no atomics, no memory windows.
- *Do max_cq 6 and max_pd 6 constrain the design?* Not for one context a device: a port's agent takes one PD and two
  CQs, and the kernel's one PD and one CQ leave at least five of each. They would bound a design that spreads one
  link over several contexts of a device (section 8, P3): at most two more contexts' CQ pairs.
- *Can a QP on 27 carry what several did on 26?* No. On this transport a queue pair carries one ring, because a SEND
  fills the oldest posted receive and must cover its frame count exactly (TN3205). Several rings on one queue pair
  need either a fixed interleave, where an idle holder blocks the rest, or receives into staging with a header and a
  copy, which gives up zero copy. 27 adds no SRQ and no immediate data that would change this, and its QP's
  capacities equal 26's.
- *macOS 27 or M5 Ultra?* Not separable with this data. Node 521750041 is both the only macOS 27 machine and the only
  M5 Ultra. The kernel objects and librdmacm are visibly the OS's (a driver and SDK change), but the count 3 could
  come from either. What would separate them: the same read-only query on an M5 Ultra running 26, or on an M4 Pro or
  M5 Max running 27. The second Studio (serial CJ473V7FJV) was not queried; its macOS version is not known here.

## 5. The change: each end states its budget, the link makes what both hold

- **Budget.** A port's agent reads its device's usable queue pairs at its opening, `max_qp - 1` as before
  (`mesh_device.max_pairs`), and tells the parent in AGENT_OPENED (`lanes`). The parent keeps it as the port's
  `budget`.
- **Stated in the HELLO.** The HELLO a bridge sends on a link's control socket carries `count` = 1 + its budget.
  `hello_budget(count)` reads it back, and a `count` of 0 is a bridge before budgets, which makes all
  `MESH_LINK_LANES`. The multicast discovery hellos are unchanged.
- **The plan.** At the greeting each end computes `plan = link_plan(min(own budget, peer's budget))`, a bit a lane of
  the existing layout. Both ends compute it from the same two numbers. It goes to the agent with AGENT_MAKE (`lanes`).
  The agent makes one queue pair per lane of the plan (`pair_up`), tells them in AGENT_MADE, connects only those, and
  serves `served = plan`. The LINK names the plan's lanes in `count` and is checked against this end's plan.

| budget (the fewer of the two ends') | plan | lanes | streams | keepalive lane |
|---|---|---|---|---|
| 0, 1 | 0 | 0 | 0 | off: the link pairs, records flow, no stream is live |
| 2 | 0xc | 2 | 1 | off |
| 3, 4 | 0xd | 3 | 1 | on |
| 5, 6 | 0x3d | 5 | 2 | on |
| 7, 8 | 0xfd | 7 | 3 | on |
| 9 | 0x3fd | 9 | 4 | on (no reserved lane 1) |
| 10 and more | 0x3ff | 10 | 4 | on, lane 1 made as before |

The keepalive lane comes first once a stream fits beside it. The reserved lane 1 is made only in the full plan, so a
full plan is lane for lane the layout of MESH_VERSION 131. The plan of a smaller budget is always a subset of a
larger one's.

- **Streams.** Streams beyond the plan's are never opened: not live, not free, never offered to a client
  (`mesh_streams_free`), and never in the node's reach (`mesh_nodes_reached`). Each live stream reports
  `bandwidth / MESH_LINK_STREAMS` as before, so the map's per-link figure is comparable across plans.
- **Completion queues.** Each lane's share of the two completion queues is `cqe / lanes`, so a two-lane link's lanes
  get 2048 frames each, against 409 on a ten-lane link. The receive window stays capped by the ring's units and the
  queue pair's capacity.

**What a two-lane link costs.** Node 521750041's link to node 1974797816 gets the 0xc plan: one stream and no
keepalive lane.
- One communicator at a time between the two nodes. A second concurrent one is refused at its init with EBUSY as a
  value ("rank 0 could not hold its streams at its bridge"). The first runs exact.
- No keepalive frames, so a controller that slows after a quiet stretch (etc/bridge/README.md, the Mac mini's 6-10 us
  after 40 us of nothing) is not kept warm. This costs latency on the first frames after a pause, not correctness.

**Version.** MESH_VERSION stays 131 and MESH_LAYOUT stays 133. The full plan is byte for byte the 131 wire (the
HELLO's `count`, which 131 sends as 0 and never reads, is the only new information), and the region's layout is
unchanged. So:
- a bridge of this branch and a 131 bridge whose devices both hold ten pair as two 131 bridges do (tested, section 7);
- a bridge of this branch holding fewer than ten and a 131 bridge cannot pair. The 131 end makes ten and names ten. The
  new end logs "node N's LINK names 10 lanes, this end's plan 2 (plan 0xc): its bridge states no budget and makes every
  lane, and this port holds fewer: the link pairs once that node runs a bridge that states its budget", and both re-pair
  every 5 s as they do today.

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

**MESH_VERSION.** `XMAGIC` adds MESH_VERSION to the magic, so bridges of different versions drop each other's
hellos. The map parts into one per version until every node runs the newer one (README, "A new bridge on a node").
For versions to join one map:
1. *Extend in place* (done here). New information goes in a field the older version sends as 0 and never reads, with 0
   meaning the older behavior. The older version's exchanges keep their meaning, so MESH_VERSION does not change.
   That covers additive changes like this one.
2. *A version-stating hello* (not done). A version-independent header (a fixed magic, then the sender's
   MESH_VERSION), so a bridge hears an older peer instead of dropping it. Where it still speaks the older wire, it
   pairs at the older version; where it does not, it logs which version the peer runs, so the operator sees a
   version split instead of a silent one. The first bridge with this header still needs a MESH_VERSION bump, since a
   131 bridge drops it.
3. *Records only appended.* `struct mesh_record` and the region's header and slots are appended to, never reordered,
   as the region rule already requires. A newer bridge then floods records an older one can still read up to its
   own length.
4. *The region stays apart from the wire* (already so: MESH_LAYOUT), so a wire change never expires a region.

## 7. Tested

At the operator's request of 2026-10-21 (AGENTS.md's suspension of runtime testing of 2026-10-14 otherwise holds).
All of it ran on node 1980260065 over loopverbs (shared memory; no verbs device was opened). `make -C rdma all` and
`make -C rdma host` build with every warning an error. `loopverbs.c` takes `LOOPVERBS_MAX_QP` (entries
`rdma_loop<i>=<max_qp>`) so the two ends of a loop cable can differ, refusing a device's queue pairs past
`max_qp - 1` as a TB5 device does. With `mesh-loop`:

| case | what came up | collectives |
|---|---|---|
| 2 nodes, max_qp 11 / 11 | "10 lanes up (plan 0x3ff: 4 streams, keepalive on), each lane's share 409 frames" | host 65,536 floats: 16 of 16 exact |
| 2 nodes, 3 / 11 (node 521750041's case) | "2 lanes up (plan 0xc: 1 stream, keepalive off), each lane's share 2048 frames" at both ends | host 65,536: 16 of 16 exact; walk 1,000,003: exact; two communicators started together: one exact, the other refused at init with EBUSY |
| 2 nodes, 3 / 3 | 0xc at both ends | walk 1,000,003: exact |
| 3-node line, 5 / 11 and 10 / 11 | cable 0 "3 lanes up (plan 0xd: 1 stream, keepalive on)", cable 1 "9 lanes up (plan 0x3fd: 4 streams, keepalive on)" | 27 of 27 relayed readings exact on both paths; the ends' sendrecv refused as `mesh-loop.c` says it must be (no cable joins them) |
| 2 nodes, 2 / 11 (budget 1) | "paired ... plan 0, 0 lanes, 0 streams", "0 lanes up", records flowing, no loss in 60 s | none (no stream) |
| this branch / main's 131, max_qp 11 / 11 | 131's end: "10 lanes up"; this end: "its bridge stating none; plan 0x3ff", "10 lanes up" | host 65,536: 16 of 16 exact |
| this branch at 3 / main's 131 | the logged mismatch above, re-pairing every 5 s | none |

The 3-node line also showed a fault that predates this change: rank 1's sendrecv waits without bound on the host
path, and returns "ok" with every value wrong on the walk path, once the ranks it pairs with have refused theirs. The
same line at default budgets on this branch behaves identically (section 8, P6).

## 8. What remains

- **P1. Measure what a 27 device holds.** The budget assumes `max_qp - 1`. To measure it, with node 521750041's
  bridge released (`launchctl kill SIGUSR1 user/$(id -u)/io.mesh.bridge`, the operator's window) and rdma_en3's link
  up, run one probe process that opens rdma_en3, allocates a PD and two CQs, and makes UC queue pairs until refused.
  Then do the same in a second context of the same device (per context or per device), then on a second device with
  its link up (pooled across a machine's devices or not). No release call; exit releases everything. The second
  Studio, cabled, gives a second device with its link up. If 3 are usable, `max_pairs` for this driver is `max_qp`,
  and the plan gains the keepalive lane (0xd).
- **P2. Make until refused.** The device, not the stated budget, has the last word. Each end could make its plan's
  queue pairs in the order 2, 3, 0, 4, 5, 6, 7, 8, 9, 1, so that the first c made always contain `link_plan(c)`.
  It would stop at the first refusal and name in LINK how many it made, and both ends would connect
  `link_plan(min(made, peer's made))`. Queue pairs made past that stay in INIT until the agent's exit. Two 131-
  compatible ends both make ten, so the wire stays compatible. This removes the reliance on `max_qp - 1` for any
  hardware.
- **P3. More streams on few queue pairs.** If P1 finds a per-context count, an agent can open more than one context on
  its device and spread a link's lanes over them, each context with its own two CQs (max_cq 6, one the kernel's,
  bounds it to two contexts). The progress threads would poll both pairs of CQs. Otherwise, more streams on two queue
  pairs would need a staging ring and a copy (section 4), which this design has refused so far.
- **P4. Publish the plan.** `struct mesh_link_info` (the bridge's observation pages) has no room for the plan without a
  MESH_LAYOUT bump, so mesh-stat cannot show it yet; the log line does. A record's `reach` could carry each link's
  stream count so a planner knows how many communicators a node pair takes at once.
- **P5. A keepalive on a two-lane link.** For example, a one-frame small-class unit sent by the bridge on the stream
  while no holder holds it, agreed by both ends as a unit no client reads. Only if the controller's slow start
  matters on that link.
- **P6. The line harness's unbounded sendrecv** (section 7). It predates this change and is a library and harness
  issue, not the transport's.
- **P7. Host names in the bridge's area** (AGENTS.md debts, not changed here). `mesh-link.c` `host_name` publishes
  SCDynamicStoreCopyLocalHostName as the record's `name`, which mesh-peers merges announcements by, so replacing it
  means changing those consumers together: publish the host UUID (or the hardware serial) there and let readers map
  identity to a person's name. `nccl-mesh.c` `ncclGetUniqueId` falls back to `gethostname()` for the bootstrap's
  listen host when the node holds no fabric address. It should fall back to an address the node holds instead
  (a global or link-local IPv6 of an up interface). Both changes reach beyond this branch's links and are left for
  their own review.

## 9. Deploying it

The change is in `rdma/mesh-bridge.h`, `rdma/mesh-verbs.h`, `rdma/mesh-link.c` (the bridge binary, `mesh-flow`) and
`rdma/loopverbs.c` (the loop build only). Neither libmesh nor libnccl-mesh changes, so clients need nothing. Each
node needs its own build of the merged source: `make -C rdma all install` by hand, then its heartbeat's SIGHUP
(`bin/mesh-user.sh`), which restarts the bridge, keeping its region (MESH_LAYOUT unchanged).

- Node 521750041 (M5 Ultra, 27): required. Until node 1974797816 also runs it, its rdma_en3 link keeps re-pairing every
  5 s, now with the cause logged.
- Node 1974797816 (M4 Pro): required for the link to node 521750041. Its link to node 1980260065 is unaffected either
  way: both hold ten, and the full plan is the 131 wire.
- Node 1980260065 (M5 Max): optional, at any time.

Either order works. Rebuilding node 1974797816 restarts its bridge, so the link to node 1980260065 drops for one
re-pair (the usual cost of a SIGHUP restart). Do it with no program running across that link. Once both node
521750041 and node 1974797816 run it, their link should log "paired with node ... (queue pairs for the link: this
port's 2, its 10; plan 0xc, 2 lanes, 1 stream)" and "2 lanes up".
