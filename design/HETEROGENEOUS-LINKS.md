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
  (`mesh_streams_free`), and never in the node's reach (`mesh_nodes_reached`). Each live stream reported
  `bandwidth / MESH_LINK_STREAMS` as before, so the map's per-link figure was comparable across plans (section 10
  makes each stream's share the link's over its live streams, and the reach entry the link's whole figure with its
  stream count).
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
- **P2. Make until refused.** Done within the plan (section 10): each end makes its plan's queue pairs in the order 2,
  3, 0, 4, 5, 6, 7, 8, 9, 1, stops at the first refusal, names in LINK how many it made, and both ends connect
  `link_plan(min(made, peer's made))`. The plan itself is still made from the stated budgets (`max_qp - 1`), so a
  device that holds more than it states (P1's 3 usable) is not found by it: making past the stated count would try,
  on the M5 Ultra, a queue pair no process has yet made there, which no probe has run.
- **P3. More streams on few queue pairs.** If P1 finds a per-context count, an agent can open more than one context on
  its device and spread a link's lanes over them, each context with its own two CQs (max_cq 6, one the kernel's,
  bounds it to two contexts). The progress threads would poll both pairs of CQs. Otherwise, more streams on two queue
  pairs would need a staging ring and a copy (section 4), which this design has refused so far.
- **P4. Publish the plan.** Done for the streams (section 10): a record's `reach` carries each link's stream count,
  and `mesh_map` and mesh-stat's map give it a link. `struct mesh_link_info` still has no room for the plan's lanes
  (the keepalive's) without a layout change; the log line has them.
- **P5. A keepalive on a two-lane link.** For example, a one-frame small-class unit sent by the bridge on the stream
  while no holder holds it, agreed by both ends as a unit no client reads. Only if the controller's slow start
  matters on that link.
- **P6. The line harness's unbounded sendrecv** (section 7). It predates this change and is a library and harness
  issue, not the transport's.
- **P7. Host names in the bridge's area** (AGENTS.md debts). The record's `name` (`mesh-link.c` `host_name`, which
  published SCDynamicStoreCopyLocalHostName) is gone: a record carries its host UUID, and readers take a person's name
  from the name table by node number (`peers.py` `label`). `nccl-mesh.c` `ncclGetUniqueId` falls back to
  `gethostname()` for the bootstrap's listen host when the node holds no fabric address. It should fall back to an
  address the node holds instead (a global IPv6 or IPv4 of an up interface).

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

## 10. Make until refused, and each link in the map (2026-10-21, the bridge review's B4 and B5)

Two faults the bridge review found in the change above, for meshes rather than pairs:
- **B5.** The budget assumed each device's queue pairs independent of the node's other links. A machine whose devices
  share them (unmeasured on macOS 27: P1; node 521750041's two links, to 1974797816 and to 455770905, both came up,
  weak evidence against sharing at 2 + 2) would refuse a later link's make past what is left; `pair_up` returned
  ENOSPC, and that link re-paired every 5 s for good.
- **B4.** The map could not hold two cables between two nodes, nor say how many streams a link has.
  `mesh_nodes_reached` listed each node once at its first live stream's bandwidth, `mesh_join` joined two records by
  one link, and every stream reported `bandwidth / MESH_LINK_STREAMS`: a one-stream link (the Studios' 0xc) and a
  four-stream one looked the same, a second communicator over the first was refused at init (EBUSY) with no sign in the
  map, and the planner's parallel links (`mesh_routes`, `mesh_trees_pack`) were never used.

**Make until refused** (`mesh-verbs.h` `pair_up`; `mesh-link.c` `link_make`, `link_connect`, `link_tell`,
`link_heard`, `link_turn`, `agent_command`, `agent_jobs`; `mesh-bridge.h` `lane_order`):
- The plan to make is as before, `link_plan(min(budgets))`. The agent makes its completion queues, then the plan's
  queue pairs in `lane_order` (2, 3, 0, 4, 5, 6, 7, 8, 9, 1), and stops at the device's first refusal: a refusal is a
  smaller make, never the link's failure. The agent tells the lanes made and the refusal (AGENT_MADE `lanes`,
  `refusal`), and the parent logs a make short of the plan.
- The LINK's `count` is the queue pairs made, `pair` naming each. A peer's LINK is taken where its count is at most
  the plan's; more is a bridge before budgets, which makes every lane, logged as before.
- Both ends connect `link_plan(min(made, peer's made))`, asked of the agent with those lanes (AGENT_CONNECT `lanes`);
  each lane's share of the completion queues is taken at the connect, over them. Queue pairs made past it stay in
  INIT until the agent's exit (no destroy, no state change: transport.md section 14).
- Why the two ends agree, for any k nodes and any links: the first c of `lane_order` hold `link_plan(c)`, each plan's
  lanes are the order's first `plan_lanes` of it, and `link_plan(plan_lanes(link_plan(b))) = link_plan(b)`. So the
  lanes each end made are the order's first `made` of the plan, `link_plan(min)` lies within both, and both compute it
  from the same two counts; where both made the plan whole it is the plan. Checked for every pair of budgets 0 to 11
  and every refusal point at each end. A link's plan is its own: a node's other links do not enter it.
- **Compatibility**, MESH_VERSION 131 kept. Where the device refuses none, the make is the plan whole and the LINK is
  byte for byte the budget bridge's, so this bridge pairs with main's wherever main's make would succeed, and with a
  bridge before budgets wherever both hold ten. Where the device refuses some, a budget bridge reads the LINK as
  planned apart and the link re-pairs, as its own ENOSPC did; two bridges of this change connect the smaller plan.
- Completion queues refused are a make of none: plan 0, the link paired and its records flowing, no stream.

**Each link in the map** (`mesh-call.c` `mesh_nodes_reached`; `mesh-dataflow.c` `mesh_join`, `mesh_joined`,
`mesh_whole_make`; `mesh-link.c` `streams_open`; `mesh-collective.c` `mesh_map_members`; `mesh-records.c`
`record_links_same`; `mesh-stat.c`):
- A reach entry is a link (a port) with a live stream: its node, its live streams (`struct mesh_reach` `streams`, the
  word that was `padding`) and their shares of its bandwidth together. Two cables toward one node are two entries.
- A stream's share is the link's bandwidth over its live streams (the remainder to the first ones), so a one-stream
  link's stream reports the link's whole bandwidth, and the entry's figure is the link's whatever its plan.
- Two records join by as many links as the fewer lists toward the other, each side's ranked by bandwidth then streams,
  the k-th with the k-th, each at the lesser bandwidth and the lesser streams either states (0 where neither does). The
  whole map lists each (`mesh_map`), and the planner takes a pair's parallel links together.
- `struct mesh_link_map` gains `streams` (NULL: none): `mesh_map` writes each link's, `mesh_map_members` carries it,
  mesh-stat prints it as each link's fourth figure, and no planner function reads it. A planner reads there how many
  communicators a node pair takes at once: its links' streams together.
- Not changed: a communicator's own map (`nccl-mesh.c` `map_of`) takes each pair's first reach entry, one link a pair,
  since a communicator holds one stream toward each member.

**Layout and ABI.** The region's bytes are unchanged (MESH_LAYOUT 133 kept): `struct mesh_record` is the same size, its
reach entries now one a link with `streams` in the former padding (0 from a bridge before it, read as none stated), and
a stream's `bandwidth` word its share over its link's live streams. The wire is unchanged in size; the LINK's `count`
reads as above. The client ABI changes: `struct mesh_link_map` has one more pointer at its end, so libmesh, its C
callers and Python's `_mesh_c` are built together (`make -C rdma all`, and `bin/mesh-user.sh`, which builds `_mesh_c`
at every deploy); `struct mesh_reach`'s second word is renamed `streams`, its size and offsets unchanged.

**Tested** over loopverbs on node 1980260065 (no verbs device opened). `LOOPVERBS_HOLDS=rdma_loop<i>=<n>` makes a loop
device refuse queue pairs past n while stating its max_qp as ever, as a machine whose devices share them would; the
loop harness's family `double` joins each neighbouring pair by two cables, and the harness counts a node converged at
one reach entry a cable.

| case | what came up | collectives |
|---|---|---|
| 3-node line, every device ten | "10 lanes up (plan 0x3ff: 4 streams)"; map: two links, 100 Gb/s and 4 streams each | host 65,536: 27 of 27 exact (the ends' sendrecv refused, no cable joins them; rank 1's wait unbounded, P6) |
| 3-node `double` line | four links, two a pair, 4 streams each | host: 27 of 27 exact, as the line |
| 2 nodes, one device holding 4 of its stated 10 | that end "4 of the plan's 10 queue pairs made (made 0x1d)", both "3 lanes up (plan 0xd: 1 stream, keepalive on)", each lane's share 1365 frames; map: 1 stream | host 16 of 16, walk 1,000,003 16 of 16 exact; two communicators at once: one exact, the other refused at init (EBUSY) |
| 2 nodes, two cables, every device holding 2 | two links "2 lanes up (plan 0xc)"; map: two links of 1 stream | three communicators at once: two exact, the third refused at init (EBUSY) |
| 2 nodes, two cables, one device holding 0, another 1 | both links "0 lanes up (plan 0)", records flowing | none (no stream) |
| 4-node ring, its cables' ends refusing at 6 / 10, budget 2 (max_qp 3) / 10, 10 / 10, 3 / 5 | plans 0x3d, 0xc, 0x3ff, 0xd; map: 2, 1, 4, 1 streams | host 65,536: 48 of 48; walk 1,000,003: 48 of 48 exact |
| this change / main's, both ten | "10 lanes up" at both | host: 16 of 16 exact |
| this change at max_qp 3 / main's at ten, and the reverse | 0xc at both ends | walk 16 of 16, host 16 of 16 exact |
| this change holding 4 of 10 / main's at ten | main's end "node 1's LINK names 4 lanes, this end's plan 10: the two ends planned apart", re-pairing each agent cycle | none |

**Still open.** A make refused short holds its smaller plan for its agent's life: a queue pair freed later (another
link's agent ending, where devices share them) reaches this link only at its next pairing. A link of plan 0 carries
records but no stream and is no link of the map, so a node reached only over such links is pruned from the map though
its records flow (`mesh-records.c` `map_put` joins records by their reach).

## 11. A stream over several cables (2026-10-21)

The fabric is recabled into rings and tori, so two nodes may be joined by more than one cable, and every cable must
carry load. Section 10 listed each cable as a link of its own: a client's stream was one cable's (a link's four slots,
`mesh_streams_free`, the least identity free at both ends), so a communicator moved at one cable's bandwidth however
many joined the two nodes. The rubric's division (design/rubrics.md, Three kinds of source) answers where that belongs:
a client "holds a stream toward each member's node" and reaches no port, link, cable or lane, so the bridge spreads
each stream over the cables (design/transport.md section 14, "A stream over several cables").

- **Each stream over every cable.** Stream s toward a node stripes its units over lane pair s of every cable joining
  the two whose plan has it (a group: `mesh-link.c` `struct group`, `group_open`). Unit u rides stripe u mod n, as its
  lane's (u / n)-th from its base, the stripes ordered by each cable's identity for that stream (`stream_wire`, alike
  at both ends). The leader's send thread serves the holding, each member's send thread its stripe, and any member's
  receive thread releases the units in order (`group_turn`, `lane_send`, `group_release`). The stream lives in the
  leader's slot. A client takes it, views it, declares on it and gives it back as before: libnccl-mesh is unchanged.
- **The map names the pair once.** A record's reach is a node an entry: every cable whose link is up toward it, their
  bandwidths together, and the most streams any one plan holds (`record_reach`). `mesh_join` reads one link a pair at
  the lesser of the two ends' figures, any entries naming the node summed (`mesh_paired`). `mesh_nodes_reached` (the
  clients' view) lists a node once, its live streams and their bandwidths together. The pair's link carries what
  section 10's parallel links did, at its cables' bandwidth together. Concurrency between two nodes is their streams:
  the most one plan holds, not the sum. Two one-stream cables (the M5 Ultras) carry one stream at both cables'
  bandwidth, where section 10 gave two streams at one cable's each.
- **Cables change.** Where the cables of a group change, both ends agree, once, on the count at which each cable's
  lanes change mapping (STREAM: design/transport.md section 3 and section 14). That count is fixed by the sender's SENDs
  posted and the receiver's receives posted together, which neither end holds alone. Each end pads with duds and
  absorbs into the link's sink to the bases both compute, then opens the group anew. A cable lost fails the calls in
  flight on it; the group goes on over the others, with no re-pairing. A held slot cancelled waits MESH_PLATFORM_NS
  before it opens again. A cable that joins is taken at the next holding. Cables all fresh (one per peer, or several
  pairing together) open at once from bases 0, so one cable per peer pairs and serves as before, its wire unchanged.
- **Waking.** A port's progress threads waited on the guard word while its link served no lane, so a link that came up
  between a thread's read of its lanes and its wait slept MESH_PAUSE_NS before its first pass. They now wait on the
  lanes' word while the guard is down, and `guard_wake` wakes both. Before this change that delay held a new link's
  first SENDs; with groups it would hold each group's freeze.

**Wire, layout and ABI.**
- Wire: a message kind, STREAM (5), whose payload is `struct stream_told` (104 bytes). A bridge before it reads STREAM as
  a protocol error and loses the link, so where two cables join two nodes the two versions re-pair each cycle.
  MESH_VERSION is not moved here: the integration bumps it once. One cable per peer sends no STREAM, and its data path is
  lane for lane the earlier one's. A bridge of this change and the base's (`kmesh-integration-b`, 131) over one loop
  cable ran host and walk 16 of 16 exact.
- Region: each stream slot's bridge words (`struct stream_lanes`, 1 KiB) after its large class's landed words, in the
  block those round to on 16 KiB pages, so a port's ring lease is the size it was (MESH_LAYOUT kept). On smaller pages
  the lease grows, and a region an earlier bridge leased on them would be too small. Every Apple silicon node's page is
  16 KiB. A record's reach entries are one a node (the record's size and layout are unchanged). A stream's
  `bandwidth` is its group's (its cables' shares for that stream together).
- Client ABI: none. `mesh_nodes_reached` and `mesh_map` keep their signatures; their entries are a node and a pair.
- Not changed here, others' files: `rdma/mesh-plan.h` (`mesh_map`'s comment still says two cables are two links, and
  `struct mesh_link_map` `streams` still says a pair's links' streams together) and `rdma/NCCL.md` ("two cables between
  two nodes have their streams together") belong to the collective layer's branch.

**Tested** over loopverbs on node 1980260065 (no verbs device opened), each bridge's shared memory under a private
prefix (`MESH_LOOP_PREFIX`, mesh-loop):

| case | what came up | collectives |
|---|---|---|
| 2 nodes, two cables, every device ten | each of 4 streams over 2 cables (50 Gb/s a stream, 200 together); a second cable joining made the group anew in 3 ms | host and walk 1,000,003: 16 of 16 exact each; on the stream used, 3,060 units released, 1,530 on each cable |
| 3-node doubled line | each neighbour pair's 4 streams over its 2 cables | 27 of 27 exact between joined ranks, host and walk; the ends' sendrecv refused (no cable joins them) and rank 1's sendrecv as section 7's P6, unchanged |
| 2 nodes, two cables, every device holding 2 (the M5 Ultras) | one stream over both (200 Gb/s) | host and walk 1,000,003: 16 of 16; two communicators at once: one exact, the other refused at init (EBUSY) |
| 2 nodes, a 1-stream cable and a 4-stream cable | stream 0 over both (125 Gb/s), streams 1 to 3 over the 4-stream cable | host and walk 16 of 16; five communicators at once: four exact, the fifth refused at init |
| 4-node ring, one cable a peer | no group line logged; streams as before | host and walk 1,000,003: 48 of 48 exact |
| one cable, this bridge and the base's | paired as two 131 bridges | host and walk 1,000,003: 16 of 16 |
| two cables, a cable's agent killed while idle | the streams made anew over the other cable in 3 ms, no re-pairing; the cable back, the streams over both again in 3 ms | host and walk 1,000,003 between: 16 of 16 each |
| two cables, the leader's cable cut under 30 back-to-back communicators | the call in flight failed ("the link to node 1 was lost at position 6", ncclRemoteError); the streams over the other cable 2 ms after | each later communicator's 16 of 16 exact (22 runs) |
| the same, the other cable cut | the call in flight failed at position 99; three streams open over the remaining cable in 3 ms, the held one after its slot's 5 s; communicators took the free ones | 53 later runs, 16 of 16 each; the cable back, the free streams over both at once, the held one at its holder's end |
| the M5 Ultras' two cables, the leader's cut under back-to-back communicators | the call in flight failed at position 173; the stream over the other cable 1 ms after | 32 later runs, 16 of 16 each |

