# Bridge configurations

A node's bridge (`rdma/mesh-flow`) starts from one operand, its configuration: a document its consumer writes and
passes, in JSON, TOML or YAML. `bin/mesh-user.sh CONFIGURATION [CHECKOUT]` reads it with the mesh library's one
loader (`rdma/bridge_configuration.py`), writes its canonical form (a property list of its name, purpose and region
in bytes, `~/Library/Application Support/io.mesh.bridge/configuration.plist`) and names that in the bridge's launchd agent
definition, so every load of the agent reads it, and records the document's path beside it. A
changed configuration is renamed into place, and `mesh-user.sh` sends the heartbeat SIGHUP: the bridge restarts on
it, its region kept (README.md, A new bridge on a node). Calls in flight across the restart fail as values, and the
next communicator runs on the new bridge (design/transport.md section 2). There
is no default anywhere in the code: without a configuration the bridge and `mesh-user.sh` refuse to start, naming what
they need.

Servicing a node (design/transport.md section 2, the heartbeat): the bridge's job runs its heartbeat, which opens and
closes the bridge.
- `launchctl kill SIGUSR1 user/$(id -u)/io.mesh.bridge` releases the bridge and its region: every device object
  freed at once, and the region's memory once none of its holders lives (each client's next lease or communicator returns the
  expiry; the heartbeat's log names the processes that still hold it);
- `launchctl kill SIGHUP user/$(id -u)/io.mesh.bridge` opens it again (once a released region is unlinked), or
  restarts it keeping the region's object (a new layout is laid over its pages; a changed region size takes effect only
  after a release);
- `launchctl kill SIGTERM user/$(id -u)/io.mesh.bridge` releases it and ends the heartbeat, which launchd then leaves
  stopped until `launchctl kickstart user/$(id -u)/io.mesh.bridge`.

A document has these keys:

- `purpose`: one line stating what the configuration is for (the loader raises `ValueError` without it,
  rdma/bridge_configuration.py).
- the region, exactly one of `region_gib` (GiB) or `region_percent` (of the node's physical memory).

The region is the bridge's pinned memory (design/transport.md). Every port's device registers all of it
when the device comes up, before any queue pair exists. The bridge does not deregister it: the kernel releases the
registration at the port's child's exit (design/transport.md section 5). Each device registers the region in extents
of 2^k bytes: the least power of two, from the arena's page up to the 4 GiB bank, at which the region's extents number
no more than that device's `max_mr` (rdma/mesh-link.c:538-541, `device_up`; design/transport.md section 1). On the
Studios' 86 GiB under `max_mr` 100 that is 1 GiB. The bank is the hardware's fact: `tbt_post_recv` keeps only the low
32 bits of an SGE address, so no extent crosses a 4 GiB boundary. Every input is the node's own (the region's size,
`max_mr`, the bank), so the count of extents fits `max_mr` by construction (02002590). A registration that fails
returns its errno as the port's open's value: the port is not open, the extents already registered stay, and it opens
again at the events that open any port, its link's coming on, a hello heard on it or its child's end (rdma/mesh-link.c
`port_opened`, line 895; 02002590 deleted the refusal's backoff). The bridge starts and serves the node either way;
the region is not clamped. Clients lease from it; a lease that does not fit fails at its start with an error naming
the execution, and the region does not grow.

## The examples

These two ship with the bridge to show the form. Neither is likely to be what an application wants; both are for
didactic programming exercises.

- `example-1pct.json`: one percent of the node's memory (245 MiB on a 24 GiB Mac mini, 1.3 GiB on a 128 GiB M5). It
  allows small streaming transfers. A libnccl-mesh communicator leases one page for its gate, and a pool for the
  pages its groups send from and land in that are not the caller's (a reduce-scatter's operand, a tree's scratch, a
  combine's landing area; rdma/nccl.h, Memory). Every operand that crosses a link is the program's own lease. It does
  not allow programs that keep large operands in the region.
- `example-10pct.yaml`: ten percent of the node's memory. It holds the operands of many crossings, and it is too
  small for many applications: a model's weights and KV pool in the region, or a long reduction's operands.

The operator, 2026-10-15 (below): "with the proper bridge configuration for their application deliberately left for
the user to derive".

## The E2B pair under test

These two are written for the source as it stands (metal-microbench d27fcd9): the rank leases its weights, its KV
pool and every buffer a crossing takes from the region (metal-microbench docs/placement.md#memory; the operator,
2026-10-15: "why would weights or kv be outside of the memory pinned pool? being outside of it means that they cannot
be zero copy passed to peers"). Until metal-microbench 773cead the rank kept them outside it, and a region beside them
overcommitted the Mini (M09: at a 9 GiB region its userspace starved and its watchdog reset it). What a member leases
is one computation, metal-microbench tools/mesh/programs.py `plans` of the call's record: the planner derives no call
whose members' leases their regions do not hold (node-info `region`) and the driver places no finished record that does
not fit, each call's record carries each member's, and the rank holds its part's load to the record's to the byte and
its calls within the record's total. The same count from the model file's header (2026-10-15, before programs.py
plans was first run) gives the leases lease5's ranks took before their Neural Engine shares to the byte.

- `m4pro-headless.toml`: the Mac mini, 16 GiB, the most that left it a 2 GiB margin beside the region's tables of
  MESH_VERSION 110 (1.11 GiB); beside 111's (0.06 GiB) it leaves 3.47 GiB, and 17 GiB would leave 2.46.
- `m5-low.toml`: the M5 Max, 14 GiB, below the Mini's: the least whole GiB that holds its rank's leases and the whole
  model alone.

Each file's `purpose` states its arithmetic from that computation for the pair's heads programs of 2026-10-15
(2057c7fae8c1930a and d5a4040e726d5b44, the member's larger part in each), counted that way from the model file's
header (2026-10-15, before programs.py plans was first run): its weights (a Neural Engine share's gate, up, down and
norm are its layer's own buffers, the down read at the layer's row pitch), the KV pool at the engine's whole page geometry
(16000 pages and the strip, with their chunk tables; a call leases its own pages, 36 MiB for each 2048 positions of a
row), the crossed capacity set, the conversion staging, a call's per-layer inputs, its communicator's session (the
library's, beside the computation), and the whole model alone. Beside the arena: the region's tables (64.5 bytes an
arena page, a row's page-table entry, buffer record and delivery word and its plane bits, and 0.5 MiB of header,
ports and client slots, rdma/mesh.h `mesh_layout`, from MESH_VERSION 111: 0.06 GiB at 16 GiB, 0.056 at 14; at 110 every row
also carried a publication's targets, 1,136.5 bytes an arena page and 0.5 MiB, 1.11 GiB at 16 GiB, touched as rows were leased; from
111 a program's targets followed its SEND cells in their lease; at MESH_LAYOUT 135 there are neither, and a row has no delivery
word: its page-table entry, buffer record and plane bits, 48.5 bytes an arena page on one-page rows), the node's base wired and macOS
memory, the rank's process, the Neural Engine's own copy of its columns, and a 2 GiB margin. A served engine (metal-microbench server/bridge.py) leases its model and pool
from its node's region too; neither file holds one.

## The Studios

Two documents are written for the four Mac Studios (M5 Ultra, 96 GiB, headless) that join the pair, one sized as the
Mini's region is and one as the M5's.

`m5ultra-headless.toml`, sized as the Mini's (the operator, 2026-10-15: "allocate essentially all of the memory on the
entire device"): the most whole GiB that leaves a 2 GiB margin, 82 GiB. Beside it
the node keeps 2.98 GiB: the region's tables (0.323 GiB, 331.1 MiB), base wired (2.61, the Studios' reading on
2026-10-23), macOS's anonymous memory (1.5), the rank's process (3.64, wired at the chain's run in m5ult2's panic of
03:08 that day) and the Neural Engine's own copy (2.95); 83 GiB would leave 1.97. At 86 GiB those readings left -1.02,
and both Studios reset. It holds 26B-A4B whole at the engine's whole page geometry (81.11 GiB) on one Studio and 12B's
(97.18 GiB) on any two members, and the six regions (four Studios at 82, the M5 at 14, the Mini at 16) hold 358 GiB.

`m5ultra-low.toml`, sized as the M5's (the operator, 2026-10-15, below: "the m5 system could even have a m5_low
configuration that allocates less buffer than a m4 pro headless machine, since it has higher flops, higher bandwidth,
and benefits less from memoization than a flops starved and bandwidth starved peer"):
its region is the least whole GiB
that holds the whole E2B model alone at the engine's whole page geometry, 14 GiB (13.27 GiB counted by programs.py `plans`
of a membership of one from the model file's header; 10.56 on the streamed codes export), and any part of E2B on any
membership holds in it. Beside it the node keeps 71.24 GiB: the region's tables (0.056 GiB), base wired (2.61, the Studios'
reading), macOS's anonymous memory (1.5), the rank's process (3.64) and the Neural Engine's own copy
(at most every FFN column at fp16, 2.95). The Mini's sizing, the most that leaves a 2 GiB margin, gives 82 GiB. The larger
models alone at a call's own pages take more than 14 GiB (12B 16.46 GiB at 4,096 positions, 26B-A4B 28.25 GiB): a Studio
holds its part of them on memberships whose regions together hold the whole, and the planner fits each member's part to
its region; at the engine's whole page geometry they take 97.18 GiB (12B) and 81.11 GiB (26B-A4B), more than the six
regions at these documents hold together (four Studios at 14, the M5 at 14, the Mini at 16: 86 GiB). Counted with
metal-microbench tools/mesh/programs.py under `MESH_MODEL=e2b-engine` (12b, 26b-a4b):
`plans({'bs': 1, 'ctx_max': 256000}, structure(), pages=16000)[0]['total']`, and at a call's pages
`pages=pool_request(1, 4096)` with `ctx_max` 4096 (metal-microbench output_data/lifecycle-20261015/studio_window.py).

The region is the operator's: "will the user ever 'give permission' to pin 96gb of memory preemptively on a userspace
machine for no reason? they *might not*" (2026-10-15, below). Each document states the arithmetic it follows from. A Studio's base wired
memory and its devices' `max_mr` are read when it arrives: the pair's devices report `max_mr` 100, and a region
registers as 2 GiB extents and a family shifted by half an extent (design/transport.md), 62 at 82 GiB, and a device
that registers fewer than its region's extents pairs no port until the region is lowered (its refusal logged with the
extent, above); the region is not clamped. Each of a node's devices registers the whole region, so a Studio cabled on
several ports registers it once a device.

## The operator's words

2026-10-15, on configuration: "in this situation we must learn to write configurations which are passed as operands by
consumers. where does a configuration get stored? Well it can be serialized in memory by any means. it can be stored as
a json, or a toml, or a yaml, or we can require all 3 be supported because we are specifying 'configuration as in the
behavior of the function is defined by an operand rather than by hardcoding or guesswork and definitely not by
ritualistically putting an argparse cli string into every single function call which breaks all of our shit if some
part is omitted'. so in such circumstances, you shoudl stop and ask. will the user ever 'give permission' to pin 96gb of
memory preemptively on a userspace machine for no reason? they *might not*. in fact, you shouldn't try to guess and
shouldn't try to hardcode a default, instead writing a configuration which is sensible for the model being tested. the
m5 system could even have a m5_low configuration that allocates less buffer than a m4 pro headless machine, since it
has higher flops, higher bandwidth, and benefits less from memoization than a flops starved and bandwidth starved peer."

2026-10-15, on default files: "a default configuration can be a defaultfile which is packaged with the bridge that
demonstrates a 1% pinning in one default file, and a 10% pinning in another default file, with a readme document to
help the user understand that a 1% pinning will allow small streaming transfers but not some kinds of programs, and
that a 10% pinning will be too small for many kinds of applications, with the proper bridge configuration for their
application deliberately left for the user to derive, as both the 1% and 10% configurations are very unlikely to be
what they want for anything but didactic programming exercises, emphasizing a disciplien of 'client writer knows what
default they will use for their mesh, and writes it explicitly instead of ignoring their basic responsibilities wrt
meshes they're paying thousands of dollars to scale out'".
