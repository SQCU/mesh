# Bridge configurations

A node's bridge (`rdma/mesh-flow`) starts from one operand, its configuration: a document its consumer writes and
passes, in JSON, TOML or YAML. `bin/mesh-user.sh CONFIGURATION [CHECKOUT]` reads it with the mesh library's one
loader (`rdma/bridge_configuration.py`), writes its canonical form (a property list of its name, purpose and region
in bytes, `~/Library/Application Support/io.mesh.bridge/configuration.plist`) and names that in the bridge's agent
definition, so the keeper's load at boot keeps it, and records the document's path beside it, so the node's
convergence (`bin/mesh-converge.sh`, etc/converge/README.md) installs each upstream revision with the node's own. A
changed configuration is renamed into place and takes effect when the bridge stops itself, between calls (README.md,
Convergence): its running calls are never ended for it. There
is no default anywhere in the code: without a configuration the bridge and `mesh-user.sh` refuse to start, naming what
they need.

Servicing a node (design/transport.md section 2, the heartbeat): the bridge's job runs its heartbeat, which opens and
closes the bridge.
- `launchctl kill SIGUSR1 user/$(id -u)/io.mesh.bridge` releases the bridge and its region: every device object
  freed at once, and the region's memory once none of its holders lives (each client's next lease or communicator returns the
  expiry; the heartbeat's log names the processes that still hold it);
- `launchctl kill SIGHUP user/$(id -u)/io.mesh.bridge` opens it again (once a released region is unlinked), or
  restarts it keeping the region (a changed region size replaces it, as a release does);
- `launchctl kill SIGTERM user/$(id -u)/io.mesh.bridge` releases it and ends the heartbeat, which launchd then leaves
  stopped until `launchctl kickstart user/$(id -u)/io.mesh.bridge`.

A document has these keys:

- `purpose`: one line stating what the configuration is for (required).
- the region, exactly one of `region_gib` (GiB) or `region_percent` (of the node's physical memory).
- `keepalive_us` (optional): the most microseconds an exclusive flow goes without a frame while it runs
  (design/transport.md): each such flow takes one queue pair more, and its send side posts a one-frame
  SEND on it that often, so a link's controller that slows after a quiet stretch (the Mac mini's, 6-10 us after 40 us of
  nothing) stays ready. Both ends of a flow state it for its frames to flow; a document that states none makes no such
  queue pair.

The region is the bridge's pinned memory (design/transport.md). Every port's device registers all of it
when the device comes up, before any queue pair exists, and deregisters it only when the device retires. Each device
checks the region against its own `max_mr` (1 GiB extents) as it comes up, on its port's queue: a region of more
extents than that device registers is refused for that port, logged with the extent, its bytes, the errno and the
extents the device holds, and the port turns again 60 s later, doubling to 30 min, keeping its extents
(design/transport.md). The bridge starts and serves the node either way; the region is never clamped. Clients lease
from it; a lease that does not fit fails at its start with an error naming the execution, and the region never grows.
How much an execution needs, on its own node and on the nodes it peers or streams with, is the client's to know.

## The loopback fabric's configurations

A loop bridge (`rdma/.build/loop/mesh-flow-loop`, mesh design/transport.md), one of N on one
host for running relays, forwarding and the library on 3 to 8 nodes before the hardware has them, reads the same
canonical form with one key more, `loop`: its region's name, its node, the loopverbs fabric it shares with the host's
other loop bridges, and its ports, each a loopverbs device and its cable's
two loopback ports. The loop's driver writes them (`rdma/.build/host/mesh-loop up`, one a node, from the map it is
given), not `bridge_configuration.py`, which refuses the key; the deployed bridge reads no `loop` (it is built without
the loop fabric) and a loop bridge refuses a configuration without one.

## The examples

These two ship with the bridge to show the form. Neither is likely to be what an application wants; both are for
didactic programming exercises.

- `example-1pct.json`: one percent of the node's memory (245 MiB on a 24 GiB Mac mini, 1.3 GiB on a 128 GiB M5). It
  allows small streaming transfers: a libnccl-mesh session toward one peer leases 24 MiB (its rings, its SEND cells
  and after them in the same lease its publication targets, its completion words and its transfer lists). It does not allow programs that stage large operands in
  the region.
- `example-10pct.yaml`: ten percent of the node's memory. It holds many sessions' crossing rings, and it is too small
  for many applications: a model's weights and KV pool in the region, or a long reduction's staging.

The proper configuration for an application is the client writer's to derive and write explicitly for each node of
their mesh.

## The E2B pair under test

These two are written for the source as it stands (metal-microbench d27fcd9): the rank leases its weights, its KV
pool and every buffer a crossing takes from the region (metal-microbench docs/placement.md#memory; the operator,
2026-10-15: "why would weights or kv be outside of the memory pinned pool? being outside of it means that they cannot
be zero copy passed to peers"). Until metal-microbench 773cead the rank kept them outside it, and a region beside them
overcommitted the Mini (M09: at a 9 GiB region its userspace starved and its watchdog reset it). What a member leases
is one computation, metal-microbench tools/mesh/programs.py `plans` of the call's record: the planner derives no call
whose members' leases their regions do not hold (node-info `region`) and the driver places no finished record that does
not fit, each call's record carries each member's, and the rank holds its part's load to the record's to the byte and
its calls within the record's total. A count of its rule from the model file's header (2026-10-15, before programs.py
plans was first run) gives the leases lease5's ranks took before their Neural Engine shares to the byte.

- `m4pro-headless.toml`: the Mac mini, 16 GiB, the most that left it a 2 GiB margin beside the region's tables of
  MESH_VERSION 110 (1.11 GiB); beside 111's (0.06 GiB) it leaves 3.47 GiB, and 17 GiB would leave 2.46.
- `m5-low.toml`: the M5 Max, 14 GiB, below the Mini's: the least whole GiB that holds its rank's leases and the whole
  model alone.

Each file's `purpose` states its arithmetic from that computation for the pair's heads programs of 2026-10-15
(2057c7fae8c1930a and d5a4040e726d5b44, the member's larger part in each), counted by that rule from the model file's
header (2026-10-15, before programs.py plans was first run): its weights (a Neural Engine share's gate, up, down and
norm are its layer's own buffers, the down read at the layer's row pitch), the KV pool at the engine's whole page geometry
(16000 pages and the strip, with their chunk tables; a call leases its own pages, 36 MiB for each 2048 positions of a
row), the crossed capacity set, the conversion staging, a call's per-layer inputs, its communicator's session (the
library's, beside the computation), and the whole model alone. Beside the arena: the region's tables (64.5 bytes an
arena page, a row's page-table entry, buffer record and delivery word and its plane bits, and 0.5 MiB of header,
ports and client slots, rdma/mesh.h `mesh_layout`, from MESH_VERSION 111: 0.06 GiB at 16 GiB, 0.056 at 14; at 110 every row
also carried a publication's targets, 1,136.5 bytes an arena page and 0.5 MiB, 1.11 GiB at 16 GiB, touched as rows were leased; a
program's targets now follow its SEND cells in their lease, design/transport.md), the node's base wired and macOS
memory, the rank's process, the Neural Engine's own copy of its columns, and a 2 GiB margin. A served engine (metal-microbench server/bridge.py) leases its model and pool
from its node's region too; neither file holds one.

## The Studios

Two documents are written for the four Mac Studios (M5 Ultra, 96 GiB, headless) that join the pair, one by each of the
pair's rules; which a Studio runs is the operator's decision.

`m5ultra-headless.toml`, by the Mini's rule (a headless node's memory is the mesh's; the operator, 2026-10-15: "allocate
essentially all of the memory on the entire device"): the most whole GiB that leaves a 2 GiB margin, 86 GiB. Beside it
the node keeps 2.71 GiB: the region's tables (0.339 GiB, 347.2 MiB), base wired (1.6), macOS's anonymous memory (1.5), the
rank's process (0.9) and the Neural Engine's own copy (2.95); 87 GiB would leave 1.71. It holds 26B-A4B whole at the
engine's whole page geometry (81.11 GiB) on one Studio and 12B's (97.18 GiB) on any two members, and the six regions
(four Studios at 86, the M5 at 14, the Mini at 16) hold 374 GiB.

`m5ultra-low.toml`, by the M5's rule:
a node of more flops and bandwidth than the Mini has less to gain from memoization, so its region is the least whole GiB
that holds the whole E2B model alone at the engine's whole page geometry, 14 GiB (13.27 GiB counted by programs.py `plans`
of a membership of one from the model file's header; 10.56 on the streamed codes export), and any part of E2B on any
membership holds in it. Beside it the node keeps 74.99 GiB: the region's tables (0.056 GiB), base wired (1.6, the Mini's
figure until a Studio's is read), macOS's anonymous memory (1.5), the rank's process (0.9) and the Neural Engine's own copy
(at most every FFN column at fp16, 2.95). The Mini's rule, the most that leaves a 2 GiB margin, gives 86 GiB. The larger
models alone at a call's own pages take more than 14 GiB (12B 16.46 GiB at 4,096 positions, 26B-A4B 28.25 GiB): a Studio
holds its part of them on memberships whose regions together hold the whole, and the planner fits each member's part to
its region; at the engine's whole page geometry they take 97.18 GiB (12B) and 81.11 GiB (26B-A4B), more than the six
regions at these documents hold together (four Studios at 14, the M5 at 14, the Mini at 16: 86 GiB). Counted with
metal-microbench tools/mesh/programs.py under `MESH_MODEL=e2b-engine` (12b, 26b-a4b):
`plans({'bs': 1, 'ctx_max': 256000}, structure(), pages=16000)[0]['total']`, and at a call's pages
`pages=pool_request(1, 4096)` with `ctx_max` 4096 (metal-microbench output_data/lifecycle-20261015/studio_window.py).

The region is the operator's decision; each document states the arithmetic it follows from. A Studio's base wired
memory and its devices' `max_mr` are read when it arrives: the pair's devices register 100 one-GiB extents, and a device
that registers fewer than its region's extents pairs no port until the region is lowered (its refusal logged with the
extent, above), never a clamped region. Each of a node's devices registers the whole region, so a Studio cabled on
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
