# Bridge configurations

A node's bridge (`rdma/mesh-flow`) starts from one operand, its configuration: a document its consumer writes and
passes, in JSON, TOML or YAML. `bin/mesh-user.sh CONFIGURATION [CHECKOUT]` reads it with the mesh library's one
loader (`rdma/bridge_configuration.py`), writes its canonical form (a property list of its name, purpose and window
in bytes, `~/Library/Application Support/io.mesh.bridge/configuration.plist`) and names that in the bridge's agent
definition, so the keeper's load at boot keeps it. metal-microbench's `tools/mesh/node-setup.sh` passes the node's
document through (`MESH_BRIDGE_CONFIG`). There is no default anywhere in the code: without a configuration the
bridge, `mesh-user.sh` and `node-setup.sh` refuse to start, naming what they need.

A document has two keys:

- `purpose`: one line stating what the configuration is for (required).
- the window, exactly one of `window_gib` (GiB) or `window_percent` (of the node's physical memory).

The window is the bridge's pinned memory (design/prepared-machine.md#M09). Every port's device registers all of it
when the device comes up, before any queue pair exists, and deregisters it only when the device retires. A window
of more extents than the node's RDMA devices register (`max_mr` 1 GiB extents) is refused, not clamped. Clients lease
from it; a lease that does not fit fails at its start with an error naming the execution, and the window never grows.
How much an execution needs, on its own node and on the nodes it peers or streams with, is the client's to know.

## The examples

These two ship with the bridge to show the form. Neither is likely to be what an application wants; both are for
didactic programming exercises.

- `example-1pct.json`: one percent of the node's memory (245 MiB on a 24 GiB Mac mini, 1.3 GiB on a 128 GiB M5). It
  allows small streaming transfers: a libnccl-mesh session toward one peer leases 24 MiB (its rings, SEND cells,
  completion words and transfer lists). It does not allow programs that stage large operands in the window.
- `example-10pct.yaml`: ten percent of the node's memory. It holds many sessions' crossing rings, and it is too small
  for many applications: a model's weights and KV pool in the window, or a long reduction's staging.

The proper configuration for an application is the client writer's to derive and write explicitly for each node of
their mesh.

## The E2B pair under test

These two are written for the state in which the rank keeps its weights and KV pool in the window. They are not
deployed while the rank still allocates them outside it; there they would overcommit the Mini (see M09: at a 9 GiB
window the Mini's userspace starved and its watchdog reset it).

- `m4pro-headless.toml`: the Mac mini, 16 GiB.
- `m5-low.toml`: the M5 Max, 14 GiB, below the Mini's.

Each file's `purpose` states its arithmetic from the measured parts: weights, KV pool, crossing leases, the node's
base wired and macOS memory, and a 2 GiB margin.

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
