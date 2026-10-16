# Convergence configurations

A node's user layers (the mesh checkout and its bridge here; a consumer's, such as metal-microbench's resident rank,
by the same commands) follow their upstreams by themselves once the node has a convergence document:
`bin/mesh-user.sh CONFIGURATION CHECKOUT CONVERGENCE` reads it with the mesh's one loader (`rdma/bridge_configuration.py`
READERS: JSON, TOML or YAML), writes its canonical form (`~/Library/Application Support/io.mesh.converge/configuration.plist`),
records its path beside it and writes the admin user's agent `io.mesh.converge` (README.md, Convergence). Without
one, nothing moves the checkout but the next run of `mesh-user.sh`. There is no default. Whether a node converges is
the operator's choice: `bin/mesh-converge.sh off` takes a node out again, removing the recorded document and every
layer's agent once no pass runs.

A document has six keys, each required:

- `purpose`: one line stating what the configuration is for.
- `period_s`: seconds between passes (the agent's StartInterval).
- `quiet_s`: seconds the node must be idle on end before a layer's staged build and before its turn: no live hold
  (`~/.cache/mesh/hold`), no other layer's turn, the bridge carrying no call (mesh-stat `idle`, and `idle_since` at least
  this long ago, where the bridge publishes them), every consumer's probe idle (`~/.cache/mesh/idle.d`), no other process
  mapping the library, and the GPU at most `gpu_percent`.
- `notice_s`: seconds a turn is announced (node-info `converging`) before the node is checked idle again and changed, so
  a driver that reads node-info before it places a call leaves the node out.
- `wait_s`: the most seconds a pass waits for `quiet_s` of idleness before it leaves the turn to the next pass (at least
  `quiet_s`), and the most its turn waits for the bridge to stop itself on a new binary and start again.
- `gpu_percent`: the most GPU utilization (IOAccelerator "Device Utilization %") an idle node shows.

`five-minutes.toml` is written for the headless nodes of the E2B pair and the Studios; its `purpose` states the
reasoning. A development machine whose checkouts carry work in progress needs none: its checkouts are held anyway
(`mesh-converge.sh held`: changes to tracked files, commits its upstream lacks, or no upstream).
