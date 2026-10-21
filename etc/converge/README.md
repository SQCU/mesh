# Convergence configurations

A node's user layers (the mesh checkout and its bridge here; a consumer's, such as metal-microbench's resident rank,
by the same commands) follow their upstreams by themselves once the node has a convergence document:
`bin/mesh-user.sh CONFIGURATION CHECKOUT CONVERGENCE` reads it with the mesh's one loader (`rdma/bridge_configuration.py`
READERS: JSON, TOML or YAML), writes its canonical form (`~/Library/Application Support/io.mesh.converge/configuration.plist`),
records its path beside it and writes the admin user's agent `io.mesh.converge` (README.md, Convergence). Without
one, nothing moves the checkout but the next run of `mesh-user.sh`. There is no default. Whether a node converges is
the operator's choice: `bin/mesh-converge.sh off` takes a node out again, removing the recorded document and every
layer's agent definition; each layer's loop ends at its next pass.

A document has two keys, each required:

- `purpose`: one line stating what the configuration is for.
- `period_s`: seconds between passes (`mesh-converge.sh follow` reads it before each). A push has no event on the
  node, so a pass asks the upstream every period.

`five-minutes.toml` is written for the headless nodes of the E2B pair and the Studios; its `purpose` states the
reasoning. `development-node.toml` is for a node that is also where its checkouts are edited: a checkout carrying work
in progress is held (`mesh-converge.sh held`: changes to tracked files, commits its upstream lacks, or no upstream), and
once it is clean and level with its upstream a push is built and installed there as on any node. Without a document, a
node's images change only when someone runs `mesh-user.sh` or `make install`, which is how a node's installed
libraries came to lag its checkout by days.
