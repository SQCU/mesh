# Numerical publication and storage source trace

The [asynchronous collective contract](async-collectives.md) defines the scope.
This describes the current source owners, not an additional acceptance program.

## Execution and publication

`mesh_execution_add` in `rdma/mesh-dataflow.c` registers the configured indexed
input and output dependencies. Presence notifications revisit affected functions.
`submit_ready` in `rdma/mesh-algebra.m` dispatches their realized numerical
functions to the existing dispatch queue; CPU arithmetic does not occupy the
serial presence handler.

Generated CPU kernels receive the sections prepared by `bind_publication`.
Their section loop stores results and calls `publish_cpu`, which calls canonical
`mesh_publish_partial`, while subsequent sections remain in progress. Native CPU
contractions use configured BLAS or typed numerical implementations and publish
their configured output region on completion. Independent output regions and
reduction contributions are separate realized functions.

Metal command completion and Core ML prediction completion publish the output
region through `complete_part`. The operation's independent configured regions
can complete separately. No numerical path waits for delivery, a peer
acknowledgement, or completion of the enclosing tensor operation before publishing
its finished region. The dispatch-group wait is in algebra destruction.

Consumers become eligible from their actual input presence. Contraction partials
feed the configured reduction; nonlinear consumers read completed sums. Host
result observations do not drive this numerical execution.

## Storage and lifetime

`mesh_tensor_create` allocates canonical backing. `mesh_view_create` maps aliases
of the shared-file pages; it does not populate another operand buffer. CPU
bindings use those addresses and configured strides. Metal buffers and Core ML
input/output arrays are bound to the same backing by the native owner.

The bridge's SEND/RECV work requests name the registered source and destination
pages. Canonical reader ownership retains each source through numerical and
NIC reads. Distinct simultaneous values use distinct configured storage.
Numerical reduction outputs and explicitly requested materialization are values
in that same arena, not hidden transport copies.

This establishes mesh's address bindings. It does not establish undocumented
scratch-memory behavior inside Apple's BLAS, MPS or Core ML implementations.
The collective contract does not require a reverse-engineering project for those
libraries. Established mechanisms and their sources are recorded in
[algorithm sources](algorithm-sources.md).
