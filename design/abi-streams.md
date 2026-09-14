# Mesh storage boundary

The [asynchronous collective contract](async-collectives.md) is the specification.
The numerical calling surface is documented in [indexed library](indexed-library.md).

The bridge owns the registered shared-memory region and verbs resources.
`mesh_attach` maps the existing named region into the application. Numerical
operands refer to the actual registered backing through configured tensor views.
Mesh binds transfers and reader lifetimes during realization; presence and
completion drive execution. Numerical callers do not manage stream states,
copy operands into a pending transport queue, or run transport progress loops.

## Registered address boundaries

The registration implementation is `rdma/mesh-verbs.h`. It assigns memory regions
and keys to configured message extents. The historical observation of a receive
landing 4 GiB below its requested address is preserved in
[the kernel recovery record](RDMA-KERNEL-RECOVERY.md).

Registration and address-to-key lookup belong to the canonical verbs owner.
They must cover the actual operand backing and retain it through device access.
The numerical API does not expose registration banks or memory-region keys.

## Source ABI

Shared-region version 26 removes the unused transfer timestamp arena and its
header offset. Bridge, native library and region readers use the same
`rdma/mesh.h` layout. Transfer descriptors, registered backing, presence and
reader ownership remain the communication data. No timestamp or reconstructed
trace participates in publication, consumption or transfer eligibility.
