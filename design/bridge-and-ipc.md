# Bridge and shared-memory ownership

The [asynchronous collective contract](async-collectives.md) defines numerical
scope. `rdma/mesh-flow.c` owns bridge progress; `rdma/mesh-verbs.h` owns the device,
queue pairs, completion queues and registered memory. `rdma/mesh-dataflow.c`
owns presence and numerical dependencies. Application numerical functions remain
outside the bridge.

Setup records configured transfers and binds their source and destination pages.
The bridge posts sends for published source regions and receives into configured
registered destinations. Receive completion publishes available data; send
completion releases the device's source read. Numerical completion independently
publishes results to subsequent functions and transfers. None of these paths
requires a numerical caller to pump communication or await an enclosing tensor.

Besides the one prepared program, the bridge serves any number of clients and
communicators at NCCL's network-plugin level (`rdma/mesh-net.h`, `ncclNet_v12_t`):
one session per link for the bridge's lifetime (its own port, the link's service +
1000), connections by listen key, clients' shared memory registered in place, and
isend/irecv/test/iflush driven by the receiver: an isend announces its message, the
receiver matches it with an irecv, posts each chunk's RECV and only then grants that
chunk, and the sender SENDs exactly the granted chunks, so no SEND meets a queue
without its RECV. The region's counters record flow-control stalls, credit waits and
each communicator's traffic (`mesh-stat`).

Source pages remain retained through actual numerical and device reads.
Independent values use their configured distinct storage. The bridge does not
interpret a model graph, choose tensor placement, or invoke application kernels.

SIGINT and SIGTERM request orderly teardown of device resources. The operational
rules are in [RDMA-RULES.md](../RDMA-RULES.md). Device teardown belongs to the bridge,
not to a numerical function. The storage boundary is documented in
[abi-streams.md](abi-streams.md).
