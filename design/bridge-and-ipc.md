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

Source pages remain retained through actual numerical and device reads.
Independent values use their configured distinct storage. The bridge does not
interpret a model graph, choose tensor placement, or invoke application kernels.

SIGINT and SIGTERM request orderly teardown of device resources. The operational
rules are in [RDMA-RULES.md](../RDMA-RULES.md). Device teardown belongs to the bridge,
not to a numerical function. The storage boundary is documented in
[abi-streams.md](abi-streams.md).
