# Native storage conversion

`rdma/mesh_storage.py::storages` is the shared conversion from the original
`metal_ir_storages` and `metal_ir_value_info` results to the existing
`mesh_value[]` planning operand. The previous `mesh_compiler.native.storages`
body lives here once; the original target and standalone consumers use it.

The result is the produced ctypes array itself. It retains the original program,
explicit and intrinsic operands, library owners, complete queried ValueInfo
collection, and borrowed storage array. Its count, stepped and quantum attributes
are the same results the original conversion computed. The original target keeps
this whole array in its existing owner collection and passes its C storage to
`mesh_function`.

This conversion supplies storage requirements and lifetime operands. It does not
allocate backing, construct page rows, bind invocation tensors or establish a
backing-address alignment guarantee. Those results belong to the original
planner and realization producers.
