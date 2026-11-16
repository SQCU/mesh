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

The same module owns the original `MetalRecorderActors` call and the original
`mesh_function[]` descriptor construction previously local to `Target.functions`.
The actual produced array retains the supplied program sequence, explicit and
intrinsic root operands, complete storage arrays, actors and library owners. Each
actor retains its queried program and machine operands. The actor allocation is
freed when the owning function array is released; a borrowed ctypes element keeps
that array alive. The original target consumes this result directly.

Function descriptors describe each supplied native entry independently. They do
not infer outer calls, chains, roots or dependency order. A joint `mesh_plan_local`
invocation must consume those original callgraph operands separately; combining
independent processor entries into a fabricated single call is not implied.
