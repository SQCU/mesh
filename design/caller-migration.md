# Single kernel-call migration

The public numerical interface is `Program.kernel_call`, with operand-independent
`BlockSpec`s and `ShapeDtypeStruct` outputs. Built-in backend kernels and ordinary
NumPy kernels use that same call. The native C ABI is implementation machinery,
not a second installed public interface.

| Caller or surface | Disposition |
|---|---|
| `examples/streaming-overlap.py` | Migrated to `kernel_call`; existing RDMA comparison retained. |
| `examples/streaming-algebra.c` | Replaced by `examples/streaming-algebra.py`, using the FFN composition. |
| `examples/streaming-expression.py`, `rdma/mesh_numpy.py` | Removed generated-C expression frontend and build rules. |
| Tensor arithmetic overloads | Removed; numerical work uses kernel calls. |
| Public `Program.bind`, `contract`, `reduce_sum`, `call`, `call_native`, `bind_native` | Removed. Region binding and completion are private implementation details. |
| Tensor-bound legacy BlockSpec form | Removed; specs describe shapes and index maps. |
| Installed raw C headers/module map | Removed from package installation. Native backend source remains canonical. |
| `xonotic/planner/plan.py` | Ported to chunked `kernel_call` using the existing expert operator compiler; local algebra checked, paired run pending. |
| `xonotic/solver/strat/runtime_transport.py` | Uses application-local canonical `Frames`; paired framing validation pending. |
| `xonotic/solver/strat/strat_responder.py` | Uses canonical `Frames`; persistent-policy runtime migration remains. |
| `xonotic/solver/strat/tensor_runtime.py` and generated encoder in `tensor_metal.py` | Outstanding: require the removed `mesh_rows_*`, `mesh-metal.h`, and `mesh-metal.m` interfaces. |

The Xonotic entries remain in scope. Import scans must include them; finishing
the algebra examples does not establish an all-callers migration. Their callers
include persistent policy execution, optimizer storage adoption, remote graph
configuration, and application frame transport. Those application behaviors must
be preserved while moving storage and numerical submission to the single owner.
