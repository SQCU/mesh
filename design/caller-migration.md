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
| Persistent policy runtime, strategy execution/learner/responder and their curriculum/demo launchers | Deleted; no compatibility path. |
| `xonotic/solver/strat/tensor_metal.py` | Retained numerical source compiler, explicit operand layouts and owner-to-peer copy edges; general region lowering remains. |

The operator explicitly removed legacy maintenance obligations. Deleted binding
consumers and their launchers are not supported alternatives. The retained
planner, Frames, numerical model/compiler and live J-oracle viewer use their
existing canonical owners. General compiler BlockSpec lowering and the paired
gold chain are separate remaining validation work; historical training records
do not establish either.
