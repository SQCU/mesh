# Single kernel-call migration

The public numerical interface is `Program.kernel_call`, with operand-independent
`BlockSpec`s and `ShapeDtypeStruct` outputs. The expression frontend binds configured
backend kernels through that call. The native C ABI is implementation machinery,
not a second installed public interface.

| Caller or surface | Disposition |
|---|---|
| `examples/streaming-overlap.py` | Removed whole-versus-streamed benchmark harness. |
| `examples/streaming-algebra.c`, `examples/streaming-algebra.py` | Removed runtime test harnesses; numerical composition remains in `mesh.nn` and the kernel-call interface. |
| `examples/streaming-expression.py`, `rdma/mesh_numpy.py` | Removed generated-C expression frontend and build rules. |
| Tensor arithmetic overloads | Removed; numerical work uses kernel calls. |
| Public `Program.bind`, `contract`, `reduce_sum`, `call`, `call_native`, `bind_native` | Removed. Region binding and completion are private implementation details. |
| Tensor-bound legacy BlockSpec form | Removed; specs describe shapes and index maps. |
| Installed raw C headers/module map | Removed from package installation. Native backend source remains canonical. |

The [asynchronous collective contract](async-collectives.md) is the complete
collective scope. Application ports and general compiler work are not additional
requirements. Current supported expression binding is described in
[indexed library](indexed-library.md).
