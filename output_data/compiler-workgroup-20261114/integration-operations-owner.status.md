# Numerical construction and native ABI ownership checkpoint

Reviewed 14 November 2026 at mesh ef4f467d. This replaces the earlier worker status
and its stale missing-import/build findings. The raw SAME593 session files beside
this record retain the implementation dialogue; this checkpoint is an observation,
not a new requirement or a claim about subsequent source revisions.

Source review found these boundaries:

- `operations.py` produces tensor operations and structured numerical regions.
  `metal.py` owns the native ABI and `_Builder` encoding. `lower_metal.construct`
  consumes both; it imports the `SourceOperand` representation and ABI constants
  from `metal.py` rather than defining another copy.
- The source dependence relation reaches `metal_ir_source_relation` with each
  original vertex's complete operation interval, actual typed views and supplied
  operands. `ConstructedMachine.binding` retains the borrowed arrays with their
  owner; the relation is not rebuilt from printed code.
- A captured mutation destination reaches the existing destination operand in
  construction. Returned aliases use the resulting typed storage view. Their
  output metadata carries that view's shape, strides and offset, plus the exported
  dtype/device/gradient role. The earlier exact-node mutation alias special case
  was displaced; it could not account for returned views.

The actual noncontiguous mutation observation is recorded by mesh 3f3a81c1 and
metal-microbench 9e64ac49, with raw output under
`metal-microbench/output_data/compiler-workgroup-20261114/frontends/torch/observe/mutation-native.*`.
That observation checked the old read, caller mutation, untouched backing columns
and returned strided alias. It does not establish arbitrary mutation programs or
remote execution. Current source inspection alone is not another execution result.

The deliberately malformed original definitions remain in `rdma/mesh_compile.py`,
`compiler/mesh_compiler/lower_coreml.py`, `examples/tp.py` and metal-microbench's
`metal_recording.m`. Their original placement and invalid syntax are deliberate
operator instructions; none is an executable compatibility path.

Priority 2/3/5 gaps are tracked in the active supervisors' specific observations.
In particular, the matrix-exp error/result handoff has a new finite assignment in
`integration-matrix-exp-error.md`; generated RNG inter-call effects and the native
multi-function link issue remain separate active producer/consumer corrections.
