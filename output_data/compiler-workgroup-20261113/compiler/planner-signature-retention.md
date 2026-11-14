# Exact schedule-worker continuation: preserve the produced native signature

Same session dc4b988b-a100-4e17-8c80-ff3f90ac97e3, model explicitly claude-opus-5-5. Old composition turn is committed
fddeb611. This is a new narrow result handoff, not another scheduling algorithm. Do not spawn, stage, commit or build
shared outputs. Implement promptly from these known owner sites; no broad inventory.

Only source hunks:

- compiler/mesh_compiler/schedule.py solved(plan, planning).
- compiler/mesh_compiler/ir.py: ONLY the trailing fields of Plan. Do not touch executor's dead ir.call deletion or
  any other IR definition.

Add Plan.signature: bytes = b"" and Plan.nodes: tuple = () as trailing defaults, preserving existing positional
constructors. signature is the immutable native mesh_graph_archive output; nodes is exactly tuple(planning.nodes),
the authoritative correspondence from native mesh map index to machine identity. No channel-derived map.

In solved, after successful mesh_plan and before mesh_planned_free, set plan.signature to
mesh_graph.archive(made.signature.contents), and plan.nodes to tuple(planning.nodes). The sibling numerical worker
is adding that thin wrapper to rdma/mesh_graph.py now. Do not copy the native signature arrays/derive operations in
Python, parse the opaque plan.record, or rerun native graph analysis. Existing plan.record remains the native plan
restoration artifact. Retain existing completion/order/sections and ownership/free behavior. No new guard/rejection.

CR compilation-composition 3 and 6/6.4: carry an already computed result to its consumer. Actual signature producer:
mesh-collective.c plan_signature1068, called by plan_graph. It describes BSL submission waits/BRIDGE_START/counts_tell;
PLAN_COUNT+a names added synchronization edges, not transfers. It does not supply transfer arrival/completion events.

Executor owns archive.py encoding/decoding of the two fields and signature.py consuming them with existing native
mesh_graph_compose after correct identities align. It is already informed. No source ownership outside named hunks.

Run py_compile/diff checks. If a true existing planning fixture is available use it for retention evidence; do not
attach a bridge or manufacture a fake planner for tests. State source/API checks separately from actual planner
execution. Write concise planner-signature-retention.status.md here; hold READY for supervisor review and named commit.
