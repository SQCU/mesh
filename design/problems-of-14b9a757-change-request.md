# Change request: collective values and cyclic storage

## 1. AlltoAll blocks

Collective lowering produces each destination block by MPI 4.1 §7.8's AlltoAll mapping. The local block uses
MPICH's local-copy operation; an in-place operand uses MPI's specified in-place layout. The memory planner
receives the resulting source and destination views.

## 2. AllGather's local block

Collective lowering places the local producer's result in its own AllGather block, or emits MPICH's local copy
into that block (MPI 4.1 §7.7; NCCL in-place AllGather). Routing consumes the remaining endpoint views.

## 3. Tensor views

Placement represents a tensor view by its storage base, offset, sizes and strides (PyTorch, *Tensor Views*).
The page table and numerical consumers use those operands to address the same storage.

## 4. Window operands

Placement constructs window operands by GSPMD §4.3's halo exchange and DynamicSlice. Replicated input supplies
local windows by DynamicSlice (§4.5); the required local layout is produced by the planned copy operation.
Numerical lowering consumes the resulting view and its dependence edges.

## 5. Local CollectivePermute

Collective lowering implements a source/destination pair on the same rank by the corresponding local copy or
in-place view (GSPMD CollectivePermute; MPI 4.1 in-place semantics). The planner carries its actual storage uses.

## 6. View and reshape

Placement applies PyTorch `view`'s size/stride compatibility and `reshape`'s view-or-copy semantics. The result
is a storage view or an explicit copy operation, consumed by allocation and numerical lowering.

## 7. Lifetimes and completion

The memory planner consumes each value's first definition and last use (ExecuTorch, *Memory Planning*) together
with communication completion (MPI 4.1 §3.7.3; rdma-core `ibv_post_send(3)`). Mutable state and invariants remain
live for their compiled owner's lifetime. Ordinary values receive reusable storage after their final use.

Collective lowering places PyTorch's `wait_tensor` before the first numerical use of an asynchronous result
(`AsyncCollectiveTensor`, `torch.distributed._functional_collectives`). The local collective's completion covers
its sends and receives. The bridge publishes successful and failed operation counts using libfabric's
`fi_cntr_read`/`fi_cntr_readerr` model. The invocation consumes their sum as completion and reports the recorded
error after the operation's endpoints complete (`fi_cancel`; `ibv_post_send(3)` buffer-reuse rule).

## 8. Initializers

Placement propagates GSPMD's replicated sharding through parameter views and uses DynamicSlice to select each
node's required part (GSPMD §§3.1/3.5/4.5). Construction loads that part from `ExportedProgram.state_dict` into
its assigned pages, following Megatron-LM core_r0.12.0 `layers.py:_initialize_affine_weight_cpu`. Numerical
execution consumes those initialized values throughout the compiled owner's lifetime.

## 10. Subarrays

Placement produces MPI 4.1 §6.1.3's `ndims`, full sizes, subsizes and starts, with the tensor's stride information.
Routing and allocation consume the dimension-sized arrays as views of endpoint storage.

## 11. Link loss and retirement

The bridge completes persistent operations through their success/error counters and releases their resources
through deferred reclamation (MPI persistent requests; libfabric cancellation; McKenney's RCU). The next link-state
view supplies the next compilation's topology operand, as plan-to-bridge R8 specifies.

## 12. Membership views

The bridge orders membership views by `(Lamport time, creator(node, boot))`, applying JGroups `ViewId`'s logical
time and creator ordering. Published schedules and their bindings retain the view identity they consume.

## 13. Queue errors

The bridge drains a failed queue through completed work-request identities and records each operation's error
(`ibv_poll_cq(3)`; Linux `ib_drain_sq`; libfabric error counters). Invocation completion consumes those records;
resource reclamation follows completion of the corresponding requests.

## 14. Cyclic storage

The allocator constructs cyclic lifetime intervals over the unrolled executions, using the definitions in
Mukerji, *Register allocation using cyclic interval graphs* (1994), §2.1. A value of `b` blocks needs `b` colors;
local and routed values share the same allocation problem. A received interval begins at its execution's start;
its endpoint includes the final numerical use and communication completion (item 7; MPI_START).

The graph module applies Mukerji §3.2.3's Main Fatcover, Establish All Covers and Choose A Cover, followed by
Appendix A's left-edge algorithm for the remaining linear intervals. Its result supplies each value's blocks
in each unrolled execution to the memory plan and page table. Shared allocation consumers retain that result
through [AOT A5/A8](aot-compilation-change-request.md).

## Sources

- [MPI 4.1](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/mpi41-report.htm), §§3.7.3/3.9/6.1.3/7.7/7.8;
  MPICH AlltoAll/AllGather local-copy implementations cited in this file at `cdd44406`.
- Xu et al., *GSPMD*, 2021, §§3.1/3.5/4.3/4.5.
- PyTorch, *Tensor Views*, `view`, `reshape`, `ExportedProgram` and `_functional_collectives`;
  ExecuTorch, *Memory Planning* (source excerpts in `cdd44406`, this file, items 3/6/7/8).
- [rdma-core manuals](https://github.com/linux-rdma/rdma-core/tree/master/libibverbs/man),
  `ibv_post_send(3)` and `ibv_poll_cq(3)`; [libfabric counters](https://ofiwg.github.io/libfabric/v1.22.0/man/fi_cntr.3.html).
- JGroups `ViewId`, and the operator's membership-view operand, recorded in this file at `cdd44406`, item 12.
- McKenney, *Is Parallel Programming Hard, And, If So, What Can You Do About It?*, RCU;
  [plan-to-bridge R8](plan-to-bridge-change-request.md).
- [Mukerji 1994](https://mcgill.scholaris.ca/items/4e5e10ea-b427-4537-8a24-9a6efa7da1b6), §§2/3 and Appendix A;
  Hendren, Gao, Altman and Mukerji, *A register allocation framework based on hierarchical cyclic interval graphs*,
  CC 1992, [DOI](https://doi.org/10.1007/3-540-55984-1_17).
