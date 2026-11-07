Sources for the revisions of design/torch-capture-change-request.md, retrieved 2026-11-07 by a retrieval agent and recorded here as it wrote them. The local paths it names are the session's download directories.

# Lifetimes and placement: sources for change-request items 6 and 7

Retrieved 2026-11-07. Every quotation is verbatim and contiguous. Separate excerpts are in separate blocks. In paper quotations, subscripts are written with `_` (for example r_i, K_min, q_i). Spelling errors in the originals are kept.

| # | Source | Status |
|---|--------|--------|
| 1 | ExecuTorch `exir/memory_planning.py`, plus torch.fx for the position of the output node | read |
| 2 | TensorFlow Lite `arena_planner.{h,cc}`, plus `simple_memory_arena`, `common.h` and `subgraph.cc`, and the 2021 `preserve_inputs` removal | read |
| 3 | TensorFlow Lite GPU delegate `memory_management/` | read |
| 4 | Pisarchyk and Lee, arXiv:2001.03288 | read |
| 5 | Rau, Lee, Tirumalai and Schlansker, PLDI 1992 | read |
| 6 | Dehnert, Hsu and Bratt, ASPLOS 1989 | read |
| 7 | Lam, PLDI 1988, §2.3 | read |

Code was fetched from raw.githubusercontent.com at the commits named below. The existing downloads in `dl-executorch`, `dl-tflite`, `dl-tflite-gpu` and `dl-tflite-2021` were compared with these fetches and are byte-identical, apart from trailing newlines in the `.diff` files. The new fetches are in `dl4-*`. The arXiv PDF was re-downloaded and matches `dl-arxiv-pisarchyk/2001.03288.pdf`.

The three older papers are ACM scans from the earlier downloads: `dl-rau/rau.pdf`, `dl-dehnert/dehnert.pdf` and `dl-lam/Lam1988.pdf`. Each was identified by its title, authors, ACM footer and printed page range, which match the Crossref records. dl.acm.org returned HTTP 403 to a fresh download, so their bytes could not be re-checked. Each quotation from these scans was checked against a rendered page image, not just the OCR text.

---

## 1. ExecuTorch: `exir/memory_planning.py`

- **Repository:** github.com/pytorch/executorch
- **Commit:** `75af78c46f465301673b03b03ed028da25871eae`, the head of `main`, 2026-11-07T05:04:46Z. `exir/memory_planning.py` and `exir/tensor.py` were last changed in `4bce1e15c7d1f0a01cceacf3fe6c08908f87441e` (2026-11-05, "Plan dynamic-shape buffers by the exact largest size when it is cheap (#24610)"). `exir/passes/memory_planning_pass.py` was last changed in `541a2fab62195a2212fcf16afe152358ae4feb11` (2026-10-08).
- **License:** the `LICENSE` file is headed "BSD License / For "ExecuTorch" software". It contains the three BSD clauses ("Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: …", and "Neither the name Meta nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission."). GitHub's license API reports it as `NOASSERTION` ("Other").

### `update_tensor_lifetime`

`exir/memory_planning.py`, lines 340–342 (docstring):

```
    Update the lifetime of the tensor to cover node_idx. A tensor's lifetime
    are represented by the index of the first and last node referring
    that tensor in its inputs/outputs.
```

Lines 348–362. A placeholder (graph input) starts at 0. Its end is its last referring node, except for mutable buffers and device-resident constants, which end at `max_node_idx`:

```python
    start, end = spec.lifetime
    if node.op == "placeholder":
        start = 0
    else:
        start = node_idx if start is None or start > node_idx else start

    if node.op == "placeholder" and (
        _is_mutable_buffer(node, gs) or _is_device_resident_constant(node)
    ):
        # Mutable buffers and device resident constants keep their data across
        # calls, so they are never freed.
        end = max_node_idx
    else:
        end = node_idx if end is None or end < node_idx else end
    spec.lifetime = [start, end]
```

### `update_all_tensors_lifetime`

Lines 572–575 and 584. Every node is visited together with its `args` and `kwargs`, so the `output` node's arguments, which are the graph's results, are extended to the output node's index:

```python
    max_node_idx = len(graph_module.graph.nodes) - 1
    for node_idx, node in enumerate(graph_module.graph.nodes):
        for spec in collect_specs_from_nodes(
            filter_nodes(itertools.chain([node], node.args, node.kwargs.values())),
```

```python
            update_tensor_lifetime(node, spec, node_idx, max_node_idx, graph_signature)
```

`exir/tensor.py`, lines 254–255:

```python
    def init_mem_planning_fields(self) -> None:
        self.lifetime = [None, None]
```

### Supporting source: where the output node sits in a torch.fx graph

- **Repository:** github.com/pytorch/pytorch
- **Commit:** `5a98911762f95259353ee036e4edfe7d3f25fb58`, the head of `main` when fetched. `torch/fx/graph.py` was last changed in `08d8c569422ce8058f89d6cfb6735b7c16233aad` (2026-11-03).
- **License:** PyTorch `LICENSE`, BSD-style: "Redistribution and use in source and binary forms, with or without modification, are permitted…"; clause 3 begins "Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America…".

`torch/fx/graph.py`, lines 1498–1502:

```python
    def output_node(self) -> Node:
        output_node = next(iter(reversed(self.nodes)))
        if output_node.op != "output":
            raise AssertionError(f"Expected output node, got op={output_node.op}")
        return output_node
```

`torch/fx/node.py`, line 286:

> ``output`` contains the output of the traced function in its ``args[0]`` attribute.

### Whether graph inputs and outputs are planned at all

`exir/memory_planning.py`, lines 1488–1489 and 1501–1502, in `apply_algo`:

```python
    # Extract the nodes and their lifespans from the graph_module
    _ = update_all_tensors_lifetime(graph_module, graph_signature)
```

```python
            ignore_graph_input=not alloc_graph_input,
            ignore_graph_output=not alloc_graph_output,
```

Lines 234–236 (docstring of `verify_graph_input_output`):

```
        alloc_graph_input / alloc_graph_output indicates if memory for graph
        input/output is allocated by the compiler. If not, the runtime will
        set them using buffers provided by users.
```

`exir/passes/memory_planning_pass.py`, lines 198–199 (the defaults of `MemoryPlanningPass.__init__`) and lines 207–208:

```python
        alloc_graph_input: bool = True,
        alloc_graph_output: bool = True,
```

```
        the graph input/output. The default behavior is the algorithm will allocate
        memory for both graph input and output.
```

### How the greedy planner uses lifetimes

Lines 1053–1054. Greedy is the default algorithm:

```python
        if algo_list is None:
            algo_list = [greedy]
```

Lines 1010–1014. Specs are taken in non-increasing size:

```python
    sorted_specs = []
    for spec in specs:
        bisect.insort(sorted_specs, spec, key=lambda x: x.allocated_memory)

    sorted_specs.reverse()
```

Lines 675–679 (`_does_not_overlap`). The closed intervals `[lifetime[0], lifetime[1]]` are compared:

```python
        if not (
            spec.lifetime[1] < alloc.spec.lifetime[0]
            or spec.lifetime[0] > alloc.spec.lifetime[1]
        ):
            return False
```

Lines 754–759 (`pick_shared_obj`). A spec goes into the first shared object whose allocations are all disjoint from it in lifetime:

```python
    for sobj in shared_objects:
        if _does_not_overlap(sobj, spec):
            assert sobj.size >= spec.allocated_memory, "Allocation specs are not sorted"
            picked = sobj
            sobj.first_used_index = min(sobj.first_used_index, spec.lifetime[0])
            sobj.last_used_index = max(sobj.last_used_index, spec.lifetime[1])
```

Lines 686–693 (`_find_max_overlapping_allocations_offset`). This is the fallback, which places the spec past every allocation it overlaps in lifetime:

```python
    max_offset = 0
    for alloc in sobj.allocations:
        if (
            spec.lifetime[1] < alloc.spec.lifetime[0]
            or spec.lifetime[0] > alloc.spec.lifetime[1]
        ):
            continue
        max_offset = max(alloc.offset + alloc.spec.allocated_memory, max_offset)
```

### How the naive planner uses lifetimes (it does not read them)

Line 1137:

```
    This algorithm simply allocates memory for each tensor sequentially without reusing memory.
```

Lines 1182–1185:

```python
        spec.realign(alignment)
        spec_alloc_result.mem_offset = _allocate_buf(
            bufsizes, spec_alloc_result.mem_id, spec.allocated_memory
        )
```

---

## 2. TensorFlow Lite: `tensorflow/lite/arena_planner.{h,cc}`

- **Repository:** github.com/tensorflow/tensorflow
- **Commit:** `6cf18d885436586b2de77e3fae38197d75ac6005`, the head of `master`, 2026-11-07T06:53:01Z. `arena_planner.{h,cc}` and `simple_memory_arena.cc` were last changed in `188ddbad65577313713d7de5940daea6b22ae6b7` (2026-08-06, "Add CPU memory allocator API to TFLite"). `core/c/common.h` was last changed in `5703b56be47de728fcad51fb2fb29f41c6799fe2`, and `core/subgraph.cc` in `0f316d5b67a7672b28393d780ef05dd9018728ce`.
- **License:** Apache-2.0. The file headers read "Licensed under the Apache License, Version 2.0 (the "License");".

### `alloc_node_` and `dealloc_node_`: the first and last node

`arena_planner.h`, lines 133–139:

```cpp
  // First node, that uses the tensor. It needs to be allocated before
  // execution of the node's operation.
  std::vector<int32_t> alloc_node_;

  // Last node, that uses the tensor. It can be deallocated after execution of
  // the node's operation.
  std::vector<int32_t> dealloc_node_;
```

`arena_planner.cc`, line 34 and lines 209–210:

```cpp
constexpr int32_t kNodeNotAssigned = std::numeric_limits<int32_t>::max();
```

```cpp
  alloc_node_.assign(num_tensors, kNodeNotAssigned);
  dealloc_node_.assign(num_tensors, kNodeNotAssigned);
```

### Graph outputs, variables and graph inputs, in `ArenaPlanner::PlanAllocations()`

`arena_planner.cc`, lines 239–246 (graph outputs):

```cpp
  // We must make sure the output tensors are never overwritten. We do that by
  // artificially adding one to their ref-counts so they are never selected
  // for deallocation.
  for (int tensor_index : graph_info_->outputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      ++refcounts_[tensor_index];
    }
  }
```

Lines 248–253 and 257–258 (variables):

```cpp
  // Variable tensors also should be ensured to be never overwritten and need to
  // be alive all the time.
  for (int tensor_index : graph_info_->variables()) {
    // Increase the reference count for variable tensors by one, so it will
    // never be deallocated.
    ++refcounts_[tensor_index];
```

```cpp
    // Variable tensor should be allocated at the very beginning.
    TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
```

Lines 262–267 (graph inputs):

```cpp
  // Queue all graph inputs for allocation and make sure they are never
  // overwritten.
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      ++refcounts_[tensor_index];
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
```

Lines 306 and 313. A node's outputs, including graph outputs, are allocated at their producing node `i`:

```cpp
    // First queue output tensors for allocation.
```

```cpp
      TF_LITE_ENSURE_STATUS(allocate(i, tensor_index));
```

Lines 316–318 and 326–328 (deallocation at the last consuming node):

```cpp
    // Then update the ref-counts of the node's inputs, and if necessary queue
    // them for deallocation.
    if (!preserve_all_tensors_) {
```

```cpp
          --refcounts[tensor_index];
          if (refcounts[tensor_index] == 0) {
            TF_LITE_ENSURE_STATUS(deallocate(i, tensor_index));
```

Lines 334–335:

```cpp
  // Note that graph outputs will never be scheduled for deallocation. We
  // could do that here for completeness, but it won't have any effect.
```

### How the interval reaches the arena

`simple_memory_arena.h`, lines 28–31:

```cpp
// allocation in the memory arena as well as first_node and last_node that use
// corresponding tensor. It means that continuous part of memory with this size
// needs to be allocated before execution of operation in the first node and can
// be deallocated after execution of the operation in the last_node. When the
```

`arena_planner.cc`, lines 543–547:

```cpp
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(
          arena_.Allocate(context_, tensor_alignment_, tensor.bytes,
                          tensor_index, alloc_node_[tensor_index],
                          dealloc_node_[tensor_index], &allocs_[tensor_index]));
```

`simple_memory_arena.cc`, lines 393–396:

```cpp
    if (alloc.last_node < first_node || alloc.first_node > last_node) {
      // Usage interval of alloc doesn't intersect with current tensor's usage
      // interval, so we skip it.
      continue;
```

`arena_planner.cc`, lines 446–449, in `CreateTensorAllocationVector`. Tensors allocated at node 0 and never deallocated are placed first:

```cpp
    // Tensors that have lifespan through the whole model inference time are
    // allocated at the beginning of memory slice. Their respective order
    // doesn't matter in fact, so here they are sorted by index.
    if (alloc_node_[idx1] == 0 && dealloc_node_[idx1] == kNodeNotAssigned) {
```

Line 459:

```cpp
    // All other tensors are sorted in non-increasing order of their size.
```

### `preserve_inputs` (removed in 2021)

The parameter existed until commit `75643b715f9ecd835cc42921120a055ad407a964` (2021-04-23), "Remove `preserve_inputs` parameter from ArenaPlanner constructor". The quotations below are from that commit's parent, `b87589cd9adfc43f62365fd4bcb06ea47c589a8c`.

`arena_planner.h`, lines 49–51 and 123–126 at `b87589cd`:

```cpp
  // ArenaPlanner is destroyed. If 'preserve_inputs' is true the inputs to the
  // graph will not share memory with any other tensor, effectively preserving
  // them until the end of inference.
```

```cpp
  // Ensure that the memory self-allocated for inputs is never reused by the
  // allocator. This allows for example, multiple runs without getting
  // unpredictable results.
  bool preserve_inputs_;
```

`arena_planner.cc`, lines 135–142 at `b87589cd`:

```cpp
  // Queue all graph inputs for allocation. If preserve_inputs_ is true, make
  // sure they never be overwritten.
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      if (preserve_inputs_) {
        refcounts[tensor_index]++;
      }
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
```

The only caller, `tensorflow/lite/core/subgraph.cc`, had this line, which `75643b71` removed:

```
-        /*preserve_inputs=*/true, preserve_all_tensors_,
```

Commit message of `75643b71`:

> Since there is no trivial use-case for the parameter, we'd better remove it for better readability.

`arena_planner.h` at `6cf18d88`, lines 51–53 (the current text):

```cpp
  // ArenaPlanner is destroyed. The inputs to the graph will not share
  // memory with any other tensor, effectively preserving them until the end
  // of inference.
```

### `preserve_all_tensors`

`arena_planner.h`, lines 151–154:

```cpp
  // If true, then no overlapping of memory areas is done, meaning intermediate
  // tensors and temporary tensors can be queried after running.
  // (modulo running delegates)
  bool preserve_all_tensors_;
```

`arena_planner.cc`, lines 355–358, in `ExecuteAllocations` (temporaries):

```cpp
      alloc_node_[tensor_index] = i;
      nodes_to_tensors_[i].insert(tensor_index);
      if (!preserve_all_tensors_) {
        dealloc_node_[tensor_index] = i;
```

Lines 144–145, in `InputTensorCanBeShared`. These lines were added by `9dfefa0d158da935a44d48adae40f86158651bad` (2024-01-31, "Disable reusing of input/output tensors if the experimental `experimental_preserve_all_tensors` is enabled."):

```cpp
  if (preserve_all_tensors_) {
    return false;
```

The rename came in commit `3b0e2315e0476d363fb97fbcc1412f939cbbfcff` (2021-04-23), "Rename preserve_intermediates to preserve_all_tensors". From its message:

> When it set, current interpreter also preserves temporary tensors in addition to intermediates tensors.

### The two arenas, `arena_` and `persistent_arena_` (added at the coordinator's request)

`arena_planner.h`, lines 141–143 and 147–149:

```cpp
  // Raw memory buffer that is allocated for all temporary and graph outputs
  // that are declared kTfLiteArenaRw.
  SimpleMemoryArena arena_;
```

```cpp
  // Raw memory buffer that is allocated for persistent tensors that are
  // declared as kTfLiteArenaRwPersistent.
  SimpleMemoryArena persistent_arena_;
```

`tensorflow/lite/core/c/common.h`, lines 464–467:

```
///  * `kTfLiteArenaRw`: Arena allocated with no guarantees about persistence,
///        and available during eval.
///  * `kTfLiteArenaRwPersistent`: Arena allocated but persistent across eval,
///  and only available during eval.
```

`tensorflow/lite/core/subgraph.cc`, lines 2109 and 2118–2119, in `SetTensorParametersReadWrite`. Read-write tensors default to `kTfLiteArenaRw`, and variable tensors are `kTfLiteArenaRwPersistent`:

```cpp
  TfLiteAllocationType allocation_type = kTfLiteArenaRw;
```

```cpp
  } else if (is_variable) {
    allocation_type = kTfLiteArenaRwPersistent;
```

`arena_planner.cc`, lines 551–557. A persistent tensor is planned from `alloc_node_` to the largest `int32_t`, unlike the `[alloc_node_, dealloc_node_]` interval that `arena_` receives (lines 543–547 above):

```cpp
    if (tensor.allocation_type == kTfLiteArenaRwPersistent &&
        allocs_[tensor_index].size == 0) {
      if (allocs_[tensor_index].size < tensor.bytes) {
        TF_LITE_ENSURE_STATUS(persistent_arena_.Allocate(
            context_, tensor_alignment_, tensor.bytes, tensor_index,
            /*first_node=*/alloc_node_[tensor_index],
            /*last_node=*/std::numeric_limits<int32_t>::max(),
```

Lines 56–60 (`BasePointer`):

```cpp
  if (type == kTfLiteArenaRwPersistent) {
    return persistent_arena_.BasePointer();
  }
  if (type == kTfLiteArenaRw) {
    return arena_.BasePointer();
```

How each arena is sized. Both are `SimpleMemoryArena`, and both are placed by the same `Allocate`. `simple_memory_arena.cc`, lines 424–427 and 441–445:

```cpp
  // Update the required buffer size.
  size_t required_buffer_size = 0;
  TF_LITE_ENSURE(context, CheckedAdd(best_offset, size, &required_buffer_size));
  high_water_mark_ = std::max(high_water_mark_, required_buffer_size);
```

```cpp
  // Resize the arena to the high water mark (calculated by Allocate), retaining
  // old contents and alignment in the process. Since Alloc pointers are offset
  // based, they will remain valid in the new memory block.
  TF_LITE_ENSURE_STATUS(
      underlying_buffer_.Resize(high_water_mark_, arena_reallocated));
```

`arena_planner.cc`, lines 433–436 (`ArenaPlanner::Commit` commits each arena separately):

```cpp
  TF_LITE_ENSURE_STATUS(arena_.Commit(&arena_reallocated));
  has_nonpersistent_memory_ = true;
  TF_LITE_ENSURE_STATUS(
      persistent_arena_.Commit(&persistent_arena_reallocated));
```

Lines 384–386. Only `arena_` is released between invocations:

```cpp
TfLiteStatus ArenaPlanner::ReleaseNonPersistentMemory() {
  // Clear non-persistent arena's buffer.
  TF_LITE_ENSURE_STATUS(arena_.ReleaseBuffer());
```

---

## 3. TensorFlow Lite GPU delegate: `tensorflow/lite/delegates/gpu/common/memory_management/`

- **Repository:** github.com/tensorflow/tensorflow
- **Commit:** `6cf18d885436586b2de77e3fae38197d75ac6005`, as for item 2. Last changes: `types.h` in `b5d2374f5e21ff0aa44ac26b039336d7443d08e3` (2020-08-25); `greedy_by_size_assignment.cc` in `5426fbec49f3a0a9469fb33b2fbf9c2b01459e95` (2025-01-04); `greedy_by_size_assignment.h` in `955be713b9da1a742b558dfb5aedf777af7df342` (2021-07-14); `internal.cc` in `bb1bf6edf20454eb365ff462c8507617e803fd0a` (2021-08-10).
- **License:** Apache-2.0.

### `TensorUsageRecord`

`memory_management/types.h`, lines 30–38:

```cpp
// Record, containing tensor size/shape and IDs of the first and the last task,
// that use this tensor as input or output. For example: tensor #3 with size
// tensor_size=65536 is first introduced in program #2 (first_task=2) and used
// for the last time in program #7 (last_task=7).
template <typename TensorSizeT>
struct TensorUsageRecord {
  TensorSizeT tensor_size;
  TaskId first_task;
  TaskId last_task;
```

### `GreedyBySizeAssignment`

`memory_management/greedy_by_size_assignment.h`, lines 30–32 and 43–45:

```cpp
// - We have tensor usage records of all intermideate tensors as an input. Each
// record consists of tensor size, first and last tasks, that use it. Let's call
// [first_task..last_task] a tensor usage interval;
```

```cpp
absl::Status GreedyBySizeAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    size_t base_addr_align_bytes, OffsetsAssignment* assignment);
```

`memory_management/greedy_by_size_assignment.cc`, lines 81–82:

```cpp
  std::stable_sort(ordered_records.begin(), ordered_records.end(),
                   CompareBySize);
```

`memory_management/internal.cc`, line 30 (the body of `CompareBySize`):

```cpp
  return first.usage_record->tensor_size > second.usage_record->tensor_size;
```

`greedy_by_size_assignment.cc`, lines 93–97, 105–107, 124–127 and 138–140:

```cpp
      if (usage_records[allocated_id].last_task < rec->first_task ||
          usage_records[allocated_id].first_task > rec->last_task) {
        // Tensor allocated_id has usage interval, that doesn't intersect with
        // current tensor's usage interval, so we skip it.
        continue;
```

```cpp
        if (diff >= rec->tensor_size && diff < best_diff) {
          best_diff = diff;
          best_offset = prev_offset;
```

```cpp
    // If no suitable gap found, we should allocate current tensor after the
    // rightmost tensor, which usage interval intersects with the current one.
    if (best_offset == kNotAssigned) {
      best_offset = prev_offset;
```

```cpp
    assignment->offsets[rec_with_idx.idx] = best_offset;
    assignment->total_size =
        std::max(assignment->total_size, best_offset + rec->tensor_size);
```

`tensorflow/lite/delegates/gpu/common/memory_management.h`, lines 130–131, and `memory_management.cc`, lines 162–164:

```cpp
// Calculates the assignment of tensors to offsets, considering those tensors
// are going to be allocated in one continuous memory block.
```

```cpp
  if (strategy == MemoryStrategy::GREEDY_BY_SIZE) {
    return GreedyBySizeAssignment(usage_records, base_addr_align_bytes,
                                  assignment);
```

---

## 4. Pisarchyk and Lee (2020)

Yury Pisarchyk and Juhyun Lee, "Efficient Memory Management for Deep Neural Net Inference", arXiv:2001.03288 [cs.LG]. Versions: v1, 10 Jan 2020; v2, 23 Jan 2020; v3, 16 Feb 2020. The arXiv comment reads "6 pages, 6 figures, MLSys 2020 Workshop on Resource-Constrained Machine Learning (ReCoML 2020)". The version read is https://arxiv.org/pdf/2001.03288, which is v3: 7 pages, created 2020-02-17.

Page 2, §3 "Definition of Terms":

> **Tensor Usage Interval** of an intermediate tensor t is defined as the pair {first_op_t, last_op_t}, where first_op_t and last_op_t are the indices of the first and the last operator that use t as its input or output, respectively. The indices are from a topological sort of the neural network which is also the operators' execution order.

> **Tensor Usage Record** of an intermediate tensor t is defined as the triple {first_op_t, last_op_t, size_t}, where size_t is t's aligned size in bytes.

Page 2, caption of Figure 1:

> note that tensor #8 is not an intermediate tensor.

Page 4, §5 "The Offset Calculation Approach":

> We call the other memory sharing approach Offset Calculation where a large chunk of memory is pre-allocated and the intermediate tensors are given parts of the memory by the offsets within the memory block.

Page 5, §5.2 "Greedy by Size for Offset Calculation". "non-increasing" is hyphenated across a line break in the PDF.

> We first iterate through tensor usage records in non-increasing order by their size (L.1,6). For each record, we check already assigned tensors whose usage intervals intersect with that of the current tensor (L.10–13) to find the smallest gap in memory between them such that current tensor fits into that gap (L.9,14–17).

---

## 5. Rau, Lee, Tirumalai and Schlansker (1992)

B. R. Rau, M. Lee, P. P. Tirumalai and M. S. Schlansker, "Register Allocation for Software Pipelined Loops", in *Proceedings of the ACM SIGPLAN 1992 Conference on Programming Language Design and Implementation (PLDI '92)*, pp. 283–299. DOI 10.1145/143095.143141; also *ACM SIGPLAN Notices* 27(7), July 1992, DOI 10.1145/143103.143141. The page-1 footer reads "ACM SIGPLAN '92 PLDI-6/92/CA". The copy read is `dl-rau/rau.pdf`, an ACM scan of 17 pages.

### Rotating register file

Pages 283–284, §1.2 "A quick overview of modulo scheduling". The last two words fall at the top of p. 284.

> A rotating register file is addressed by adding the register specifier in the instruction to the contents of the Iteration Control Pointer (ICP) modulo the number of registers in the rotating register file. A special loop control operation, brtop, decrements the ICP amongst other actions. As a result of the brtop operation, a register that was previously specified as r_i is specified as r_{i+1}, and a different register now corresponds to the specifier r_i.

Page 284, the same paragraph continued:

> Newly generated values are written to successive locations in the rotating register file and do not overwrite previously generated values even though exactly the same code is being executed repeatedly. This also introduces the need for the compiler to perform value tracking; the same value, each time it is used, may have to be referred to by a different register specifier depending on the number of brtop operations that lie between that use and the definition [6].

Page 286, §1.3:

> Once a value is generated into a physical register r_i, it occupies that register until the lifetime ends. The same loop-variant produced in the next iteration is generated into physical register r_{i-1} due to the decrementing of the ICP (even though the register specifier used is the same). If r_i is 0, r_{i-1} will be the highest numbered register.

Page 289, §3 "Formulation of the register allocation problem":

> When discussing a vector lifetime, we shall use the lifetime produced by the first iteration of the loop as the point of reference. Thus, a vector lifetime has been allocated to physical register 7 if the defining operation in the first iteration writes its result to physical register 7. Iterations 2, 3, etc., would write their results to physical registers 6, 5, and so on.

> However, assuming the presence of the kind of register renaming that is provided by the rotating register file, the register specifier in the instruction would be invariant.

### The two schemes

Page 286, §2 "Code Generation Strategies":

> We have two schemes for handling the overlapping lifetimes for a loop-variant that are encountered in modulo scheduled code:
> • static renaming via modulo variable expansion, (MVE), and
> • dynamic remapping via the use of rotating register files (DR).

### Modulo variable expansion

Page 288, §2.2 "Code generation without hardware support":

> When no hardware support is available, modulo scheduling can be accomplished by modulo variable expansion (MVE), i.e., unrolling the kernel and renaming the multiple copies that now exist of each virtual register definition [11]. The unrolling and renaming is required to prevent successive lifetimes, corresponding to the same loop-variant, from overwriting one another in the same register.

Page 288, §2.2:

> The minimum degree of unroll, K_min, is determined by the longest lifetime among the loop-variants. Because iterations are initiated every II cycles, K_min can be calculated as:
> K_min = MAX_i ( ⌈(end_i-start_i) / II⌉ ).

Page 290, §3.1 "The register allocation constraint":

> The DIST matrix for scalar lifetimes, with the MVE strategy, is much simpler. If two scalar lifetimes X and Y are live simultaneously, then DIST[X,Y] = DIST[Y,X] = 1. Otherwise, DIST[X,Y] = DIST[Y,X] = 0.

Page 294, §4.3 "Computational Complexity":

> (Note that for the MVE strategy, the number of lifetimes will be K_min times greater than for the other strategies.)

### Lifetimes, live-in and live-out values (relevant to item 7)

Page 284, §1.3 "Lifetimes of values within loops":

> The lifetime of a value starts when the producer is issued and ends when all of the consumers have finished.

Page 286, §1.3:

> Live-in values, which are pre-loaded prior to loop entry, must remain in the registers until their lifetimes end. This adds a leading blade to the wand of recurring values which extends, on the left, all the way through cycle 0.

> As discussed, the lifetimes of live-out values, which are used after loop exit, must be extended until the loop completes. This adds a trailing blade to the wand of recurring values.

---

## 6. Dehnert, Hsu and Bratt (1989)

James C. Dehnert, Peter Y.-T. Hsu and Joseph P. Bratt, "Overlapped Loop Support in the Cydra 5", in *Proceedings of the Third International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS III)*, April 1989, pp. 26–38. DOI 10.1145/70082.68185; also *ACM SIGARCH Computer Architecture News* 17(2), DOI 10.1145/68182.68185. The copy read is `dl-dehnert/dehnert.pdf`, a scan of 13 pages printed 26–38.

Page 30, §4.1 "Register Address Generation":

> The register overwrite problem is solved by making multiconnect register addresses relative to a hardware register called the *Iteration Control Pointer (ICP)*. Each physical multiconnect register referenced by an instruction is calculated as the sum of the register specifier in the instruction field and the ICP. The ICP is decremented on each iteration through the loop. The logical effect of this on the program is that the multiconnect register numbers rotate. For example, a reference to physical register 35 on one iteration will be a reference to physical register 34 on the next iteration, and so on.

Page 30, §4.1:

> it completely avoids the necessity of copying values to preserve them through lifetimes longer than a single iteration stage. In this regard, it produces the same effect as "modulo variable expansion" in [Lam87], but does not require replication of the loop body.

Page 32, §4.3 "The brtop Operation", text accompanying Figure 6:

> Note that for a value produced in one stage (a single cycle in the example) and consumed in a later stage, the register number used to refer to it is offset by the stages which have been executed in the meantime, because of the decrementing of ICP.

Page 32, two lines from Figure 6 "Inner Product with Registers". A value written as `%3:1` in stage 0 is read as `%3:7` in stage 6:

```
%3:1 = *%5:0 [0]
%2:7 = %3:7 * %4:7 [6]
```

Page 35, §6.4 "Value Renaming":

> The rotating multiconnect in the Cydra 5 effectively provides renaming for free for the multiple definitions of a scalar that occur on successive iterations of an overlapped loop.

Page 36, §7.4 "Exposed Operand Processing and Recurrence Initialization". This bears on values live into and out of the loop (item 7):

> In order to guarantee that all such TNs are available for their entire lifetimes, they are allocated to the rotating multiconnect. This requires that the exposed uses be annotated to refer to the previous iteration's definition (unique and always available after the above transformation), and that a copy to the initial multiconnect register be inserted in the DoHead block. In addition, if the value is live on exit from the loop body, it must be copied back to its GPR in the DoTail block.

---

## 7. Lam (1988), §2.3

Monica Lam, "Software Pipelining: An Effective Scheduling Technique for VLIW Machines", in *Proceedings of the SIGPLAN '88 Conference on Programming Language Design and Implementation*, Atlanta, Georgia, June 22–24, 1988, pp. 318–328. DOI 10.1145/53990.54022; also *ACM SIGPLAN Notices* 23(7), DOI 10.1145/960116.54022. The copy read is `dl-lam/Lam1988.pdf`; `dl-lam/lam-sp.pdf` is byte-identical.

Pages 322–323, §2.3 "Modulo variable expansion".

Page 322. The section's example is a value written and used two cycles later. It reads "This code can be sped up by using different registers in alternating iterations:", followed by this code (column alignment approximate):

```
  Def(R1)
  op       Def(R2)
L:Use(R1)  op       Def(R1)
           Use(R2)  op       Def(R2) CJump L
                    Use(R1)  op
                             Use(R2)
```

Page 322:

> We call this optimization of allocating multiple registers to a variable in the loop *modulo variable expansion*. This optimization is a variation of the variable expansion technique used in vectorizing compilers [18]. The variable expansion transformation identifies those variables that are redefined at the beginning of every iteration of a loop, and expands the variable into a higher dimension variable, so that each iteration can refer to a different location.

> Modulo variable expansion takes advantage of the flexibility of VLIW machines in scalar computation, and reduces the number of locations allocated to a variable by reusing the same location in non-overlapping iterations.

> When modulo variable expansion is applied, code sequences for consecutive iterations differ in the registers used, thus lengthening the steady state. If there are *n* repeating code sequences, the steady state needs to be *unrolled n* times.

> Next, we pretend that every iteration of the loop has a dedicated register location for each qualified variable, and remove all *inter-iteration* precedence constraints between operations on these variables. Scheduling then proceeds as normal. The resulting schedule is then used to determine the actual number of registers that must be allocated to each variable.

> If each variable v_i is allocated its minimum number of locations, q_i, the degree of unrolling is given by the lowest common multiple of {q_i}.

Page 323:

> We observe that the minimum degree of unrolling, *u*, to implement the same schedule is simply max_i q_i. This minimum degree of unrolling can be achieved by setting the number of registers allocated to variable v_i to be the smallest factor of *u* that is no smaller than q_i, i.e.,
> min n, where n ≥ q_i and u mod n=0.
