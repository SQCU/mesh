# Sources for the stops of two change requests: quoted

Retrieved 2026-11-07 for the stops of design/torch-capture-change-request.md (items 1, 2, 3, 4 and 6) and
design/problems-of-14b9a757-change-request.md (its three sections of sources for the stops). Downloads are in their own
directories under `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107/`; text extracted from PDFs with `pdftotext -layout` is under `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107-text/`. Every
excerpt below was copied by script (`/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107-scripts/record.py`) from the named file's lines; none was retyped. `[…]`
marks an omission. Page numbers are the printed ones.

Sources this record only points to, because an earlier record already quotes them: RenderDoc, GFXReconstruct and the CUDA
Programming Guide (design/capture-replay-sources-2026-11-07.md, CS §1, §3, §4); Reed and Kanodia 1979 and Lam 1988
(design/completion-and-period-sources-2026-11-07.md, CP §3, §7); TensorFlow Lite's two arenas, `TensorUsageRecord` and
`GreedyBySizeAssignment`, and Pisarchyk and Lee 2020 (design/lifetimes-and-placement-sources-2026-11-07.md, LP §2, §3,
§4); MPI 4.1 §4.2 partitioned communication (design/worklists-and-arrival-sources-2026-11-07.md, WA §5); Apple's
MTLSharedEvent documentation (design/replay-grounding-2026-11-06.md, RG, the Metal paragraph of item 6's sources);
Hendren, Gao, Altman and Mukerji's memo §2.2, Mukerji 1994 §3.1, and Dagor Engine's `do_arcs_overlap`
(design/problems-of-14b9a757-change-request.md, item 14).

---

## 1. Bhattacharyya, Sriram and Lee (1997), §4.2 and §4.3: execution time estimates, and the bound Bff on a feedforward edge

**Bibliographic:** Shuvra S. Bhattacharyya, Sundararajan Sriram, Edward A. Lee, "Optimizing Synchronization in
Multiprocessor DSP Systems", *IEEE Transactions on Signal Processing* 45(6):1605–1618, June 1997. DOI 10.1109/78.600002.

**Read:** the authors' preprint (Ptolemy project, https://ptolemy.berkeley.edu/papers/97/synchronization/synch.pdf),
already downloaded by the CP retrieval at `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit3/bsl1997/synch.pdf` (sha256 `af16289c1695e0cd92466362d6bea34ca7e133f745009e0cf0732c263b91b53c`); text extracted to `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107-text/bsl1997.txt`. Page numbers
are the preprint's. CP §4 quotes the first sentences of §4.3 on feedforward edges and stops before equation (6); the
passage is given whole here.

**§1** (text lines 196–198):

```text
implementations of iterative dataflow programs. We assume that “good” estimates are available for the
execution times of actors and that these execution times rarely display large variations so that self-timed
```

**§4.2 Execution Time Estimates, p. 12** (text lines 583–592):

```text
4.2           Execution Time Estimates
If we only have execution time estimates available instead of exact values, and we set t ( v ) in the
12
previous section to be these estimated values, then we obtain the estimated iteration period by calculating
1
λ max . Henceforth we will assume that we know the estimated throughput ----------- calculated by setting the
λ max
```

The "12" is the page number and the "1" over "λ max" the printed fraction 1/λ_max.

**§4.3, pp. 13–14** (text lines 617–627 and 637–684; Fig. 2 and the page number "13" between them are omitted):

```text
Feedforward edges have no such bound on buffer size; therefore for practical implementations we
need to impose a bound on the sizes of these edges. For example, Fig. 2(a) shows an IPC graph where the
IPC edge ( A, B ) could be unbounded when the execution time of A is less than that of B , for example. In
practice, we need to bound the buffer size of such an edge; we will denote such an “imposed” bound for a
feedforward edge e by Bff ( e ) . Since the effect of placing such a restriction includes “artificially” con-
straining src ( e ) from getting more than Bff ( e ) invocations ahead of snk ( e ) , its effect on the estimated
```

```text
throughput can be modelled by adding the reverse edge d m ( snk ( e ), src ( e ) ) , where
m = Bff ( e ) – delay ( e ) , to Gipc (grey edge in Fig. 2(b)). Since adding this edge introduces a new cycle in
Gipc , it may reduce the estimated throughput; to prevent such a reduction, Bff ( e ) must be chosen large
enough so that the maximum cycle mean remains unchanged upon adding d m ( snk ( e ), src ( e ) ) .
Sizing buffers optimally such that the maximum cycle mean remains unchanged has been studied
by Kung, Lewis and Lo in [13], where the authors propose an integer linear programming formulation of
the problem, with the number of constraints equal to the number of fundamental cycles in the DFG (poten-
tially an exponential number of constraints).
An efficient albeit suboptimal procedure to determine Bff is to note that if
Bff ( e ) ≥  ∑ t ( x ) ⁄ λ max                                      (6)
          
x∈V
holds for each feedforward edge e , then the maximum cycle mean of the resulting graph does not exceed
λ max . This is because the reverse edge that gets added as a result of imposing a buffer bound on e intro-
duces new cycles; the maximum execution time along any such newly introduced cycle can be at most
∑ t ( x ) ; and hence adding the number of delays given by (6) guarantees no change in the maximum
x∈V
cycle mean.
Then, doing a binary search on Bff ( e ) for each feedforward edge, and computing the maximum
cycle mean at each search step and ascertaining that it is less than λ max results in a buffer assignment for
the feedforward edges. Although this procedure is efficient, it is suboptimal because the order that the
edges e are chosen is arbitrary and may effect the quality of the final solution. However, as we will see in
Section 9, imposing such a bound Bff is a naive approach for bounding buffer sizes and, in terms of syn-
chronization costs, there is a better technique for bounding buffers. Thus, in our final algorithm, we will
not in fact find it necessary to use or compute these bounds Bff .
```

The summation in (6) is over x ∈ V, as printed under the sum sign on p. 14.

---

## 2. Linux v6.12, `kernel/sched/completion.c` and `Documentation/scheduler/completion.rst`: `complete_all`

**Bibliographic:** Linus Torvalds et al., *Linux kernel*, tag v6.12. GPL-2.0. `kernel/sched/completion.c` and
`Documentation/scheduler/completion.rst` ("Completions - 'wait for completion' barrier APIs").

**Read:** https://raw.githubusercontent.com/torvalds/linux/v6.12/kernel/sched/completion.c, saved at `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107/linux-completion/completion.c` (sha256
`0a0426c7ba68149a7018383513591439e97f84a9af56fb5bd7a53095de3d8204`); https://raw.githubusercontent.com/torvalds/linux/v6.12/Documentation/scheduler/completion.rst, saved at
`/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107/linux-completion/completion.rst` (sha256 `831181b32bd5b934fd6080731bbcce00fd2e4a21804f2fd8e237546d836f30e0`).

`completion.c` L52–78:

```c
 * complete_all: - signals all threads waiting on this completion
 * @x:  holds the state of this particular completion
 *
 * This will wake up all threads waiting on this particular completion event.
 *
 * If this function wakes up a task, it executes a full memory barrier before
 * accessing the task state.
 *
 * Since complete_all() sets the completion of @x permanently to done
 * to allow multiple waiters to finish, a call to reinit_completion()
 * must be used on @x if @x is to be used again. The code must make
 * sure that all waiters have woken and finished before reinitializing
 * @x. Also note that the function completion_done() can not be used
 * to know if there are still waiters after complete_all() has been called.
 */
void complete_all(struct completion *x)
{
	unsigned long flags;

	lockdep_assert_RT_in_threaded_ctx();

	raw_spin_lock_irqsave(&x->wait.lock, flags);
	x->done = UINT_MAX;
	swake_up_all_locked(&x->wait);
	raw_spin_unlock_irqrestore(&x->wait.lock, flags);
}
EXPORT_SYMBOL(complete_all);
```

`completion.rst` L252–263:

```text
... or calls complete_all() to signal all current and future waiters::

	void complete_all(struct completion *done)

The signaling will work as expected even if completions are signaled before
a thread starts waiting. This is achieved by the waiter "consuming"
(decrementing) the done field of 'struct completion'. Waiting threads
wakeup order is the same in which they were enqueued (FIFO order).

If complete() is called multiple times then this will allow for that number
of waiters to continue - each call to complete() will simply increment the
done field. Calling complete_all() multiple times is a bug though. Both
```

---

## 3. Metal, macOS 26.4 SDK headers: the event's 64-bit value, `newLibraryWithSource:`

**Bibliographic:** Apple, Metal framework headers, macOS SDK 26.4 (`SDKSettings.json` Version 26.4), as installed at
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`. Not downloaded.

`MTL4CommandQueue.h` L261–275 (sha256 of the header `51bfecbd27ee6ba37891f42f0b75532dd915b9ccb6d6316f6f2966e8983bb881`):

```objc
/// Schedules an operation to signal a GPU event with a specific value after all GPU work prior to this point is complete.
///
/// - Parameters:
///   - event: ``MTLEvent`` to signal.
///   - value: the value to signal the ``MTLEvent`` with.
- (void)signalEvent:(id<MTLEvent>)event
              value:(uint64_t)value;

/// Schedules an operation to wait for a GPU event of a specific value before continuing to execute any future GPU work.
///
/// - Parameters:
///   - event: ``MTLEvent`` to wait on.
///   - value: the specific value to wait for.
- (void)waitForEvent:(id<MTLEvent>)event
               value:(uint64_t)value;
```

`MTLEvent.h` L60–64 (sha256 `c9edb00f4a1a94b6d023b6ce80c0c02137eabbfe2741ffa5f55f6216ed182087`):

```objc
// Synchronously wait for the signaledValue to be greater than or equal to 'value', with a timeout
// specified in milliseconds.   Returns YES if the value was signaled before the timeout, otherwise NO.
- (BOOL)waitUntilSignaledValue:(uint64_t)value timeoutMS:(uint64_t)milliseconds API_AVAILABLE(macos(12.0), ios(15.0)) NS_SWIFT_UNAVAILABLE_FROM_ASYNC("Use 'await valueSignaled(...)' instead.");

@property (readwrite) uint64_t signaledValue; // Read or set signaled value
```

`MTLDevice.h` L782–786 (sha256 `d1ef1bb5bc85faeeec9bc85f49124a453a8cd33b1182b0a71a41ba0dc4ebbb89`):

```objc
/*!
 @method newLibraryWithSource:options:error:
 @abstract Load a MTLLibrary from source.
 */
- (nullable id <MTLLibrary>)newLibraryWithSource:(NSString *)source options:(nullable MTLCompileOptions *)options error:(__autoreleasing NSError **)error;
```

---

## 4. Janghaeng Lee (2015), SKMD: partition-ready kernels and the index-space analysis of a kernel's source

**Bibliographic:** Janghaeng Lee, *Virtualizing Data Parallel Systems for Portability, Productivity, and Performance*,
Ph.D. dissertation, Computer Science and Engineering, University of Michigan, 2015 (committee chair Scott Mahlke).
Chapter III, "SKMD: Single Kernel Execution on Multiple Devices", §3.2.1 Kernel Transformation, §3.2.2 Buffer
Management. The chapter's journal form is J. Lee, M. Samadi, Y. Park, S. Mahlke, "SKMD: Single Kernel on Multiple
Devices for Transparent CPU-GPU Collaboration", *ACM Trans. Comput. Syst.* 33(3), 2015, DOI 10.1145/2798725 (not read:
Semantic Scholar reports it closed access).

**Read:** https://cccp.eecs.umich.edu/theses/jhaeng-thesis.pdf, saved at `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107/kim2011/jhaeng-thesis.pdf` (sha256 `2cfc47e86822a332651f4ce66484ddce6f3ced0aa7fbf466ea6352f71d6164c5`); text
extracted to `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107-text/jhaeng-thesis.txt`.

**Not read:** J. Kim, H. Kim, J. H. Lee, J. Lee, "Achieving a single compute device image in OpenCL for multiple GPUs",
PPoPP 2011, pp. 277–288, DOI 10.1145/1941553.1941591, the thesis's reference [38] (its §3.1 sentence on it is quoted
below). ACM's copy is paywalled; the authors' page https://aces.snu.ac.kr/~jlee/pubs/ppopp11.pdf returns 404;
https://snucl.snu.ac.kr/publications.html did not answer; Semantic Scholar's API rate-limited the query. The two
other files fetched while looking for it (`/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107/kim2011/namyst-article2.pdf`, SnuCL, and
`/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107/kim2011/osti-1295154.pdf`, "A Distributed OpenCL Framework using Redundant Computation and Data Replication")
are other papers by the same group and were not used.

**§3.1 Introduction** (text lines 982–988):

```text
applicable data parallel kernels are limited by usage of the APIs, which requires access
locations of all threads to be analyzed statically. Kim et al. [38] proposed the illusion of a
single compute device image for multiple equivalent GPUs. Although they improved the
portability by using OpenCL as their input language, their work also puts several constraints
```

**§3.2, p. 17** (text lines 1195–1210):

```text
tioner, and performance predictor as shown in the grey boxes in Figure 4.2. The kernel
transformer changes the original kernel to the Partition-ready kernel, which enables the
kernel to operate on a subset of work-groups. After kernel transformation, the buffer man-
ager performs static analysis on kernels to determine the memory access pattern of each
work-group. If the memory access range of each work-group can be analyzed statically,
the buffer manager will transfer only necessary data back and forth from each device once
the partitioning decision is made. On the other hand, if the memory access range cannot
be analyzed, the entire input should be transferred to each device and the output must be
```

**§3.2.1, p. 19** (text lines 1294–1304):

```text
Partition-Ready Kernel: Assigning partial work-groups can be done through the code
transformation shown in Figure 4.1. The lines of code with gray background illustrate the
generated code by dynamic compiler. As shown in the figure, it adds a parameter WG from
and WG to to represent the range of the flattened work-group indices to be computed on
a device. In other words, SKMD runs (W G f rom − W G to + 1) work-groups and skips
the rest on a device. If a kernel launches more than a one dimensional NDRange, flattening
```

**Figure 3.4, p. 20, "Partition-ready Blackscholes kernel"** (text lines 1320–1341, as printed):

```text
5         __global float *strike, float r, float v,
6         int WG_from, int WG_to)
7{
8    int idx = get_group_id(0);
9    int idy = get_group_id(1);
10    int size_x = get_num_groups(0);
11    int flattened_id = idx + idy * size_x;
12    // check whether to execute
13    if (flattened_id < WG_from || flattened_id > WG_to)
14        return;
15    int tid = get_global_id(1) * get_global_size(0)
16              + get_global_id(0);
17    float c, p;
18    BlackScholesBody(&c, &p,
19            price[tid], strike[tid], r, v);
20    call[tid] = c;
21    put[tid] = p;
22 }
Figure 3.4: Partition-ready Blackscholes kernel.
```

**§3.2.2, p. 25** (text lines 1603–1628):

```text
checks if the kernel is a contiguous kernel by analyzing index space of each work-group.
For index space analysis, the buffer manager uses data flow analysis focusing on the index
operand of store and load instructions, which is represented as tid in Figure 3.6(b). Using
use-def chains, the buffer manager computes the function of index. If the function is affine
and represented as a · (W0 · w0 + l0 ), it is defined as a Contiguous Kernel. In this equation,
a is a constant or an induction variable of loop, and wi , li , and Wi represents work-group
ID, work-item ID, and size of work-group in the i-th dimension, respectively. For this
type of kernel, it is safe to transfer a subset of data to each device proportional to assigned
work-groups.
On the other hand, if the index space of the kernel cannot be determined statically, or
the affine function fails to be recognized as above, the buffer manager gives up optimizing
data transfer and defines it as a discontiguous kernel. In this case, the entire input and
output will be transferred back and forth between the host and external devices if the kernel
```

---

## 5. Megatron-LM, `megatron/core/tensor_parallel/layers.py`: each rank's slice of a weight, taken at construction

**Bibliographic:** NVIDIA, *Megatron-LM*, tag `core_r0.12.0`, `megatron/core/tensor_parallel/layers.py`. BSD-3-Clause.
The method is Shoeybi et al.'s (arXiv:1909.08053); this file is its reference implementation. The tag's commit SHA was
not retrieved (the GitHub API returned no object for the tag name, though raw.githubusercontent.com serves it).

**Read:** https://raw.githubusercontent.com/NVIDIA/Megatron-LM/core_r0.12.0/megatron/core/tensor_parallel/layers.py,
saved at `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107/megatron/layers.py` (sha256 `d20bf5a85610b995799d5a9372600cdf9ad32a6763d6a68e804a6b0f15ef4e8b`).

L125–169, `_initialize_affine_weight_cpu`:

```python
def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
    rank=None,
    world_size=None,
    skip_set_tensor_parallel_attributes=False,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    if not skip_set_tensor_parallel_attributes:
        set_tensor_model_parallel_attributes(
            tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
        )

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)
    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    if rank is None:
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        # all tensors must live on the same device
        cpu_weight = torch.cat(my_weight_list, dim=partition_dim).to_dense()
        weight.data.copy_(cpu_weight)
    if return_master_weight:
        return master_weight
    return None

```

---

## 6. Dawn, `src/dawn/native/metal/ShaderModuleMTL.mm`: the MSL source reaches Metal through `newLibraryWithSource:`

**Bibliographic:** The Dawn & Tint Authors, *Dawn*, commit `797946823414be58054116de563d9ec4636dcb2e` (the commit CS §5
reads). BSD 3-Clause.

**Read:** https://raw.githubusercontent.com/google/dawn/797946823414be58054116de563d9ec4636dcb2e/src/dawn/native/metal/ShaderModuleMTL.mm,
saved at `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107/dawn-shader/ShaderModuleMTL.mm` (sha256 `845a0537ece56a58e2f746eee152ed88df1925138dceb1d50e0584eb970805ef`).

L547–556, in `ShaderModule::CreateFunction`:

```objc
    auto mtlDevice = ToBackend(GetDevice())->GetMTLDevice();
    NSError* error = nullptr;

    NSPRef<id<MTLLibrary>> library;
    platform::metrics::DawnHistogramTimer timer(GetDevice()->GetPlatform());
    {
        TRACE_EVENT(DAWN_TRACE_CATEGORY(), "MTLDevice::newLibraryWithSource");
        library = AcquireNSPRef([mtlDevice newLibraryWithSource:mslSource.Get()
                                                        options:compileOptions.Get()
                                                          error:&error]);
```

---

## 7. GSPMD (Xu et al., arXiv:2105.04663v2): replicated tensors, propagation through preserved dimensions, resharding

**Bibliographic:** Yuanzhong Xu et al., "GSPMD: General and Scalable Parallelization for ML Computation Graphs",
arXiv:2105.04663, v2 (23 Dec 2021).

**Read:** https://arxiv.org/html/2105.04663v2, saved at `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107/gspmd/gspmd-2105.04663v2.html` (sha256 `3a2fa1cf09c47edaf81fc96b8e1b8ddb5c9cda56f40178c9ec5370e4afd4873a`). Sentences were extracted by
regular expression from the HTML's text.

§3.1 Representation of tensor sharding; §3.5 Intuitive sharding completion; §4.5 Resharding:

> Replicated. All devices have the same full data.

> XLA operators typically preserve some dimensions from inputs to outputs. We simply propagate sharding of such a dimension from inputs to outputs, and vice versa.

> Elementwise operators are assigned the highest priority to propagate through in both directions, because there is no communication if their inputs/outputs are sharded consistently, and propagating through elementwise operators would be the most intuitive decision to users.

> We decided to keep the sharding propagation simple, so it does not try to create new sharding patterns on dimensions.

> It uses AllGather to replicate data along sharded dimensions, AllToAll to switch sharded dimensions, CollectivePermute to change device order, and DynamicSlice to shard replicated dimensions.


---

## 8. MPI 4.1, §4.2.1: MPI_PRECV_INIT's partitions and count

**Bibliographic:** Message Passing Interface Forum, *MPI: A Message-Passing Interface Standard, Version 4.1*, November
2, 2023. WA §5 quotes §4.2's semantics; the receive-side binding is added here.

**Read:** the copy WA downloaded, `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit3/dl-mpi41/mpi41-report.pdf` (sha256 `f32920662645ed9e51ce7ba266f10b4832d5f612ea14be698169cd6a81f1fee5`);
pages 116–119 (PDF 158–161) extracted to `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/s1107-text/mpi41-p116-119.txt`.

**p. 118** (text lines 121–131; the right-hand column of line numbers is the standard's own margin numbering):

```text
MPI_PRECV_INIT(buf, partitions, count, datatype, source, tag, comm, info, request)              24
25
IN         buf                          initial address of recv buffer (choice)
26
IN         partitions                   number of partitions (non-negative integer)           27
IN         count                        number of elements received per partition             28
(non-negative integer)                                29
```

