# Producer and consumer streaming across backends

Streaming is part of the meaning of a mesh tensor program. Every backend consumes
available indexed regions and publishes the values it finishes. A backend chooses
how to execute those regions; it does not choose whether those regions are usable.
This contract applies to MPS, ANE, generated Metal, and a symbolic NumPy-style
frontend targeting Metal alike.

The three worked examples below specify execution and observable progress. The
MPS/Metal path, symbolic expression compiler, and asynchronous Core ML adapter
are implemented. The Core ML adapter permits CPU and Neural Engine execution;
the recorded compute plans prefer CPU. ANE placement and absence of internal
Core ML operand copies are still unverified. Neither is a streaming exemption.

## One contract

A logical value has coordinates, a scalar type, a numerical meaning, and storage
in the canonical registered pages. A produced region is immutable until its
configured numerical and transport readers finish. Different live values,
including reduction contributions, have different output storage.

Configuration expands indexed arithmetic into the existing static mesh functions:
input rows, output rows, and a realized numerical encoder. It binds layouts,
allocations, device functions, reduction association, and paired transfers before
invocation. The runtime scans these functions. Their existing page stamps determine
which input regions are usable. Successful numerical completion publishes the
corresponding output regions and their configured sends through mesh completion.
There is no second graph interpreter in transport and no optional publication hook.

Producer streaming means publishing each finished value region without waiting
for unrelated output regions. Consumer streaming means executing the partial
problem enabled by available inputs without waiting for unrelated input regions.
An intermediate operator has both obligations. Sending pieces of an operand but
waiting for the complete operand before computing is insufficient. Computing
pieces but publishing only when the complete logical operation finishes is also
insufficient.

There are two distinct numerical meanings of a partial tensor:

| Value | Meaning | Immediate legal consumer |
|---|---|---|
| `Y[I,J]` | Final values at selected output coordinates | Any operation using those coordinates, including nonlinear functions |
| `P[I,J,Q]` | Contribution to `Y[I,J]` from reduction domain `Q` | The declared reduction and operations distributive over that reduction |

For contraction, `P[I,J,Q] = sum(k in Q, A[I,k] * B[k,J])` and
`Y[I,J] = sum(Q, P[I,J,Q])`. Q is a value coordinate identifying a contribution,
not a task state or arrival counter. The reduction tree and its output instances
are realized in advance. Floating-point association and accumulation type are
part of that numerical program. Arrival order must not silently change them.

`tanh(P0 + P1)` requires the sum; it cannot be replaced with
`tanh(P0) + tanh(P1)`. This is a dependency of the algebra on the corresponding
coordinates. Other coordinates and upstream contributions continue to execute.
A consumer can therefore begin a contraction before the final output is knowable.

Compute regions are chosen from the indexed dependencies and backend operation
curves. They need not equal transport messages. Binding groups their storage into
legal page-publication and SEND regions; transport does not choose the GEMM shape.
If two independent regions share one current readiness page, realization must
separate their storage to expose their independence. A larger dense alias must
not accidentally turn those separate dependencies into one whole-value stamp.

A device completion establishes visibility of the region actually written. With
an opaque device call, configuration creates independently completable subcalls.
With an exposed numerical pipeline, the backend can publish at supported internal
completion boundaries. Both implement the same tensor semantics. Publication
never precedes device visibility, and device errors must not certify numerical
success. No hardware backend is exempt from either streaming obligation.

## Example 1: MPS consumes K-panels and produces contraction regions

Take one output row and two output columns:

```
A = [1, 2 | 3, 4]
B = [[1, 0],
     [0, 1],
     --------
     [1, 1],
     [2, 1]]
```

Bind two MPS contractions at configuration time, each with interior dimension 2:

```
P0 = A[:, 0:2] @ B[0:2, :]
P1 = A[:, 2:4] @ B[2:4, :]
Y  = P0 + P1
Z  = tanh(Y)
```

P0 and P1 occupy separate registered output extents. For a large matrix, each
also has independently usable I/J regions. The corresponding receives target
separate peer extents. Their addresses and numerical MPS matrix descriptors are
bound before any input is produced.

1. The producer writes `[1,2]` and publishes its region. `[3,4]` is absent.
2. The first MPS contraction consumes that region and writes `P0 = [1,2]`.
3. Its completion publishes P0 and the configured send. A peer can receive and
   consume P0 while the second input region is still absent.
4. When `[3,4]` arrives, the second contraction writes `P1 = [11,7]`.
5. The indexed addition writes `Y = [12,9]`; its own completion publishes Y.
   The elementwise consumer computes Z for this row without waiting for other rows.

The first contraction is both a streaming consumer of A and a streaming producer
of P. It does useful contraction work while K is incomplete. MPS does not have to
inspect an incomplete large matrix: its realized invocation names exactly the
available partial problem. Apple's [MPSMatrixMultiplication interface](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
provides result dimensions, interior dimension and matrix operands for this
lowering. The splitting and publication contract is mesh's design.

Today `mesh_algebra_bind(MESH_CONTRACT, ...)` can express each P using input
slices and distinct output tensors. The binder automatically splits its output
into publication parts. `mesh_algebra_contract` now accepts the corresponding indexed operand views,
allocates their contributions, and realizes their fixed adjacent-pair reduction
tree. The symbolic frontend supplies these partitioned operands automatically.
The primitive binder still computes the full K domain of its individual view;
there is no implicit transport-sized K split of a dense unpartitioned operand.

Acceptance: withhold Q1, require P0 to be numerically correct at the peer, then
release Q1 and require Y and Z to match the declared expression. Repeat with an
unrelated output row withheld. The existing acceptance now checks both delayed-Q and delayed-row cases on
both participants, including invocation-slot reuse.

## Example 2: ANE consumes completed neuron regions and streams down-projection contributions

A DNN library expresses a gated feed-forward operation using the same algebra:

```
G[I,H] = X[I,K] @ Wg[K,H]
U[I,H] = X[I,K] @ Wu[K,H]
A[I,H] = gelu(G[I,H]) * U[I,H]
D[I,J,H] = A[I,H] @ Wd[H,J]
Y[I,J] = sum(H, D[I,J,H])
```

Here H identifies disjoint neuron regions. For each H, the adapter realizes an
accelerator program that produces G/U and A and composes their D contribution.
If A has an external reader, the program must publish A asynchronously as soon
as its writes are visible, allowing that reader and its own D computation to
proceed independently. If A has no external reader, its storage may remain local
to the numerical computation. Publication does not require ending that computation.
The adapter binds every
input and output region and compiles the needed shapes during setup. The DNN
module remains a composition supplied by the caller; mesh does not contain it.

For a concrete down-projection, suppose the first completed activation region
is `[1,2]` with weights `[[1,0],[0,1]]`. Its contribution is `[1,2]`.
The next activation region, `[3,4]` with weights `[[1,1],[2,1]]`, contributes
`[11,7]`. The output is again `[12,9]`.

1. Once X's required coordinates for H0 are present, the H0 program can execute.
   X for unrelated token rows does not participate in that dependency.
2. A0 becomes available to its readers as soon as its required writes are
   visible. Its down-projection consumer can start while A1 is absent. The
   producing program may publish A0 and continue computing D0; publication does
   not wait for A0's delivery, consumption, or the program's remaining work.
3. D0 is sent to its configured peer destination immediately upon publication.
   The peer can combine available contributions in the configured reduction tree.
4. H1 produces D1 later. Their sum completes Y for I/J; the next layer can use
   that region while other token rows remain unfinished.

If G/U themselves receive K incrementally, their K contributions are separate
values and are reduced before GELU. The down projection still consumes completed
H regions incrementally. This preserves the nonlinear dependency without adding
whole-token-batch or whole-hidden-tensor dependencies.

The ANE implementation obligation includes proving that device input/output
bindings name the actual canonical registered operand storage, with no hidden
copied operand store, and that completion establishes visibility there. The Core ML adapter now binds inputs and outputs to canonical addresses and
checks the returned output backing identity on every completion. The public
backing checks pass; internal direct binding and actual ANE execution remain
unverified. An unsupported binding path is an adapter implementation gap; it is not an alternative whole-tensor streaming contract.
Apple's [compute-unit configuration](https://developer.apple.com/documentation/coreml/mlcomputeunits)
permits CPU and Neural Engine execution together. Setting it alone does not
establish that this example ran on ANE; backend validation must establish actual
placement and operand interoperability.

Acceptance: delay H1 production and require the peer to receive correct D0;
then complete H1 and check Y. Independently delay another token row and require
this row's complete result. Both producer and consumer progress must be observed,
in addition to verifying ANE placement and canonical storage use.

## Example 3: symbolic NumPy-style expressions lower to streaming Metal

A symbolic frontend captures this expression during configuration:

```python
u = 0.5 * x + 0.125
s = peer_sum(u)
a = np.tanh(s)
y = np.einsum("ik,kj->ij", a, w)
z = y * scale + bias
```

`rdma/mesh_numpy.py` implements symbolic pointwise operations feeding a
partitioned matrix contraction, including explicit FP16/FP32 casts. It supports
NumPy ufunc/einsum dispatch as well as its own array namespace. The post-contraction
scale/bias in this full design is expressed through ordinary subsequent algebra
bindings today; arbitrary nested symbolic contractions are not yet supported. Pointwise
intermediates use FP32 unless an explicit cast specifies FP16; this is not a
drop-in implementation of all NumPy dtype-promotion rules.
The [NumPy einsum notation](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
provides the index equation; incremental execution is supplied by mesh lowering.

The frontend propagates an output index region backward through each operation.
Elementwise operators preserve coordinates; broadcasts map them to the appropriate
operand indices; contraction introduces a reduction coordinate. For each I/J/Q
it lowers the expression to:

```
U[I,Q] = 0.5 * X[I,Q] + 0.125
S[I,Q] = U_local[I,Q] + U_received[I,Q]
A[I,Q] = tanh(S[I,Q])
P[I,J,Q] = sum(k in Q, A[I,k] * W[k,J])
Y[I,J] = reduce_Q(P[I,J,Q])
Z[I,J] = Y[I,J] * scale[J] + bias[J]
```

It allocates these live values in canonical storage, realizes explicit transfers,
and compiles Metal numerical functions. The invocation contains no Python graph
walk, symbolic evaluation, dynamic allocation, backend selection, or second mesh
scheduler. A symbolic all-gather is an indexed distributed view: a consumer of
I/Q depends on those coordinates, not on materializing all gathered coordinates.

1. X[I,Q0] arrives. Generated Metal writes U[I,Q0] and its completion publishes
   that region's send while X[I,Q1] is absent.
2. The matching peer contribution enables S[I,Q0] and then A[I,Q0].
3. The contraction computes P[I,J,Q0] immediately. It publishes that contribution
   for its reduction consumers and any configured peer copy.
4. Q1 follows independently. Once the reduction for I/J completes, Z[I,J] runs
   without waiting for other I/J output regions.

Pointwise fusion may combine S and A, or fuse A into its local partial contraction,
when the required publication edges survive. It must preserve U's peer send and
any externally used P output. A frontend-wide final `eval()` cannot be the first
point at which these intermediate values become executable or publishable.
Device tiles can differ from symbolic regions; lowering must retain the region's
actual input dependencies and independent partial availability through that
change, not historical kernel or command-buffer completion boundaries.

Acceptance: withhold X[I,Q1], observe U[I,Q0] at the peer and correct P[I,J,Q0]
from the consumer, then release Q1 and verify Z. The frontend's generated program
must run through the existing algebra acceptance path, not a separate Python
simulation presented as evidence of device progress.

## Reductions, gathers, and scatters follow the same rule

A row sum consumes column contributions as partial sums. A gather reads the
indexed source values needed by each output region. A scatter with unique target
indices publishes independent writes; colliding indices require the declared
combining algebra, with separately represented contributions and a configured
association. Softmax can consume regions into mergeable max/sum-exp statistics
before its final normalization is available. Final normalization depends on the
corresponding reduction's completion, while other regions continue to execute.
No one of these dependencies grants a backend permission to wait for unrelated
coordinates.

## Source disposition and implementation order

The current `bind_part` takes the full `x.columns` for contraction and row sum.
It derives readiness from pages covering those sliced operands. `emit_part`
provides mandatory completion publication. Both mechanisms should remain shared
owners rather than being independently rewritten in each backend.

The setup-time indexed contraction lowering now emits separate K contributions
and their reductions through the existing algebra functions. It must preserve output
regions requested by downstream consumers instead of universally selecting them
from page or transport-block size. The symbolic frontend emits that representation. Core ML supplies realized
numerical functions through the shared execution completion. Actual ANE placement
and internal storage interoperability still need to be established. None needs its own transport, readiness state,
model interpreter, or participant scheduler.

The existing numerical acceptance now includes the delayed-Q observation and
runs the same program through each implemented path. Report
backend placement, early producer output, early consumer output, final numerical
agreement, and storage interoperability separately. The current MPS/Metal
row-prefix measurement remains evidence for that implementation only.

## Implemented entry points and running the examples

- `mesh_algebra_contract(a, x, w, count, output, alpha)` consumes corresponding
  arrays of indexed views, returns their contribution tensor, and binds the final
  reduction. Its reduction tree is fixed at setup; intermediate sums use FP32.
- `examples/streaming-expression.py` captures the producer, peer sum, tanh and
  contraction expression. `mesh_numpy.emit_c` generates setup bindings into the
  existing executable. Python is absent from numerical invocation.
- `mesh_algebra_coreml(a, python, generator, cache)` selects the Core ML contraction
  encoder before any numerical functions are bound. `rdma/mesh_coreml.py` compiles
  per-part matrix programs during setup. Metal and Core ML then share `emit_part`
  and successful mesh publication. Device failure never publishes success.

Build with `make -C rdma .build/streaming-algebra`. With the existing two bridges
configured for three queues, run corresponding ranks on the two participants:

```sh
rdma/.build/streaming-algebra RANK 19 257 8 0
rdma/.build/streaming-algebra RANK 19 257 8 1
rdma/.build/streaming-algebra RANK 19 257 8 1 PYTHON GENERATOR CACHE
rdma/.build/streaming-algebra RANK 19 257 8 1 PYTHON GENERATOR CACHE 1
```

The first is explicit C setup; the second uses generated symbolic setup. The
third uses Core ML with FP32 operands. The fourth explicitly rounds contraction
inputs to FP16. PYTHON is an absolute path to a Python environment with compatible
coremltools native wheels; GENERATOR is the absolute `rdma/mesh_coreml.py` path;
CACHE is a dedicated compiled-artifact directory. The recorded local environment
uses Python 3.12/coremltools 9.0 and the peer uses Python 3.9/coremltools 9.0.
Rebuild clients against the extended algebra report structure.

Each run has 19 invocations, three delayed windows and a final window with three
live slots. Both early K contributions and early output prefixes must be received
and checked before the missing input is supplied. Core ML additionally reports
native submission count, output-backing identity count, and preferred NE operation
count from its compute plan. These are distinct evidence: public output identity
does not establish internal zero-copy execution, and a supported device does not
establish selected device placement.
