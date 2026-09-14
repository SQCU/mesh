# Algorithm sources

Sources are keyed to the public constructs in [collective goals](collective-goals.md#symbol-allowlist).
Implementation flow and limitations belong in [streaming algebra](streaming-algebra.md).

## Program

The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html): realize functions, layouts and storage before numerical invocation. POSIX supplies process and thread lifetimes; mesh owns its registered storage through completion and teardown.

## Program.tensor

The JAX authors, [Refs and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html), and Apple, [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt): tensor blocks name actual registered pages. Ref slices, transposes and broadcasts describe those same operands; they do not create a copied transport store. Generated CPU and Metal expression loads/stores resolve logical byte indices through the canonical page table. Setup binds the actual arena in Metal-sized banks, its page table, and per-operand geometry; invocation allocates no address tables and copies no operands. MPS and supplied encoders still bind fixed extent views and require a corresponding indexed-address integration before transport can relocate all operands.

The llama.cpp authors, [GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): the engine's existing `ModelFile` loader reads configured weight slices directly into registered `MatrixView`s. `Program.constant(ref)` marks a region initialized by such a setup loader; passing a value also fills the region.

## Program.kernel_call

Papadopoulos and Culler, [Monsoon: an Explicit Token-Store Architecture](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf), ISCA 1990: operand-associated presence drives function issue. The JAX authors, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html): a grid and index maps bind independently usable regions.

Apple, [Metal command submission](https://developer.apple.com/documentation/metal/mtlcommandbuffer/commit()): prebound encoders submit device work; successful completion makes its output visible to mesh consumers. Configured function watches own their dependency edges; each edge also records its position in the row index. Removing a function unlinks those edges directly, without scanning all arena rows. Reader fanout is counted at realization; grouped readers receive stamp indices directly from spans reserved using that count. The per-reader pending-node list and later assignment pass are removed. Unbinding derives source rows from the configured map rather than a duplicate source-index array. A removed reader’s stamp becomes a constant satisfied dependency; reset clears only active stamps, so removal cannot create a missing reader on the next value. After resetting a reused value, retirement evaluates the remaining dependency stamps in the same event. If unbinding removed the final reader span, the empty conjunction completes immediately instead of requiring another notification from a nonexistent reader. Reader groups and receive-side checks still remain. Dependency-index mutations remain on the existing numerical dispatch queue. A supplied encoder performs numerical work only. It does not commit, wait, publish or manage readers. `mesh_algebra_buffer` returns the existing registered extent buffer; `Program(functions=...)` binds supplied numerical implementations during setup.

## Program.copy

Apple, [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt): SEND/RECV posting, registered operands, finite frame queues and completion handling. Dedicated send and receive threads spin independently of numerical execution, including initial receive posting. The send worker drains each polled completion batch and refills from its ready indices without walking the publication list between individual completions. It collects new publications after the CQ sweep. Setup follows the configured connector/listener role. Nonblocking connection and duplex metadata progress have no software deadline, socket timeout or sleep; explicit stop or client detachment cancels setup. An absent peer is incomplete pairing, not a transport failure. Each queue exchanges its configured receive bindings once; each sender validates the corresponding identities, relative offsets and sizes. The shared-memory transfer record contains only local row, binding identity, relative offset, reader plane and byte count. Its array position is the transfer identity within a queue. Runtime announcements carry those indices, and the receiver gathers its own destination from its realized table. Neither participant stores the other participant’s physical page addresses. Posting resolves local rows through the canonical page table; completion retains the exact posted page. Shared-memory ABI 30 includes the indexed transfer layout, reduced port status record and realized send-source mask. Realization marks the source rows of configured transfers. Publication intersects its row range with that mask before emitting send notices; numerical-only regions do not enter the transport notice queue. Initial transport setup still indexes already-present source values, including constants published before realization. Row allocation and client retirement clear routing membership. Port status retains phase, error domain and error code; unused copies of device/peer identity and error timestamps are removed. Transport performs no clock reads or timestamped setup logging. Real socket/provider errors remain reported. The wire signature distinguishes this protocol from the former double exchange. Index descriptions identify configured transfers in ready order. Receive posting still follows these announcements and still checks prior readers; this change does not implement up-front posting or eliminate receive-side reuse checks. The send thread reserves hardware queue capacity while selecting ready transfers and posts their payloads directly after the description, without an intermediate announced-payload queue. `Program.replicate` composes these copies. Local scatter/gather materialization plans logical source/destination byte indices and splits each segment at page boundaries during setup. Invocation resolves those indices through the canonical page table and uses the existing `memcpy` operations; it creates no staging operand and performs no allocation.

## Program.write

Papadopoulos and Culler, *Monsoon* (1990): host writes publish operand presence after their writes are visible. The host-writer check belongs to `Program.write`; the separate `Ref.writable` polling property and its native query wrappers are removed. The existing host-writer reuse path retains previous-reader lifetime checks; distinct configured instances use distinct storage. The unused program-wide publication-size query is also removed; setup still derives regions from each tensor’s actual publication geometry.

## Program.export

Papadopoulos and Culler, *Monsoon* (1990): `Result` observes output presence and records consumption in the canonical reader state. Export does not insert an intermediate tensor or numerical operation.

## kernels.expression

The JAX authors, [Pallas indexing](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs): arithmetic, casts, indices and masks describe functions over Ref regions. `kernels.arguments` names the inputs; the retained scalar operations support the listed neural-network compositions. Floor-division and remainder expression operators, modular integer reduction trees and the custom multiword integer SIMD sum are removed. Tensor sums accumulate floating-point inputs in FP32 through ordinary addition and Metal’s existing `simd_sum`; integer indices and masks remain available. Reduction lowering no longer carries an enclosing output dtype through its recursive traversal. Indexed loads use configured pages. Setup specializes statically known access regions and binds their dependencies directly. A whole-table gather with value-dependent indices binds the table as an input; indices and masks execute only in the numerical kernel. Selector tensors, candidate retirement, payload-index scans in the dependency scheduler, and their native binding API are removed. Constant embedding weights need no runtime reader retirement. This removes selective scheduling of sparse, dynamically produced table candidates; it does not change numerical gather indexing or masks.

## kernels.dot

Dongarra, Du Croz, Hammarling and Duff, *A Set of Level 3 Basic Linear Algebra Subprograms*, ACM TOMS 1990; Apple, MPSMatrixMultiplication, Accelerate/BNNS and Core ML: existing numerical matrix operations compute the configured contraction contributions. Contraction setup derives each partial’s relative input/output views once, together with its input dependencies. BLAS and BNNS bind those same views instead of separately reconstructing rectangles from output offsets; MPS and Core ML use the same setup traversal. Numerical calls retain their prebound dimensions, strides, storage and workspace. This does not yet change their fixed physical bindings or receive-buffer reuse. The JAX authors, [Pallas accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation): each K contribution writes separate storage and addition combines contributions.

PyTorch DTensor authors, `Partial` placement, and the Legion authors, reduction privileges: setup-only `Partial` records required and present contribution sets. Duplicate terms are rejected; addition clears the marker only when all required terms are present. Nonlinear public kernel calls reject unfinished sums.

## kernels.add

The JAX authors, [Pallas reductions and accumulation](https://docs.jax.dev/en/latest/pallas/pipelining.html#reductions-and-accumulation): additions combine independently available contributions. Caller-supplied addition bindings are also used inside contraction and normalization reduction trees.

## collective.reduce_scatter

Rabenseifner, *Optimization of Collective Reduction Operations* (2004), and Patarasuk and Yuan, *Bandwidth optimal all-reduce algorithms for clusters of workstations* (2009): copies and additions reduce each block at its caller-configured owner. `collective.send` is `Program.copy`; `collective.all_gather` distributes owner blocks; `collective.all_reduce` composes reduce-scatter and all-gather. No placement search or numerical algorithm is added to transport.

## nn.ffn

Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053), and the MLX authors, [tensor-parallel layers](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/distributed.py): column-partitioned expansion followed by row-partitioned projection. `nn.linear` uses the existing contraction. `nn.ffn` composes `linear`, balanced additions, and activation directly, keeping projection accumulation in FP32 and casting at activation/output boundaries; it does not rebuild contraction tiling in an enclosing expression. FFN down-projection contributions pass through reduce-scatter and all-gather. The caller supplies partitions and owners.

Hendrycks and Gimpel, [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) (2016): the engine's existing gated activation computes GELU(gate) times up. Its encoder now binds a separate output operand; existing in-place calls use the same buffer explicitly.

## nn.rmsnorm

Zhang and Sennrich, *Root Mean Square Layer Normalization* (2019): sum squared features, normalize by the reciprocal root mean square and apply the scale. The expression composition reduces feature contributions before normalization. The engine binding uses its existing FP16 or FP32 numerical kernel on canonical regions.

## nn.embedding

The JAX authors, [Pallas indexed Refs](https://docs.jax.dev/en/latest/pallas/design/design.html#indexing-refs): indices select table rows and masks delimit valid elements; configured output blocks publish independently.
