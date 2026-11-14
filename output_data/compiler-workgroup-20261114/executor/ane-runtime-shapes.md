# ANE runtime shape consumer — proposed finite boundary, 2026-11-14

Scope: SAME e48, existing `metal_neural.mm` Lowering/Mil owner only; no shared build or target run yet.
Source baseline metal `e9912b93`; public ABI and fixed NeuralIO descriptors need not change.
This contract is published before implementation for compiler/executor review.

1. Source authority is existing Operation.bounds/maps/terms/symbols/symbolBounds/extents.
   `extents[l]` is a domain dimension `bounds.size()+symbol ordinal`, not a ValueID.
   `symbols[s]` is the actual scalar ValueID; `Need(symbols[s])` obtains its existing MIL value.
   A value produced earlier within this same function reuses `values[value]`; an outside scalar
   enters `inputs` and then the existing actual Neural binding/IO producer exactly once.
   Current code does NOT call Need for symbols, so symbols do not currently reach Neural
   bindings just because general operation effects list them.
2. Existing Tensor.shape remains the result type shape: known dimensions >=0, unknown -1.
   TensorType emits MIL Dimension.unknown for -1, never casts -1 to a constant uint64 size.
   Runtime sizes are ordinary MIL i32 scalar/vector values, not guessed host values.
   A per-operation lowering result owns actual scalar symbols and clipped loop extent tensors;
   its static type shape records -1 only for genuinely runtime axes. No public shape cache.
3. Mil gains only direct existing-operation forms needed by that result: runtime shape-vector
   reshape, slice_by_index/slice_by_size bounds, range_1d, and tensor shape/concat as needed.
   Preserve Expr/Term coefficient, divisor, modulus and offset semantics in generated indices;
   do not use the existing compile-time Evaluate with absent symbol coordinates.
4. Operand consumes this same domain. Structured affine slices use runtime begin/end;
   general maps build actual-size coordinate tensors and gather_nd, not N-sized coordinates
   followed by a mask. Iota is range_1d over the actual extent, reshaped to its loop axis.
5. Lower's contribution broadcast/tile, reduction domain, and epilogue indices consume that
   same produced shape. Contraction's B/M/N/K are products of the actual relevant extents;
   its existing matmul result gets unknown type dimensions and the actual reshape operands.
   Retain the existing constant-weight convolution specialization only when its required
   channel/group/weight geometry is statically valid; do not pad dynamic contractions to N.
6. Store generates actual parallel-domain destination indices and scatter_nd updates into
   the existing prior full physical tensor, preserving every inactive destination element.
   The final function output retains its fixed physical shape and existing NeuralIO backing.
   Full-permutation shortcuts require the actual full domain, not just matching static bounds.
7. Zero PARALLEL extent means no output write: preserve prior output exactly.
   Zero REDUCTION extent means the existing combiner/state identity, then the existing
   start/ACCUMULATE, FINALIZE and epilogue semantics; it does not mean zero for every combiner.
   Use existing MIL conditional blocks if an empty slice/reduction would be unsupported;
   select of already-executed invalid work is not an adequate guard. Custom coupled reduction
   states must retain their existing combine/identity semantics; publish a concrete limitation
   rather than silently running a full-bound reduction or substituting a different algorithm.
8. Source symbols read by later neural kernels must remain actual observable outputs;
   inspect LowerNeural outside()'s inputs-only scan and consume original symbols there too.
   The numerical source effect/copy/planning owners remain unchanged.
9. Compilation remains CoreML ML Program (CoreML8, specification9) via compileModelAtURL,
   then MLModel with CPUAndNeuralEngine. It is not a direct ANE compiler/invoker.
   Existing MLComputePlan per-operation device results remain the placement evidence owner;
   neither model compilation nor successful prediction establishes ANE-only execution.
10. NativeNeural still selects actual copy pointers and invokes the same prepared model.
    NeuralModelOf/Bind saves/restores compiled files and fixed IO; dynamic internal MIL
    results are in that same compiled model, not a new per-shape artifact or scheduler.

References inspected: Apple coremltools9.0 local MIL slice_by_index/slice_by_size/range_1d
InputSpec (runtime operands), MIL Dimension.unknown, installed CoreML SDK fixed/ranged
MLMultiArray shapes, outputBackings and MLReshapeFrequencyHint. Apple's flexible-input
guide only says Infrequent can enable ANE; SDK says shape changes can trigger re-optimization.
No hardware support, reduced physical work, performance or zero-shape acceptance claim.

- https://apple.github.io/coremltools/mlmodel/Format/MIL.html#dimension
- https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html
- https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html

After paired boundary review: implement in the SAME owner, run private strict source syntax,
publish exact changed fields/limitations and HOLD. Canonical build and one selected-ANE
existing pipeline client are coordinated separately; no numerical source/placement changes.
