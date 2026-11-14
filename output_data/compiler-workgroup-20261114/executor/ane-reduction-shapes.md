# Runtime MIL reduction continuation — exact operands and result

Continue SAME e48, metal `38f67112`, only `metal_neural.mm`; no build/import/target/commit.
Compiler authorized the remaining reduction consumer. Publish this mapping before code; no ABI change.

Source authority is the same Operation: iterators, actual Domain extent tensors, states, identities,
accumulator types, combiner, start, combine and finalize. Native reference is metal_lower.mm Reduction::Begin
and Results (475–491 and 545–553), plus Identity. Do not reinterpret a missing MIL consumer as absent source data.

Empty reduction result:
- REGION: exact typed bits from states[s]/identities[s] (default bits0), broadcast over actual parallel
  extents with reduction dimensions1; invoke existing FINALIZE with those states, then existing Finish.
- ARGMIN/ARGMAX: returned index identity is UINT32_MAX; preserve its 32-bit representation in MIL i32.
- ASSIGN: existing Identity defaults to typed zero. Preserve start/ACCUMULATE and epilogue through Finish.
- MIN/MAX identities depend on the original accumulator type, not merely MilType (which maps U32 to i32).
  Reuse exact source identity semantics, including bool and typed floating infinities.
- Nonempty ASSIGN with reduction must match existing Combine's last contribution; current returning the
  full unreduced tensor cannot become a correct runtime scalar just by enabling its empty branch.

Coupled tree operands:
- `parallelAxes` and `reduceAxes` retain original loop ordinals.
- `parallelSizes[j] = Extent(parallelAxes[j])`, with static/unknown result-type dimensions retained.
- `R = product(Extent(r))` for the actual reduction axes, never their upper bounds.
- Existing tree requires `W = smallest power of two >= R`. For positive dynamic R, the existing
  MIL while_loop can carry scalar i32 width, start1, condition width<R, body width*2.
  This reproduces the current power-of-two tree; it is not a new reduction algorithm or schedule.
- Initial state[s] is the actual contribution transposed parallel-first/reduction-last and reshaped to
  `[parallelSizes..., R]`. Append exactly W-R typed identity elements only when nonzero, using a true cond
  if needed to avoid zero-size fill. No static bound-sized padded tensor.
- Tree while_loop carries `(width:i32 scalar, state[0],...,state[S-1])`; every state has fixed rank P+1,
  source state dtype, the retained parallel dimensions, and an unknown final dimension.
- Condition block returns scalar `width>1`. Body sets half=width/2, slices first and second halves of
  each state using runtime shape/index operands, invokes the SAME COMBINE region, and yields
  `(half, merged states)`. This preserves the existing half/half tree and identity placement.
- After exit, reshape each state to original rank L (parallel axes actual, reduction axes1), run the
  SAME FINALIZE, then Finish. Static reduction width with dynamic parallel axes uses this same shape
  owner or the current static tree with actual parallel shape operands; no blanket rejection.
- Region/Stack/LoopShape inside the flattened COMBINE must use its actual `[parallelSizes..., half]`
  correspondence, not assume compact parallel axis j is original source loop j. Restore outer domain
  after nested emission. No source operation IDs, placement, or native repeat/protocol events change.

Existing MIL schema (coremltools9.0 iOS15/control_flow.py while_loop): `loop_vars` is a tuple of tensors;
cond and body are two blocks, each with same loop-var input arity. Cond returns one scalar bool; body returns
same arity/types with compatible shapes. Changing dimensions are unknown in the carried type; fixed rank
and dtypes remain. Existing Operation blocks field4 and Block inputs1/outputs2/operations3 encode this.

Prior-output review: before either zero-domain branch, preserve the same incoming values/rawValues mapping.
Function input bindings created via Need inside a branch may be retained globally, but branch-local results
must not escape except via cond outputs. A parallel-empty branch yields the original prior full tensors.
Store's scatter source is that incoming prior, preserving inactive entries; do not read a true-branch update
as the false branch's prior. Inspect actual alias cases supported by current owner, without adding a new alias solver.

Use only existing MIL operations and retained fields. No numerical formula change, new scheduler/cache,
whole-model clone, outer shape variants, fallback claim, or source placement change. Private strict syntax
after coherent implementation, publish exact limits and HOLD. CoreML compile/predict success still does not
establish ANE execution; actual MLComputePlan/caller evidence remains a separate coordinated gate.
