# ANE runtime shape consumer (ane-runtime-shapes.md) — SOURCE-READY, HOLD

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5, on top of metal e9912b93.
Supervisor source checkpoint: metal `38f67112`; no source edits after the independent strict check.
- Scope: metal-microbench `metal_neural.mm`, the `Mil` and `Lowering` owner only.
- **Unchanged:** public ABI, NeuralIO, placement, numerical formulas.
- No build, import, target run, harness or commit. Original malformed sources untouched.

## Changes (`metal_neural.mm`)

1. **TensorType.** A shape entry of `-1` emits MIL `Dimension.unknown` (`dim` field 2). No constant size is cast from
   `-1`. Static entries are unchanged.
2. **Mil.**
   - `Broadcast` treats `-1` as runtime.
   - New direct-op helpers, each one existing MIL operation:
     - `Join` (concat axis 0), `First` (`reduce_max`, `keep_dims` false: the rank-0 scalar);
     - `Fill` (fill with a const value), `Range` (`range_1d(0, end, 1)`);
     - `Reshaped` (reshape by a runtime shape vector), `Sliced` (`slice_by_index` with runtime begin and end).
3. **Lowering domain.** Built per operation by `Domain` from `Operation.symbols`/`extents`:
   - `scalars[s]` is `Need(symbols[s])`, as i32 `[1]`: an earlier result of this function, or the actual Neural input.
   - Loop l is static (`loops[l] = bound`) unless `extents[l]` names a symbol. Then `loops[l] = -1` and
     `extent[l] = minimum(maximum(symbol, 0), bound)`, the extent `metal_ir_loop_extent` defines.
   - `Extent(l)` and `LoopShape` build actual-size shape vectors, `ToLoops` reshapes, and `Spread` broadcasts. Static
     geometry keeps tile and constant reshape; runtime geometry uses add or logical_or with a zero `fill`.
   - `Stack` packs coordinate columns.
   - `Coordinate` evaluates a map's Expr/Term over the domain: its constant, plus coefficient × floor-div/floor-mod per
     term over the actual iotas and the symbol scalars, as metal_ir.mm `Evaluate`/`Term`/`Quasi`.
4. **Iota.** A runtime loop is `range_1d` over its actual extent, reshaped to its loop axis. A static loop keeps its
   constant.
5. **Operand.**
   - **Structured affine slices:** loop terms plus symbol offset terms (divisor ≤ 1, no modulus). Runtime begin is
     `constant + Σ c·symbol`, and end is `begin + (E − 1)·stride + 1`, through `slice_by_index`; then the existing
     transpose and a reshape to actual loop extents.
   - **General maps:** a map referencing a symbol or a runtime loop uses actual-size `Coordinate` columns and
     `gather_nd`. A fully static map keeps the compile-time `Evaluate` indices. The old "operand map over runtime
     symbols" failure is gone.
   - **LogicalRead:** builds its indices through `Stack`.
6. **Lower, Compute and the zero extents.**
   - The former Lower body is now `Compute`. Under a runtime domain:
     - prepared materialized inputs are clipped (`Clip`) to the actual box;
     - the constant-weight specialization is not taken;
     - contributions use `Spread` over the actual space.
   - `Lower` builds `parallel` and `reduction` predicates, rank-0 bool: all runtime extents of that kind > 0. It then
     uses true MIL `cond` blocks (`Cond`/`Branch`, op field 4):
     - **Parallel cond:** the true branch runs Compute, then Finish (start/ACCUMULATE, epilogue, Store) and yields each
       output. The false branch yields each output's prior tensor: no write.
     - **Reduction cond, inside it:** the true branch runs Compute. The false branch (`Empty`) fills each output's
       combiner identity over the actual parallel shape, as metal_lower.mm `Identity`: SUM/ANY 0, PRODUCT/ALL 1, MAX
       −inf (INT32_MIN), MIN +inf (INT32_MAX). Finish's ACCUMULATE, epilogue and Store follow.
   - `Branch` restores `values`/`rawValues`, so no block-local value escapes. Function inputs made inside a branch are
     kept.
7. **Contraction.** Under a runtime domain, B/M/N/K are products of the actual extents. The existing matmul takes
   runtime reshape vectors and gives unknown result dimensions. The layout reshape uses actual extents, then the same
   transpose. The conv specialization is static-only.
8. **Store.**
   - The permutation shortcut requires every PARALLEL loop to be static (the actual full domain) and no symbol term.
   - Otherwise a map with runtime parallel loops or symbol terms scatters:
     - `Coordinate` destination indices over the actual parallel domain, reshaped to `[−1, rank]`;
     - updates reshaped to `[−1]`;
     - `scatter_nd` mode update into the prior full physical tensor, so every inactive element is kept.
   - Static maps keep the compile-time indices.
9. **LowerNeural `outside()`.** A value that another non-member operation reads as one of its `symbols` stays an
   observable output.

## Limits (explicit failures, not padded or substituted)

- **Coupled REGION states** (`op.states`) with a reduction and any runtime loop domain are rejected by the current
  `runtime && !reduce.empty()` check. This includes a runtime parallel dimension with static reduction width.
  The present diagnostic names the COMBINE tree; this is an incomplete consumer, not a hardware limitation.
- **Runtime REDUCTION** for ARGMAX, ARGMIN or ASSIGN is rejected during construction of the empty branch, even when
  a particular invocation's extent would be positive. Its empty-domain result is not implemented in this MIL owner.
  This is not evidence that the source identity is unavailable elsewhere in the compiler.
- **No placement claim.** Core ML compilation or prediction is not ANE execution. The existing `MLComputePlan`
  per-operation devices in EmitNeural stay the placement evidence.
- **Not established:** hardware support for unknown-dimension or `cond` operations on the Neural Engine, any reduction
  in work, and performance.

## Check

```
cd /Users/mdot/metal-microbench
clang++ -x objective-c++ $(TINT_CFLAGS) -fobjc-arc -O2 -Wall -Wextra -Werror -fsyntax-only -I . -iquote .build \
  -I ../dox/mesh/rdma metal_neural.mm
```
- Run with the Makefile's actual `TINT_CFLAGS`: rc=0.
- `git diff --quiet HEAD -- metal_recording.m`: unchanged.
- Supervisor repeated the strict command independently: rc=0; `ane-runtime-shapes-syntax.log` is empty.

**References:**
- coremltools 9.0 local op defs: iOS15 `range_1d`, `tile`, `fill`, `concat`, `cond`, `gather_nd`, `scatter_nd`; iOS17
  `reshape`, `slice_by_index`;
- MIL.proto Dimension: https://apple.github.io/coremltools/mlmodel/Format/MIL.html#dimension

**Changed:** `metal_neural.mm` only. HOLD.
