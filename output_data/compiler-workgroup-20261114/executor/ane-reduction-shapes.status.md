# ANE runtime reduction consumer (ane-reduction-shapes.md) — SOURCE-READY, HOLD

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5, on top of metal 38f67112.
- Scope: `metal_neural.mm`, the `Mil` and `Lowering` owner only.
- **Unchanged:** ABI, NeuralIO, placement, numerical formulas.
- No build, import, target run, evaluator or commit. Original malformed sources untouched.

## Changes (`metal_neural.mm`)

1. **Typed identities.** References: metal_lower.mm `Reduction::Begin`/`Results`, `Identity`.
   - `Mil::Bits(dtype, bits)` is an exact typed constant from retained bits: f16 low 16 bits, bool `bits != 0`, other
     dtypes low 32 bits. So u32 identities keep their 32-bit representation in MIL i32.
   - `Mil::Fill(shape, dims, value)` fills with that constant.
   - `IdentityBits(combiner, accumulator type)` mirrors `Identity` by the original compute type:
     - PRODUCT/ALL → one;
     - MAX/ARGMAX → lowest: −inf for f16/f32, INT32_MIN for i32, false for bool, 0 for u32;
     - MIN/ARGMIN → highest: +inf, INT32_MAX, true, UINT32_MAX;
     - SUM/ANY/ASSIGN → zero.
2. **`Empty`** (zero runtime REDUCTION extent, in Lower's existing reduction `cond`), over the actual parallel shape
   with reduction dims 1:
   - **REGION:** each `states[s]` is filled with `identities[s]` bits (default 0), typed by `ComputeType(states[s])`,
     then the same FINALIZE region.
   - **ARGMAX/ARGMIN:** the index identity is `UINT32_MAX`, as an i32 bit pattern.
   - **Other combiners:** `IdentityBits` by `op.accumulator[j]`.
   - Results are spread and cast to the true branch's types. Finish (start/ACCUMULATE, epilogue, Store) follows as
     before. The earlier "no retained MIL identity" failure is removed.
3. **Nonempty ASSIGN with REDUCTION loops** (`Compute`). `Combine`'s last contribution is a `slice_by_index` of the
   contribution at each REDUCTION loop's last actual index (`Extent − 1`), with parallel extents actual. This replaces
   the former return of the unreduced tensor, statically too.
4. **Coupled REGION tree under a runtime domain** (`Tree`, from `Compute`). Any runtime loop, parallel or reduction,
   takes this path. The static tree is unchanged for fully static domains.
   - **Inputs:** `parallelAxes` keep their original loop ordinals, and `sizes[j] = Extent(parallelAxes[j])`. The
     actual count is `R = Π Extent(r)` over the actual reduction axes.
   - **Reshape:** each state is transposed parallel-first, reduction-last, and reshaped by `[sizes…, R]`, declared
     `[flat…, −1]`.
   - **Width:** `W = least power of two ≥ R`, from a MIL `while_loop` carrying i32 width from 1. The cond is
     `width < R`, the body `width·2`.
   - **Identity tail:** a true `cond(W − R > 0)` appends exactly `W − R` typed identity elements (`identities[s]`
     bits) along axis P by concat. The false branch keeps the state, so there is no zero-size fill and no bound-sized
     padding.
   - **Tree loop:** a `while_loop` carries `(width, state₀…state_{S−1})`, each state of rank P+1 with an unknown last
     dimension.
     - The cond is `width > 1`.
     - The body takes `half = width / 2`, slices `[0, half)` and `[half, width)` of each state by runtime begin/end
       vectors, and evaluates the SAME COMBINE region on `(a…, b…)`. It yields `(half, merged…)`, cast and spread to
       the carried type. This is the existing half/half order and identity placement.
   - **Compact domain:** during COMBINE, `loops`/`extent` are swapped to `[sizes…, half]`, so Spread/LoopShape/Stack
     use the actual compact axes. The outer domain is restored afterwards.
   - **After the loop:** each state is reshaped to rank L (parallel actual, reductions 1). The SAME FINALIZE and Finish
     follow. The earlier coupled-tree rejection is removed.
5. **`Mil` `while_loop` encoding** (`While`; coremltools 9.0 iOS15 control_flow.py `while_loop`):
   - `loop_vars` is one repeated argument.
   - There are two blocks (op field 4). Each has its own NamedValueType inputs (Block field 1), one per loop var, and
     outputs (field 2) through identity ops declaring the loop-var types.
   - The cond returns one scalar bool.
   - `values`/`rawValues` are restored after each block, keeping only function inputs.

## Prior-output review (Branch/Cond/Store)

- `Cond` builds the true branch, restores the incoming `values`/`rawValues`, then builds the false branch. A
  parallel-empty false branch therefore yields the original incoming prior tensors, never a true-branch Store result.
- Store's scatter source inside the true branch is that same incoming prior, so inactive entries are kept.
- Function inputs made by `Need` inside a branch stay global. Branch-local results escape only through `cond` or
  `while_loop` outputs.

## Limits

- **Aliases.** `values` is keyed by ValueID. An output aliasing storage that an earlier operation in the same kernel
  wrote under a different ValueID reads its prior from the function input, not that write. This is an existing owner
  limit: no alias solver was added.
- **Unsigned comparison.** Nonempty MAX/MIN over a u32 accumulator still uses MIL's i32 `reduce_max`/`reduce_min`, as
  before. Only the empty identities are typed here.
- **No placement or hardware claim.** Core ML compilation or prediction is not ANE execution, and `MLComputePlan`
  stays the placement evidence. There is no claim about `while_loop`, `cond` or unknown dimensions on the Neural
  Engine, or about performance.

## Check

```
cd /Users/mdot/metal-microbench
clang++ -x objective-c++ $(TINT_CFLAGS) -fobjc-arc -O2 -Wall -Wextra -Werror -fsyntax-only -I . -iquote .build \
  -I ../dox/mesh/rdma metal_neural.mm
```
- Run with the Makefile's actual `TINT_CFLAGS`: rc=0.
- `metal_recording.m` is unchanged.
- Supervisor removed one added prose comment in favor of the boundary-document citation, then independently repeated
  the exact strict command: rc=0, empty `ane-reduction-shapes-syntax.log`. No later source mutation.

**Changed:** `metal_neural.mm` only. HOLD.
