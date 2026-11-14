# ANE storage-backed SSA views (ane-storage-views.md, final corrections) — SOURCE-READY, HOLD — supervisor checkpoint 7f7ed436

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5, on top of metal ff3609f7.
- Scope: `metal_neural.mm`, the `Lowering` owner and LowerNeural only.
- **Unchanged:** public ABI and NeuralIO runtime contract, placement, algorithms.
- No build, import, target run, evaluator or commit. Original malformed sources untouched.

## Source changes

1. **Class owner** (`Lowering::Storage`, `storage`, `classes`). There is one canonical storage-backed SSA tensor per
   actual alias class (`Value.storage`), as a fixed 1-D physical span.
   - **`Classes(k)`** runs once before lowering. It walks the kernel's non-erased operations' `inputs`/`outputs`/
     `gathers`/`symbols`, skipping immutable constants.
     - **Element kind:** taken from the live views' scalars. Every live view must share the MIL dtype and element
       width. I32/U32 share i32.
     - **Span:** max(`Bytes(p, root)`, `Bytes(p, view)` of every live view) ÷ the element bytes, rounded up.
     - **Flat alias:** a new typed Value of the class (`metal_ir.h` Value) with scalar the live kind, shape `[span]`,
       dense layout `{dim 0, coefficient 1, divisor 1}`, offset 0, `storage` = root, and the root's role.
     - The root's declared byte type sets only its reach. It is not an interpretation.
   - **`Read(value, coordinates, shape)`** gathers from the class's current tensor at
     `Physical = Value.offset + Σ Term(layout, logical coordinate)` (metal_ir.mm `Element`, `Term`). The coordinates
     are the operation's actual `Coordinate` tensors.
   - **`Write(value, coordinates, shape, y)`** scatter-updates (`scatter_nd`, mode update) the class's current tensor at
     the same composed physical indices. Each update consumes the prior whole class tensor. There is no last-writer
     search or overlap test.
2. **Consumers.**
   - **Operand:** a mutable view's composed actual-size coordinates go through `Read`.
   - **LogicalRead:** a mutable source is a `Read` at its region coordinates.
   - **Domain symbols:** a mutable scalar is a `Read` at the origin.
   - **Store:** all stores are a `Write` over the actual parallel domain, so ACCUMULATE's prior (Operand) sees
     accumulated partial updates.
   - **Immutable constants** (CONSTANT with data), including quantized decode, keep the existing `Need`/`Constant`
     paths unchanged.
   - **`Need`** for a mutable view now fails ("mutable view read outside its storage class"). It no longer makes a
     per-view input.
3. **Branches and loops.**
   - `Branch` and `While` snapshot and restore `storage` with `values`/`rawValues`, so block-local class tensors never
     escape.
   - In Lower's parallel `cond`, each written class of the operation's outputs is initialized outside, then yielded
     once:
     - the true branch yields Finish's accumulated class tensor;
     - the false branch yields the incoming class tensor.
     - After the cond, the class's current tensor is the cond output, so there is no per-view output state.
4. **Fixed IO** (LowerNeural; the internal decision is in the lifecycle section below):
   - **Inputs:** each non-internal class first read or partially written is one model input `in_<flat>`, bound through
     the existing `bind` to the flat alias: offset 0, `[span]`, stride 1, the actual access.
   - **Outputs:** each written non-internal class is one output `out_<flat>` of its final SSA tensor.
   - **Bytes:** `Binding.bytes = Bytes(p, flat)`, so Artifact's later `Realize` folds the same flat reach into the
     root. `p.realized` is not read.
   - **Unchanged:** the original source Values stay immutable logical views, and operation identities are unchanged.
     NeuralModelOf/Bind already persist these fixed descriptors.

## Lifecycle choice: incoming versus internal authority (final, after ane-storage-review78.md)

**Membership.** `outside()` in LowerNeural is computed before lowering over the same maximal-run membership:
`ExternallyVisible`, or any non-member non-erased operation whose `inputs`, `outputs`, `gathers` or `symbols` names the
class. It feeds only `Classes`. There is no role-only, read-history, alias or allocator inference.

**Initializer authority.** `Classes` scans the original Values once by storage root. If any `.data`, alias-attached
data included, or `.zeroed` names the root, the class is initialized:
- it stays native-backed;
- the existing native initialization consumes its original payloads;
- there is no MIL copy, repack, clamp or shift. The former Current branch that copied `Contents(root)` into a MIL
  constant is removed.

**Internal (whole class).** A class is internal only when it has no declared initializer, and either
`Internal(p, root, kernel)` already holds or the kernel writes it with `outside(root)` false.
- **Marking:** every original Value of that root, and the appended flat view, get `internal = kernel`, once, in
  `Classes`. Artifact's later `Realize` therefore excludes its backing.
- **No output decision:** the former per-output `!outside(value)` internal loop in LowerNeural is removed, so no
  independent output-view decision overrides this one.
- **Incoming tensor:** a local zero `fill([span])`. This refines source contents the program leaves undefined. No proof
  is supplied that each element is written before it is read; confinement is not such a proof.

**Not internal.** Every other touched class has its actual native backing as its incoming tensor, through the one fixed
flat model input:
- initialized classes;
- classes touched by another kernel;
- externally visible classes;
- classes only read here.

Each one written here is one fixed flat output. Immutable CONSTANT classes and their quantized decode are unchanged.

## Unsupported (explicit failures)

- **Mixed live element interpretations of one class** (for example F32 and I32 views, or 16- and 32-bit views):
  "storage R: live element kinds a and b (no bit reinterpretation)". There is no numeric reinterpretation cast and no
  split into unrelated tensors.
- **Class kinds without a Core ML tensor or MLMultiArray type** (bool, 64-bit and similar) fail at class or bind time,
  as before.

## Limits

- Core ML compute partitioning, model input/output overlap behaviour and actual work are not established.
- There is no hardware, placement or scaling claim. MLComputePlan, and the compiler's requested-ANE 2×2 Hermitian
  client after a matched build, remain the gate.

## Check

```
cd /Users/mdot/metal-microbench
clang++ -x objective-c++ $(TINT_CFLAGS) -fobjc-arc -O2 -Wall -Wextra -Werror -fsyntax-only -I . -iquote .build \
  -I ../dox/mesh/rdma metal_neural.mm
```
- Run with the Makefile's actual `TINT_CFLAGS`: rc=0.
- `metal_recording.m` is unchanged.

**Changed:** `metal_neural.mm` only. HOLD.

Supervisor final review: `initialized` guards both internal paths; one pass over original plus appended Values stamps the accepted internal classes. Strict syntax was rerun after the final citation-only source cleanup and passed with an empty log. Metal checkpoint `7f7ed436`; no shared build or target was run.
