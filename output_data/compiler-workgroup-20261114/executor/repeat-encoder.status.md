# Repeat source encoder and actual leaf correspondence: status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`. Unit:
[repeat-encoder.md](repeat-encoder.md), at mesh HEAD `e0f52110`. These names were published before the source. No
import, build or run.

## Producers consumed (source-live)

- **`metal_ir.h:269–303`:**
  - `uint32_t metal_ir_repeat(p, first, end, condition, carried_count, const uint32_t *carried)` returns a region
    index, not an operation;
  - `struct metal_ir_repeat_region` (10 × u32, 40 bytes);
  - `metal_ir_repeats`, `metal_ir_repeat_carried`, `metal_ir_dispatch_repeats`;
  - the kind `METAL_IR_REPEAT` = 10.
- **`operations.py` (numerical64):** `_Repeat(records, condition, carried)` and the recursive `_placed`, used as they
  are.

## Builder result names (`compiler/mesh_compiler/metal.py`)

- **`Emitted(record, operation, first, end)`:** one leaf record's actual emission by `_Builder.generic`.
  - `record` is the operations.py record itself, the source owner.
  - `operation` is its returned operation ID.
  - `[first, end)` is the operations it created (`metal_ir_operations` before and after; a sort creates many).
- **`Repeated(record, region, first, end)`:** one registered repetition.
  - `record` is the `_Repeat`.
  - `region` is its `metal_ir_repeats` index, never an operation.
  - `[first, end)` is its pass's interval, nested children included.
- **`_Builder.emitted` / `_Builder.repeated`:** the lists in emission order. A nested region is registered before its
  enclosing one (metal_ir.h: equal intervals, the later registered is the outer).
- **`_Builder.generic(g)`:**
  - for a leaf record, it encodes it as before (the existing body, renamed `_Builder.encoded`, unchanged, 64's Sort
    branch included), appends its `Emitted` and returns its operation;
  - for a `_Repeat`, it calls `_Builder.repeat(g)`: each pass record through `generic` recursively, then
    `metal_ir_repeat(first, end, value(condition), carried roots)` once. It returns None, because a repetition is no
    operation.
- **ABI:** `RepeatRegion` (ctypes, the canonical 40-byte layout). `_repeats(lib)` binds the four repeat symbols on
  first use, so a program with no repetition never resolves them.

## lower_metal consumer (`compiler/mesh_compiler/lower_metal.py`)

- **Per top-level record:** `made` is its actual `[before, after)` interval, every created child included. An empty
  interval adds only a real returned operation, never None or UINT32_MAX.
- **Work axes:** after emission and before construction, each `work_axes` node's references are matched against the
  `Emitted` leaves of its own records (`_Builder.emitted` slices).
  - An operand occurrence (`_Generic`/`_Lapack` inputs then outputs, or `_Generic.source_axes`) attaches to that leaf's
    own `operation` with that operand ordinal.
  - A reference with no occurrence in any of its leaves attaches as NONE on each created operation of the node's
    records.
  - Region IDs never reach `metal_ir_operation_axis`.
- **`ConstructedMachine.emitted` and `.repeated`:** the Builder's retained facts.
- **Not resolved here:** a region's Work multiplicity (the pass count) stays unresolved, with no fixed count.

## Source (working tree; SOURCE-READY, HOLD)

**`metal.py`:**
- **Imports:** `collections`, and `_Repeat` from operations.py.
- **`RepeatRegion`:** a ctypes struct of 10 × u32.
- **`REPEATS`/`_repeats(lib)`:** binds `metal_ir_repeat`, `metal_ir_repeats`, `metal_ir_repeat_carried` and
  `metal_ir_dispatch_repeats` on first use. They are not in `_compiler`'s eager table.
- **`Emitted`, `Repeated`:** namedtuples.
- **`_Builder`:**
  - `emitted` and `repeated` lists;
  - `generic`: a `_Repeat` goes to `repeat`, and a leaf goes to `encoded` with its `Emitted` appended and its operation
    returned;
  - `repeat`: the records through `generic`, then one `metal_ir_repeat(p, first, end, value(condition),
    len(carried), roots)`, its `Repeated` appended, returning None;
  - `encoded`: the former `generic` body, unchanged (64's Sort branch included).
- Frontend41's `Numeric`/`dispatch_numerics`/`operation_axis` hunks are untouched.

**`lower_metal.py`:**
- **`associate(node, leaves, made)`:** runs after emission, over the node's `Emitted` leaves.
  - A matched operand goes on its leaf's own `operation`.
  - A reference with no occurrence anywhere gets NONE on each operation the node's records made.
  - It calls `metal_ir_operation_axis` directly. The per-record-index `axes` map is removed, and the nodes it covers
    (`associated`) are exactly those the `_instruction` branch covered before.
- **Emission loop:** `made` is the interval, or the returned operation only where it is real (not None or NONE).
  Each node's leaves come from `builder.emitted[first:]`.
- **`ConstructedMachine`:** gains `emitted` and `repeated` (`tuple(builder.emitted)`, `tuple(builder.repeated)`).
- 974's and 64's other lines are unchanged. Two lines over 120 characters (95, 172) predate this unit and are
  untouched.

**Not touched:** native.py, archive.py, `Bound`/observer, operations.py (`_Repeat` and `_placed` are 64's), and every
C source file.

## Checks (AST and source only)

- `py_compile` and `pyflakes` pass for `metal.py` and `lower_metal.py`, `git diff --check` is clean, and no added line
  is over 120 characters.
- The symbol scan finds 97 bound names. The four repeat symbols are each declared in `metal_ir.h`.
- **`RepeatRegion` layout:** the class body compiled from metal.py's AST against plain ctypes has size 40, `test` at
  24, `order_first` at 28 and `order_last` at 32, which equals the header's `_Static_assert`.
- **Callers:** `lower_metal`'s loop and native's composite copies are the two `generic` callers. Native passes copy
  leaves only and is unaffected.
- **Open:** the repeat region's Work multiplicity stays unresolved; no pass count is assumed.
