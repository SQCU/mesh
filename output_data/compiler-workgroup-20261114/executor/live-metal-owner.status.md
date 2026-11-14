# Live native ABI owner (`compiler/mesh_compiler/metal.py`): status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`. Unit:
[live-metal-owner.md](live-metal-owner.md). **Owner source ready (section 5); paired operations and lower_metal source are now live.** No shared build or
invocation is claimed by this checkpoint. `rdma/mesh_compile.py` stays untouched, poisoned and unimported (`ast.parse` fails, as intended).

## 1. For root593: what `_Builder` imports from `operations.py`

`operations.py` owns these, and `metal.py` imports them with `from .operations import …`. `operations.py` imports
nothing from `metal.py`.

| Name | `_Builder` reads |
|---|---|
| `_key(t)`, `_stored(t)` | `_key`: a tensor's storage identity. `_stored`: the actual real tensor view (a complex tensor's real-pair view), with its stride and offset kept. It is no whole-storage normalization; `_Builder.whole` makes the storage root from it |
| `_Words` | `.tensor`, a packed view (two words per element) |
| `_Check` | `.node`, `.info` (tensor), `.code`, `.message` |
| `_Lapack` | `.node`, `.place`, `.batch`, `.inputs`/`.outputs` (`(tensor, rank)` pairs), `.routine`, `.job`, `.rcond` |
| `_Sort` | `.node`, `.x`, `.axis`, `.k`, `.descending`, `.values`, `.indices`, `.place` |
| a generic record (`_Generic`) | `.node`, `.inputs`/`.outputs` (`(tensor, rows or None)`), `.sizes`, `.iterators`, `.combiner`, `.start`, `.accumulator`, `.precision`, `.place`, `.states` (`(kind, identity)`), `.body`, `.combine`, `.finalize` |
| a region (`_Region`) | `.code`, the builder records `(call, operands…)` with calls `parameter`, `loop`, `literal`, `binary`, `unary`, `convert`, `bitcast`, `builtin`, `extract`, `extract_if`, `scatter` and `yield`, and `.kinds` |
| `_rows(dims, picks)`, `_bits(value, kind)` | Map rows for a LAPACK operand; literal bits per kind |
| `SCALAR`, `KINDS` | dtype → `metal_ir.h` scalar code; code → name (for the builder's error text) |
| `CPU` | Placement code (`_Lapack.place != CPU`) |

**One owner for the numerical enums: `operations.py`.** Construction records carry these codes, so `operations.py`
must hold the tables:
- the scalar codes `BOOL…F64` with `KINDS`/`SCALAR`/`CLASS`/`STORED`/`REAL` and the dtype groups;
- `PARALLEL`/`REDUCTION`, `ASSIGN`/`SUM`/`REGION`, `IDENTITY`/`ACCUMULATE`, `REASSOCIATE`/`CONTRACT`;
- `GPU`/`CPU`/`ANE`, used by `_Check`, `_Lapack`, `_Generic`, `_Sort`, `_typed` and `_instantiate`;
- `BINARY`/`COMPARE`/`BOOLEAN`/`COMPLEMENT`/`NEGATION`/`NOT` and `BUILTINS`.

A scan of the construction functions finds none of them using the roles, the region kinds, `NONE` or `Term`. So
those stay ABI names in `metal.py`.

Also root's, consumed by `lower_metal`/`driver`, not by `metal.py`: `Generator`, `seeded`, `_stepped`,
`_instruction`, `_local_call`, `_copy`, `_held`, `_placed`, `_metadata` and `_same_view`.

## 2. `metal.py` contents (live responsibilities only)

| Group | Names | Consumers |
|---|---|---|
| ABI enums | `INPUT, CONSTANT, TEMPORARY, OUTPUT, STATE`; `BODY, COMBINE, FINALIZE`; `NONE`; `METAL_IR_STEP` | `lower_metal` (roles), `_Builder`, `native` |
| ctypes structs | `Term`, `Expr`, `Matrix`, `Lapack`, `ValueInfo`, `Storage`, `_Actors`, `Range`, `Release`, `Sending`, `Transport`, `Replay`, `Call`, `Span`, `CheckFailure` | `_Builder`, `native` |
| Libraries | `_repository`, `_compiler` (libmetal_ir binding table), `_recorder` (libmetal_recording), `_transport` (mesh-metal), `_machines` | `native`, `lower_metal`, `driver` |
| ObjC/Metal | `OBJC`, `SEND`, `GET`, `PUT`, `SELECTOR`, `_SEL`, `_message`, `OBJECT`, `VALUE`, `STATE_BUFFER`, `_NEW_BUFFER`, `SHARED`, `_metal_device` | `native`, `_Builder` |
| Resources | `RELEASE`, `_array`, `_FREE`, `_INCREF`/`_DECREF`/`_RELEASED_OBJECT`/`_lent`, `_archive_loaded`, `_saved_bytes`, `_placed_buffer` | `native`, `bridge` (`_lent`), `_Builder` |
| Run | `START`, `STARTED`, `_wait`, `_execution_error` | `native.Bound` |
| Typed encoder | `_Builder` | `lower_metal.construct` |

Not carried: `_bound`, `program`, `load`, `_runner`, `_replays` and every closure orchestration, plus the dead
`_programs`/`_actors`/`_lifetimes`/`_finalized`/`_expanded`/`_period`/`_outputs`/`_specialize`/`_save_program`/
`_restore_programs`. The binding tables keep only the symbols that live callers use.

## 3. Names other owners must import from `metal.py`

- `lower_metal` (root, frontend): `_Builder`, `_compiler`, `_machines`, the roles `INPUT, CONSTANT, TEMPORARY,
  OUTPUT, STATE`. Its `GPU/CPU/ANE` and construction names come from `operations.py`.
- `driver` (frontend974): `_machines`, from `metal.py`. `Generator` and `seeded` come from `operations.py`.
- `bridge` (compilerdc4): `_lent`, from `metal.py`.
- Library names stay current until native e48's live-native-owner status changes them: `.build/libmetal_ir.dylib`,
  `libmetal_recording.dylib` (`DYLD_INSERT_LIBRARIES`/`find_library`), and `rdma/libmesh.dylib` through `rdma/mesh.py`.

## 4. Implementation (working tree; source-only checks)

**Source files:**
- `compiler/mesh_compiler/metal.py` (new, 613 lines);
- `compiler/mesh_compiler/native.py`.

`archive.py` needed no change: it imports only `native` and `ir`.

- **`metal.py`:** the section 2 groups. Each definition was selected from its live span in `rdma/mesh_compile.py` at
  `039ad8ed`, which is unchanged, still poisoned, and never imported, sliced or executed.
  - **One binding table per library:** `_compiler` binds 53 `metal_ir_*` symbols, the live set that `native.py`,
    `lower_metal.py` and `_Builder` call. It adds `metal_ir_value_layout`, `metal_ir_value_external`, `metal_ir_sort`
    and `metal_ir_extract_if`, which callers used to rebind at each call.
  - **Unused symbols dropped:** `metal_ir_archive_solved`, `_records`, `_retain`, `metal_ir_values`,
    `metal_ir_operations` and `metal_ir_value_storage`.
  - **`_recorder`:** 16 symbols, including `MetalRecorderActors`/`Free`, `MetalProgramSignatureNode`,
    `MetalProgramExternals` and `MetalProgramRunArguments`, now bound once here. It drops `MetalReplayNative`,
    `MetalProgramHorizon` and `MetalProgramRun`, which have no live caller.
  - **`_transport`:** its eight `mesh_metal_*` symbols.
  - **Moved here:** `Argument`, `ExternalUse` and `METAL_EXTERNAL_GPU`, from `native.py`.
  - **Dropped as having no live caller:** DLPack, `_dlpack_tensor`, `WRAP`'s selector, and the `newBufferWithBytesNoCopy`
    selector.
  - **`_Builder`:** its methods unchanged, except that `metal_ir_sort`/`metal_ir_extract_if` come from the table.
    Its numerical records and tables are imported from `.operations`: `CPU, GPU, KINDS, SCALAR, _Check, _Lapack,
    _Sort, _Words, _bits, _key, _rows, _stored`.
- **`native.py`:**
  - `from . import metal`, `from .metal import METAL_EXTERNAL_GPU, Argument`.
  - `libraries()` returns `(metal, metal._compiler(), metal._recorder(), metal._transport())`, and its runtime
    rebinding is gone.
  - `Solution.loaded` uses `metal._lent`.
  - Citations of the live helpers point to `metal.py`. Citations of dead functions (`_lifetimes`, `_actors`,
    `_programs`, `_finalized`) now name the design clause they cite.
  - No `mesh_compile` import remains. The joint `Body`/`Row`/`Operand` source is unchanged.

**Checks:**
- `py_compile` passes for both files, and no line is over 120 characters.
- An AST check finds `metal.py` uses no undefined name. The imports from `.operations` are exactly the section 1 list.
- `/tmp/archive-worker/scripts/metal_owner_check.py` finds:
  - every `M.*`/`metal.*` name in `native.py` defined in `metal.py`;
  - every `compiler.*`/`lib.*`/`transport.*` call bound in `metal.py`'s tables and declared in `metal_ir.h`,
    `metal_recording.h` or `mesh-metal.h`;
  - no `mesh_compile` import in `native.py` or `archive.py`.

  The one exception is LiteRT41's three observation symbols (below).
- **Not run:** no import of `metal.py`. `operations.py` doesn't exist yet, and the canonical build waits.

### Paired source connection

The final batch below centralizes the observation and source-relation ABI. Root593 has now supplied the real
`operations.py` and changed `lower_metal` to direct imports from both owners. `driver` imports `metal`, and `bridge`
imports `metal._lent`. The joint compiler/loader connection and coexisting publication work remain active separate
units; this ABI extraction does not claim those are complete.

## 5. Final owner batch (supervisor additions; source-ready, HOLD)

- **Source ABI:**
  - `metal.SourceOperand {uint32 kind, value}` (`struct metal_ir_source_operand`) and `SOURCE_VALUE, SOURCE_LITERAL,
    SOURCE_SENT, SOURCE_RECEIVED, SOURCE_EXTERNAL = range(5)`.
  - `metal_ir_source_relation` bound once in `_compiler`, as `[program, relation, size_t *operation_first, uint32
    *operations, uint32 *view_values, size_t givens, SourceOperand *, RELEASE, owner, error, room] -> int`, from
    `metal_ir.h` 382–385.
  - **For root593:** `lower_metal` takes `metal.SourceOperand` and the `SOURCE_*` codes, keeps only its operand
    mapping, and drops its inline `argtypes`. Its borrowed call must pass `metal.RELEASE()` (a typed null), not
    `None`: `None` raises `ArgumentError` against the `RELEASE` argtype. The owner stays `None`.
- **Observation ABI:**
  - `metal.Unit {uint32 kind, call, site, copies; uint64 slice}`, `metal.UnitSlice {uint64 member, count, identity,
    identities}`, `metal.Observed {double seconds; uint64 executions, residue}` and `METAL_UNIT_GPU, METAL_UNIT_HOST`.
  - In `_recorder`: `MetalProgramRunObserved` (the `MetalProgramRunArguments` signature), `MetalProgramUnits(program,
    uint32 *count, UnitSlice **, uint32 **members, uint64 **identities) -> Unit *`, and
    `MetalProgramObserved(program, Observed *, count, error, room) -> int`.
  - `native.py` no longer declares `Unit`/`Slice`/`Observed`/`METAL_UNIT_*` or `Bound.observe`'s inline prototypes.
    `observe` reads `metal.UnitSlice`/`metal.Observed`. Its invocation, result and observation logic is LiteRT41's
    and unchanged.
- **Header mismatch, for native e48:** this is the agreed form, which `observe` consumes. The working-tree
  `metal_recording.h` 174–183 and `metal_program.m` 1732 still declare the earlier form:
  `MetalUnit {uint32 kind, call, site, count; uint64 member}`, `MetalProgramUnits(program, &count, &members)` and
  `MetalProgramObserved(program, double *seconds, count, …)`, with no `MetalSlice`/`MetalObserved`. The source
  header is the authority, so the observation binding must not be used until e48's header and source carry the agreed
  records. I made no third form.
- **`_Builder.attach`:** the inherited `HOLDERS` id → tensor registry and `_released_constant` are removed.
  - `metal_ir_value_data` (`metal_ir.h` 93–97) returns `void`: the program holds `owner` and calls `release(owner)`
    once at `metal_ir_program_free`.
  - So `attach` lends `base` through the same `_lent`: `Py_IncRef`, then `metal_ir_value_data(…, owner=id(base),
    release=_RELEASED_OBJECT)`. Adoption always holds (`lambda _: True`), because the call has no failure result.
  - The tensor keeps its storage alive until the program's free.
  - `RELEASE`'s comment now names its three C adopters.
- **Checks (AST and syntax only):**
  - `metal.py` and `native.py` compile, with no line over 120 characters.
  - `metal.py` uses no undefined name.
  - `metal_owner_check.py`: every `native.py`/`archive.py` `M`/`metal` name is defined. All 77 bound symbols are
    covered, with none unbound or undeclared by name; the observation argtypes differ from the header, as above.
  - No `mesh_compile` import, and `rdma/mesh_compile.py` is unchanged and still fails `ast.parse`.
- **Owner source ready:** `compiler/mesh_compiler/metal.py`; `native.py` imports and duplicate ABI removal.
  `operations.py`, direct `lower_metal` imports, its typed `RELEASE()` null callback, and driver/bridge live imports
  were inspected by the supervisor on 14 November. No original poisoned module is imported.
- **Remaining matched-build boundary:** frontend41 is completing the agreed observation header/producer records in
  `metal_program.m`. Its current draft and the Python binding must match before native imports/invocation. The
  current joint `compile_program`/`solve_program` is still being implemented and has no execution claim here.

The public library names remain `libmetal_recording.dylib`, `.build/libmetal_ir.dylib` and `libmesh_metal.dylib`.
Makefile now builds the recorder library from live `metal_capture.m` and `metal_program.m`; deliberately malformed
original `metal_recording.m` stays outside compilation. This checkpoint records ownership and connected source ABI,
not numerical, archive-roundtrip, transport or performance evidence.
