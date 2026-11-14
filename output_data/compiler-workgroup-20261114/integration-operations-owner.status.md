# Operations owner: status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

Integration worker, session `59368744-da56-491a-871a-1be809c0cc93`, `claude-opus-5-5`. **Implemented, HOLD** for
root's staging and commit. The malformed `rdma/mesh_compile.py`, `lower_coreml.py`, `examples/tp.py` and
`metal_recording.m` are not edited, imported or parsed around.

## Dependency direction

`lower_metal` → `operations` and `metal`; `metal` → `operations`; `decompose` → `operations`. `operations` imports
only the standard library, torch and pytree, plus rdma `mesh_code` (the `mesh::fragment` op that `SOURCES` keys on).
It imports nothing from `metal`, `mesh`, ctypes, ObjC or any runtime.

## `compiler/mesh_compiler/operations.py` exports

- **For `metal._Builder` (encoder047):** `_key`, `_stored`, `_Words`, `_Check`, `_Lapack`, `_Sort`, `_rows`, `_bits`,
  `SCALAR`, `KINDS`, `CPU`, `GPU`, and the records `_Generic` and `_Region` with the fields in
  `executor/live-metal-owner.status.md` §1.
- **Codes and tables:** `BOOL … F64`, `KINDS`, `SCALAR`, `CLASS`, `STORED`, `REAL`, `COMPLEXES`, `FLOATING`,
  `INTEGRAL`, `CONTRACTED`; `PARALLEL`/`REDUCTION`, `ASSIGN`/`SUM`/`REGION`, `IDENTITY`/`ACCUMULATE`,
  `REASSOCIATE`/`CONTRACT`; `GPU`/`CPU`/`ANE`; `BINARY`, `COMPARE`, `BOOLEAN`, `COMPLEMENT`/`NEGATION`/`NOT`,
  `BUILTINS`.
- **For `lower_metal` and `driver`:** `Generator`, `seeded`, `_stepped`, `_instruction`, `_local_call`, `_copy`,
  `_held`, `_placed`, `_metadata`, `_same_view`.
- **For `decompose` (compiler64):** `_matrix_exponential`, `_polar` and the other cited numerical formulations.
- **Contents:** the dependency closure of these names in the source definitions: `_instruction` → `_lowering` →
  `SOURCES`, and every structured lowering, numerical formula and record it reaches. They are carried verbatim, with
  their citations.
- **Not carried:** the IR, partition, collectives, `lower`/`_place`, DLPack and ObjC tables, `_called`, `_Builder`,
  and every closure orchestration.

## `lower_metal` imports (direct, no facade)

- `from .operations import ANE, CPU, GPU, _copy, _held, _instruction, _key, _local_call, _metadata, _placed,
  _same_view, _stored`.
- `from .metal import CONSTANT, INPUT, OUTPUT, STATE, TEMPORARY, _Builder, _compiler`.
- The `sys.path` and `import mesh_compile as M` lines are deleted.

## Made

- **New `compiler/mesh_compiler/operations.py` (4,645 lines):** a citation header, the imports above, then 15 verbatim
  runs of source definitions. They come from `rdma/mesh_compile.py` at `039ad8ed`, lines 31-32, 35-40, 78-81, 341-352,
  1028-1037, 1156-1167, 1269-1273, 2050-2057, 2059-2587, 2595-3050, 3071-6557, 6691-6693, 6708-6718, 6744-6792 and
  8067-8070. The header lists them too.
  - The selection is the closure of the required exports. It was read statement by statement, with `symtable` global
    references, from the source, which is never imported or edited. Each run keeps its own citation comments.
  - It also carries the two `torch.library.register_fake` shape functions for `aten::geqrf` and `aten::linalg_lstsq`
    (source 3952-3991). No name references them, but they use `_lstsq_driver`/`_linalg_vector_rhs` and give fake
    shapes to operators that `SOURCES` lowers. Other owners shouldn't register them a second time.
- **`compiler/mesh_compiler/lower_metal.py` (+35 −36):** the `sys.path`/`import mesh_compile as M` shim and the then
  unused `sys`/`pathlib` imports are deleted. It imports directly, as published above. All 38 `M.` references are now
  bare names, and two comments name `operations`/`metal` in place of `mesh_compile`.
- **Not carried, with their only consumers:**
  - `_shard_select` (source 3057-3070), used only by the old `_specialized`.
  - `_called`, `_writes` and `_Paged`, used by the old IR call path.
  - `NONE` (`metal.py` owns it), `DOT`, and the DLPack `CODES`/`TYPES`.
  - Everything in the IR, partition, collective, `lower`/`_place`, native or closure orchestration.
- **Live consumers still on the dead module, which I didn't edit:**
  - frontend `driver.py:19,57,61` (`mesh_compile.Generator`, `seeded`, `_machines`). The replacements are
    `operations.Generator`/`seeded` and `metal._machines`.
  - compiler `bridge.py:58,61` (`mesh_compile._lent`). The replacement is `metal._lent`.

## Checks

- `ast.parse` passes on `operations.py` and `lower_metal.py`. A `symtable` scan finds no undefined global in either
  module, apart from the module-provided `__file__`.
- Imported symbols resolve:
  - `lower_metal`: 12 names from `.operations` and 7 from `.metal`.
  - `metal.py`: 12 names from `.operations`.
  - `decompose.py`: `_matrix_exponential` from `.operations`.
  - Every export the assignment lists, including `_Generic` and `_Region`, is defined at the top level of
    `operations.py`.
- A one-time private import of `operations.py` by file path, without the package `__init__`, evaluates:
  - It builds 246 `SOURCES` lowerings and 13 `KINDS`.
  - `metal`, `mesh`, `mesh_graph` and `mesh_compile` stay unloaded.
  - No native library is involved, and nothing ran beyond the import.
- The malformed originals still fail, and none was edited:
  - `ast.parse` fails on `rdma/mesh_compile.py` (line 7628), `lower_coreml.py` (321) and `examples/tp.py` (121).
  - `metal_recording.m` still holds both malformed wrapper headers. It differs from HEAD only by the executor's hunks.
- `git diff --check lower_metal.py` is clean. `operations.py` has no trailing whitespace. Its 31 lines over 120
  characters are verbatim from the source.
- Not done: any build, any run of `lower_metal`, and an import of `metal.py` (executor-owned; it loads native
  libraries).

## Released to compiler64

`operations.py` lines 1956-2557 are released for spectral, least-squares and decomposition edits. They run from
`ORTHOGONAL` through `_least_squares`'s `return ops`:
- `_lstsq_driver`, the two fake registrations and the LAPACK codes.
- `_singular`, `_eigen_hermitian`, `_eigen`, `_pseudoinverse` and `_orthogonal`.
- `PADE`, `SQUARING`, `_matrix_exponential` (2145) and `_polar` (2190).
- The `_linalg_*` rotation and least-squares helpers, and `_least_squares` (2507).
