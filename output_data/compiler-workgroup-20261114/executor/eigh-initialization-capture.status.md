# Live initialization snapshot: status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`. Mesh HEAD `31636687`, native source `1f605fa9`, with the installed matching libraries recorded in the compiler's `eigh-sites` manifests. **HOLD.** One run, no production edit, no build, no `Program` and no invocation.

## Run

- **What ran:** `compiler/eigh-native-command.md`'s environment, interpreter, model, input and MPS mapping, through
  `compile_module` only (`/tmp/archive-worker/diag/eigh_init_capture.py`).
- **The hook:** a `sys.settrace` `return` event of `native.Target.placed.__code__` with `finalize=True`. The facts were
  collected in that frame and written, the hook removed, then a sentinel raised and caught after ordinary cleanup.
- **Result:** process **exit 0**. The **sentinel stopped it before ProgramMake** (`MetalRecorderProgram` was never
  called): `{"sentinel": true, "program_made": false}`.
- **Files:** facts in [eigh-initialization-capture.json](eigh-initialization-capture.json); logs in
  `eigh-initialization-capture.stdout`/`.stderr`.

## Facts

- **Rows:** 9 rows, each its own instance, `every` 1 and `ordinal` 0 (so the block-0 execution is 0):
  - row 0, the prefix: source F 0, 64 values, 1 part;
  - rows 1–7, the Steps: source F 1, 4524 values each, 5 parts;
  - row 8, the suffix: source F 2, 16 values.

  No composite, and the value and storage counts are equal in every row.
- **Arena:** one shared buffer, 6944 bytes, equal to `Local.bytes`.
- **Initializers: none.** No value in any row meets `MetalInitialContents`' predicate (data, `data_bytes`, no fixed
  buffer, nonzero stand-in section, placed buffer). So nothing is written from host data during preparation.
- **The prefix's agenda (output 2, root 6) and top (output 3, root 8):**
  - both are role OUTPUT, with no `data`, `data_bytes` 0 and no buffer;
  - agenda is placed once (1 copy, shares 1) at arena 128, 40 bytes (int32 5×2), and top at arena 400, 4 bytes;
  - before any numerical entry both are all zero, with no initializer overlapping either;
  - they are therefore runtime outputs of the prefix's single dispatch, and these bytes are not evidence of the
    formula.
- **Inspected interface bindings:** each listed interface root is placed only through its joined joint root. Each listed Function value is
  run 0, kind 1. These inspected roots have no operand occurrences and **no double placement** (no ValueID with both a run value and
  a joined placement).

| Position | Joint root s @ arena offset (bytes) |
|---|---|
| Prefix outputs 0–5 (h, vectors, agenda, top, norm, site) | s1 @64 (64), s2 @3584 (64), s3 @128 (40), s4 @400 (4), s5 @6928 (4), s6 @6936 (4) |
| Step 1 inputs 0–5 | s1, s2, s3, s4, s5, s6, the same roots as the prefix outputs |
| Step 1 outputs 0–3 | s7 @3648, s8 @3712, s9 @3840, s10 @6552 |
| Step 7 inputs 0–5 | s27 @3584, s28 @3776, s29 @3880, s30 @6552, s5, s6 |
| Step 7 outputs 0–3 | s31 @3648, s32 @3712, s33 @3840, s34 @6456 |
| Suffix inputs 0–1 (blocks, vectors) | s31 @3648 and s32 @3712, Step 7's outputs |

- **Arena reuse between distinct joint roots** (each 1 copy):
  - s2 and s27 share @3584; s7 and s31 share @3648; s8 and s32 share @3712; s9 and s33 share @3840; s10 and s30
    share @6552.
  - These are the actual produced offsets. This snapshot does not validate reuse correctness or infer intermediate values after reuse.
- **Step rows' Function-value lifetimes (dispatch positions):**
  - inputs: blocks, agenda, top and norm are [0, 0], vectors [0, 61], site [0, 60];
  - outputs: blocks, vectors and agenda are [0, 61], top [61, 61].

## What this establishes and what it does not

- **Refuted for this run:** the initializer-overlap lead. No prefix constant is initialized from host data on a
  joined EXECUTION root, there is no initializer at all, and there is no double placement.
- **Consistent at the plan level:**
  - the prefix's agenda and top roots are Step 1's agenda and top inputs (s3, s4);
  - the suffix reads Step 7's outputs (s31, s32).
- **Not decided:** the numerical cause. Prefix and Step results are runtime-produced and were not observed, and the
  reuse pairs above are listed, not judged.
