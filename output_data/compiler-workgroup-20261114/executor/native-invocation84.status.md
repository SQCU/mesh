# Consumed execution and error completion: status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`, at mesh HEAD `37e472b9`.
**SOURCE-READY, HOLD.** Only the source checks below ran. Nothing was run, built, staged or committed, and the C
sources are unchanged.

> Implement direct actual issued-count handoff after run, retaining allocated/borrowed operands until the issued end
> completes. Raise the original execution error after that completion; preserve distinct pre-entry no-consumption
> behavior and existing wait/error semantics.

## Source producer (read, not changed)

- **The count:** `metal_program.m` `MetalProgramExecute` (`:1963–2023`):
  - it sets `program->executions = t` after each entered execution, a failing one included;
  - on failure it cancels, submits, and signals `ended` at `t` (`:2009–2017`), and keeps the operand buffers until
    `ended` reaches `t` (`MetalExternalRelease`, `:2021`);
  - if `MetalStepOrigin` fails before entry (`:1968–1969`), it returns with `executions` unchanged.
- **The getter:** `MetalProgramIssued` (`metal_recording.h:163`, `metal_program.m:445`) returns `program->executions`.
  - `metal.py` had no binding for it.
  - `nm -gU` on `metal-microbench/.build/libmetal_recording.dylib`, the library named by `DYLD_INSERT_LIBRARIES`,
    lists `_MetalProgramIssued` (a static symbol read; nothing was imported).
- **The check record:** `metal_program.h` `NativeChecked` (`:256–272`) writes `checkFailure`, `checkedFailure` and the
  error text inside the entry, before it returns `EDOM`.
  - Only `MetalProgramExecute`'s entry (`:1967`) resets `checkedFailure`.
  - `metal._wait` only polls `ended.signaledValue`.

## The mismatch (source)

- **Before:** `Bound.invoked` raised on a nonzero status before it advanced `self.executions` or waited.
- **The retry:** the next call passed the old `first`, one below the C count.
- **The wait:** `ended` already stood at the earlier failed run's `t`, so the old success wait (`ended() > execution`)
  held at once.
- **Not established here:** whether this produces the observed `check 3, batch 0, info -4`. No run was made.

## Consumer change

- **`metal.py`:** one line binds `MetalProgramIssued` (returns `Q`, takes `[P]`) in `_recorder`'s table.
- **`native.py` `Bound.invoked`:** after `run` returns, whatever its status:
  - it reads `issued = MetalProgramIssued(program)`;
  - it makes the error then, with `M._execution_error` and the unchanged message (`execution {execution} of the
    program: …`), which reads the check record before anything else happens;
  - it sets `self.executions = issued`, the next call's first.
  - Where `issued > execution`, it waits for `ended() >= issued` with the unchanged `_wait` message.
    That is before the outer `finally` releases the borrowed buffers, and while `allocated` is still in scope.
  - Then it raises the original error.

| Case | `issued` | Wait | Raised |
|---|---|---|---|
| Success | execution + 1 | `ended ≥ issued`, the same as the old `> execution` | nothing; outputs returned |
| Failure in an entry | execution + 1 (the C count) | until the signalled end | the original error, after the end |
| `MetalStepOrigin` failure | execution (nothing consumed) | none | the original error, at once |
| No end in `MESH_PLATFORM_NS` | — | times out | the existing `_wait` timeout error |

- **Unchanged:** both callers (`__call__` and `observe`, through one `invoked`); the C code; cancellation and drain;
  `MetalProgramEnd`; the event; and the error strings. No counter or guessed count was added.

## Checks

- **Compile and lint:** `py_compile` and `pyflakes` pass on `native.py` and `metal.py`. No line is over 120
  characters, and `git diff --check` is clean. Supervisor review removed the unnecessary inner try/finally;
  AST and diff checks passed again. Final diff is +16/−6 (`metal.py` +1).
- **AST:**
  - `invoked(self, run, given)` makes one `MetalProgramIssued` call, one `_execution_error` and one `_wait`;
  - its only `executions` assignment is `self.executions = issued`;
  - it has 2 callers, each passing 2 arguments;
  - the binding tuple reads `('MetalProgramIssued', Q, [P])`.
- **Not run:** the same-case retry, which the compiler owns.
