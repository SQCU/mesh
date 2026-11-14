# Matrix exponential result/error ordering: actual caller observation

14 November 2026. Source checkpoints mesh e51793b1, ae2c9564 and e0acf4fe;
native consumer metal-microbench 3bdfbaf7. These observations cover the explicit
float32 MPS caller below, not every numerical operator, dtype or processor.

The public caller was `compile_module` followed by `Program` for a module whose
forward returns `torch.ops.aten.linalg_matrix_exp.out(a, out=out)`. Its input was
`a=[[0,1e6],[0,0]]`, with a separate output buffer initialized to 7. The cited
scaling formula requires 17 squarings, above the default limit of 16.

The first attempt found a discarded write-only address operand in assembly;
mesh 1e8c9671 retained that actual destination placeholder. The next attempt
reached the runtime error but returned before the output write completed: an
immediate read still saw 7, and a subsequent read saw NaNs. Python statement
ordering and a source-level dependency alone had not connected the lowered copy
to the native check.

The correction retains the functional assertion's actual result/destination as
its `dep_token` in `_Check`, encodes that operand through `metal.py`, and passes
it to the existing whole-value gathered-effect owner. The CHECK kernel binds
that storage so capture and normal dependency lowering retain it. The status
remains its original input and binding; native error/cancellation semantics
were not changed. No numerical dummy load or separate graph algorithm was added.

`make -j4 .build/libmetal_recording.dylib` completed successfully with canonical
Makefile inputs (`integration-check-dependency-build.log`). Attempt 3 imported
operations.py while the numerical worker was changing it, and stopped with a
name error before native construction; its unequal source hashes identify that
transitional observation. Attempt 4 waited for source hold and retained equal
before/after import hashes, plus actual native binary hashes, in its stdout.

Attempt 4 exited 0 with empty stderr:

- The expected RuntimeError named `linalg.matrix_exp`, check 4, info 1.
- The first snapshot of the caller's output after that exception was all NaN.
- The same prepared Program then accepted `[[0,.25],[-.125,.5]]` and returned a
  result with relative error 3.365406924352477e-8 against the double reference.
  Its returned tensor shared the caller's output storage.
- Preparation reported restoration of the native program without solving the
  relation again. This is not a separate complete distributed archive observation.

Raw evidence is `integration-matrix-exp-error-{1,2,3,4}.stdout` and `.stderr`.
Attempt 4 clones the output immediately upon catching the error and computes
both displayed values and NaN classification from that single snapshot; it
does not combine two reads made at different times. The remaining numerical,
capture/state and distributed scopes retain their separate outstanding work.
