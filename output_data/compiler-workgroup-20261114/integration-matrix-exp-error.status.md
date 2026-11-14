# Matrix exponential error/result handoff: status

Integration worker, session `59368744-da56-491a-871a-1be809c0cc93`, `claude-opus-5-5`. **Implemented, HOLD** for root's
review and public-caller observation.
No native or API edit is needed. The change is Python only, in `compiler/mesh_compiler/operations.py`
(the matrix-exp tail, `ASSERTIONS`, `_instruction`). Compiler64's real-eigen sections are not touched.

## Contract

- **Producer:** `_matrix_exponential` writes `result = torch.where(over[..., None, None], nan, result)` first, then
  `A._functional_assert_async.msg(over.logical_not(), 'linalg.matrix_exp', result)`, upstream's functional form of
  `_assert_async.msg` (torch `_export/passes/functionalize_side_effectful_ops_pass.py`). The `dep_token` is the
  actual result: operand identity, with no arithmetic read.
- **Retained value/effect:** `make_fx` traces the operator's output as the functional assertion's result, because
  upstream's meta (`functional_assert_async_meta`) returns `dep_token`. The schedule's relation gets the existing
  GIVEN "value" cause `where` → assertion. The assertion's readers follow it.
- **Consumer:** `_instruction` gives the functional assertion's result its `dep_token` operand's typed value (upstream's
  meta semantics). There is no temporary and no copy, except into an existing destination operand `outs`, which it
  must hold. `_asserted` (now also for `_functional_assert_async`) emits the same check on `self` as for
  `_assert_async`. Native `_Check`/NativeChecked are unchanged.

## Resulting source (`compiler/mesh_compiler/operations.py`, three hunks; nothing staged or committed)

Other workers' concurrent hunks in this file (compiler64's `_zlahqr` section, and the frontend's `seeded`, `Generator`,
`_rrelu` and `_stepped`) are untouched.

- **Producer, `_matrix_exponential` (hunk at 2262):** the algorithm and the source error condition are unchanged:
  `over = count > max_squarings`, then `result = torch.where(over[..., None, None], math.nan, result)`, then
  `return A._functional_assert_async.msg(over.logical_not(), 'linalg.matrix_exp', result)`.
- **Displaced code:** `A._assert_async.msg(over.logical_not(), 'linalg.matrix_exp')`, a side-effectful node with no
  data or effect dependence on the result, and the `where` that followed it as the return.
- **Retained value/effect:** a one-time fake `make_fx` trace of the actual `_matrix_exponential` (3×3, `max_squarings`
  16) gives `where_4 = where(unsqueeze_7, scalar_tensor, matrix_exp_squares)`, then
  `_functional_assert_async(logical_not, 'linalg.matrix_exp', where_4)`, then `output(_functional_assert_async)`.
  - `decompose._expanded` copies those nodes in unchanged.
  - `schedule.dependence` then gives the assertion the existing GIVEN "value" causes from `where_4` and `logical_not`
    (its `all_input_nodes`). The operator's readers take GIVEN causes from the assertion.
  - So the check is ordered after the result's write, and the result's consumers after the check. That is upstream's
    `dep_token` semantics, not blanket serialization.
- **Consumer, `_instruction` (hunk at 7209):** for `_functional_assert_async.msg`, the result is the `dep_token`
  operand's typed value itself: `functional_assert_async_meta` returns `dep_token`.
  - There is no `_made` temporary, so no root is allocated for the token. The relation's typed witnesses
    (`lower_metal.typed`) name the actual result's ValueID.
  - Only where `lower_metal` supplies an existing destination operand (`outs`, the `ir.views` CAT/PAD record) is
    the token's value copied into it, and that copy comes before the check.
- **`ASSERTIONS` (hunk at 1883):** gains `A._functional_assert_async`, so `SOURCES` routes it to the existing
  `_asserted`.
  - That emits the same `ne`/`where` and `_Check(info, 4, 'linalg.matrix_exp')` on `self` as `_assert_async` does,
    ignoring `dep_token`.
- **Unchanged:** `lower_metal`, `metal._Builder`'s `_Check` encoding, native `NativeChecked` and the
  `MetalProgramExecute` cancel, and every other assertion, including the direct
  `_asserted(A._assert_async.msg, ...)` at the pre-existing call site.
- **Native or API contract needed:** none.

## Checks (syntax and source consistency only)

- `ast.parse` of `operations.py` passes. `git diff --check` is clean. No added line is over 120 characters.
- The trace above, and `SOURCES[A._functional_assert_async] is _asserted`: True.
- The upstream facts used were read from the venv's torch:
  - The schema is `aten::_functional_assert_async.msg(Tensor self, str assert_msg, Tensor dep_token) -> Tensor`.
  - `_meta_registrations.py` has `return dep_token`.
  - `functionalize_side_effectful_ops_pass.py` maps `aten._assert_async.msg: aten._functional_assert_async.msg`.
- Not done: any target run, library build or caller observation. Root observes the existing matrix-exp command's
  returned and bound result. Existing numerical observations don't establish this error path.

## Supervisor observation after source review

The first public `compile_module` observation used a real exported module returning
`aten.linalg_matrix_exp.out(a, out=out)`, with `a=[[0,1e6],[0,0]]` and a separate
output tensor initialized to 7. The source squaring count is 17, above the default 16.
Torch export retained `out` as USER_INPUT and the result as both
USER_INPUT_MUTATION(target=out) and USER_OUTPUT; capture did not mutate the argument.

Compilation stopped before native invocation: `lower_metal.construct` could not
resolve the retained destination name `out`. The actual assembly retained ordinary
value-use placeholders but omitted this write-only address operand. Frontends41 owns
the correction using the existing destination records. Raw stdout/stderr are
`integration-matrix-exp-error-1.stdout` and `.stderr`; imported source hashes were
unchanged across import. This is evidence of a missing handoff, not a numerical pass.

The functional assertion source is a working checkpoint. NaNs-before-error remains
unproved until this same caller executes. In particular, where construction supplies
a destination copy, statement order alone does not prove copy completion before the
native check; the actual retained dependency and observation must establish it.
