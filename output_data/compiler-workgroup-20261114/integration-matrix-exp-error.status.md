# Matrix exponential error/result handoff: status

Integration worker, session `59368744-da56-491a-871a-1be809c0cc93`, `claude-opus-5-5`. **Correction 2: contract
published, Python side implemented, HOLD for executor/native pairing** (see the last section, which needs one
native call and its `metal.py` binding).
Correction 1 needed no native or API edit. That change was Python only, in `compiler/mesh_compiler/operations.py`
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

## Correction 2 (integration-matrix-exp-error-2.md): the check's dependency operand

### What is missing (source, inspected)

The actual caller is `aten.linalg_matrix_exp.out(a, out=out)`. Its `out` is a mutation target, so `lower_metal`
supplies the assertion node a destination, and `_instruction`'s functional-assertion branch emits
`_copy(out_view, dep_token)` and then `_asserted`'s ops.
- **Gap 1:** `_asserted` builds `_Check(info, 4, msg)`, which drops `dep_token`.
- **Gap 2:** metal_ir's check operation (`metal_ir.mm` `metal_ir_check`) has `op.inputs = {value}` (the status alone).
  So the native CHECK dispatch's effects are a read of `info` and nothing else.
- **Result:** within the assertion's source vertex, nothing orders the GPU copy into `out` before the host CHECK. The
  GIVEN source cause `where_4` → assertion orders both after `where_4`, but neither before the other. That's why the
  error reached the caller with `out` still `[[7, 7], [7, 7]]`, and the NaNs landed afterwards.

### Contract (published before any shared edit)

1. **`operations.py` (mine, done below):**
   - `_Check(info, code, message='', dep_token=None)` gains `.dep_token`, the typed value the check depends on, named
     after torch's `_functional_assert_async.msg` operand `dep_token`.
   - `_asserted` sets it, for `_functional_assert_async` only, to `outs[0]`: the assertion's result as `_instruction`
     gives it. That's the destination written by the copy where `lower_metal` supplies one, else `dep_token` itself.
   - Every other `_Check` keeps `dep_token=None`.
2. **Native, metal-microbench `metal_ir.h`/`metal_ir.mm` (executor/native owner, not edited):**
   - `void metal_ir_check_dep_token(struct metal_ir_program *, uint32_t operation, uint32_t value);`
   - Executor review identified the existing `op.gathers` carrier: retain `value` there as a whole-value
     dependency. `OperandEffects` already consumes such values with `Whole()`, without a map over the status box;
     cloning, archive and realization retain them. The status remains `inputs[0]`.
   - Dependence construction then orders the CHECK dispatch after `value`'s writers, with the same host completion
     wait it already gives the status `value` written on the GPU. So the error follows the output write and its
     completion.
   - The check's test still reads only its status (`inputs[0]`). `metal_ir_check_info`, NativeChecked, the cancel and
     every other check are unchanged.
3. **`metal.py` `_Builder` (executor047, not edited):**
   - Add `('metal_ir_check_dep_token', None, [P, U, U])` to `_compiler`'s table.
   - After `metal_ir_check`, if `g.dep_token is not None`, call
     `self.lib.metal_ir_check_dep_token(self.p, operation, self.value(_stored(g.dep_token)))`.
     `_stored` is needed because a complex result's typed value is its real pairs.
4. **No change to** `lower_metal`, the source relation, native failure semantics or other assertions. There's no
   numerical read, no added synchronization beyond the actual producer's completion, and no new graph algorithm.

### Python side (implemented; `operations.py`, two hunks, not staged)

- **`_Check`** (at 1802): it gains `dep_token=None` (`self.dep_token`), with a citation to the operand of
  `_functional_assert_async.msg`.
- **`_asserted`:** for `target.overloadpacket is A._functional_assert_async`, it sets `dep_token = outs[0]`, the
  assertion's result that `_instruction` hands it.
- **Unchanged:** the producer (`_matrix_exponential`), `_instruction`'s branch, every other `_Check`, and other workers'
  hunks in the file (`_zlahqr`, `seeded`/`Generator`/`_rrelu`/`_stepped`).

### Checks

- `ast.parse` passes, and `git diff --check` is clean.
- **Source consistency:** the actual `operations._instruction` was called on meta tensors for
  `_functional_assert_async.msg(predicate, 'linalg.matrix_exp', token)`. This is Python construction only: no native
  library and no target.
  - **No destination:** the result is `dep_token`; the ops are `_Generic ne`, `_Generic where`, `_Check`; and
    `check.dep_token is` the result.
  - **Destination:** the result is the destination; the ops are the `_Generic` copy, `ne`, `where`, `_Check`; and
    `check.dep_token is` the destination.
  - Plain `_assert_async`'s check has `dep_token None`.

### Still missing (pairing required, then root's observation)

- The native `metal_ir_check_dep_token` (item 2) and `metal._Builder`'s binding and call (item 3).
- Until both land, the `dep_token` is carried to `metal._Builder` but not encoded, so the native ordering is as
  observed.
- Root's public-caller observation is the acceptance: `out` already NaN when the RuntimeError reaches the caller.
- **For the frontend (optional, not needed for correctness):**
  - The copy exists because the destination record lands on the assertion node. A functional assertion's result is
    its `dep_token` (upstream meta). So `ir.views` could carry the destination to `dep_token`'s producer (`where_4`),
    which would then write `out` directly.
  - The check's `dep_token` would then be `out` itself, with no copy.
