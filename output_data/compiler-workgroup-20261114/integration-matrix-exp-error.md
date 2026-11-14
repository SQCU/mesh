# Matrix exponential error/result handoff

SAME593, explicit claude-opus-5-5. This is a finite implementation unit under
design/torch-source-lowering-change-request.md §6 and the operator's active priority 2.
Read that clause and the actual current owners before editing. The operator requires
"a runtime error and some NaNs" when the required squarings exceed the bound.

Root source observation: operations._matrix_exponential currently emits
_assert_async(over.logical_not()) before returning where(over, NaN, result).
metal_lower.mm emits NativeChecked followed by an immediate return on failure;
MetalProgramExecute then cancels. A source-level where expression does not establish
that its output is written before this error. Merely exchanging Python statement
order is insufficient when the operations have no data/effect dependence.

Own only the matrix_exp result/error ordering and its necessary actual typed
producer/result/consumer handoff. Compiler64 owns the real-eigen sections in the same
operations.py; frontend owns export/capture and inter-call RNG. Do not edit those
sections. Before a shared native/API edit, publish the precise needed contract in
your status for root to pair with its owner. Do not change native failure semantics
globally or make numerical errors silently continue.

Implement the required ordering using the existing canonical typed data/effect
representations. Inspect upstream Torch's functional assertion/effect operators and
the existing IR relation consumer if useful; preserve actual operand identity rather
than manufacture an arithmetic read of the result, reparse source, or introduce a
new graph algorithm. No runtime graph evaluator, fallback, blanket serialization,
extra numerical checks, target restriction or parallel backend. Keep the existing
matrix-exp algorithm and source error condition. Existing legitimate assertions
elsewhere retain their behavior.

Use schedule-emit's producer-to-consumer composition, not its exact file names.
Names must use the literal cited source vocabulary, with ordinary inflection.
Original intentionally malformed source files remain untouched in place.

Read the exact resulting source and publish the producer, retained value/effect,
actual consumer and displaced code. Check only syntax/source consistency for the
affected live Python files. Do not run a target, build shared libraries, stage or
commit yet. Root will review and observe the public caller using the existing
matrix-exp command and its actual returned/bound result. No new test/evaluator.

Write a compact status to integration-matrix-exp-error.status.md beside this order,
then hold for review. Keep raw session output with this order. Existing numerical
passing observations do not prove this error path or whole-operator performance.
