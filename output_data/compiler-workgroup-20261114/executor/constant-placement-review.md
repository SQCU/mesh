# Constant initialization is not backing storage

The existing `pl.Inference` → `parallel.chosen` client reached native replay with Value68 unplaced. Actual retained
records identified a 16-byte F32 CONSTANT: data present, buffer absent, storage68, ordinary read at command0 slot45.
Its native Storage had bytes16/alignment4, but `native.storages` set run0 solely because initial data existed.

The correction tests actual `buffer` ownership instead. The sole planner now allocates the constant's invariant
storage; existing `metal_capture.m` `MetalInitialContents` initializes its planned copy once. There is no second
allocation path or per-invocation copy. SAME047 captured the actual failed client and implemented the predicate;
supervisors checked the producer and consumer independently.

The same client retry passed compile, artifact load with no relation solve, and native invocation. One actual
observed unit (call0/site20/dispatch0) reported 75.541 microseconds, with numeric scope [0,0,0,126]. This isolated
duration is not a performance comparison. The chooser then stopped at unresolved Work terms, products of share
factors, and a cycle of stage classes; it did not produce an accepted cost model or partition.

Input/environment remain the exact prior command in metal-microbench's
`output_data/compiler-workgroup-20261114/frontends/litert/chosen-local-review.md`. Raw bounded records here are
`chosen-placement-diagnostic.json`, `chosen-placed.stdout` and `chosen-placed.stderr`. The run used the matched
v35 libraries; completion-scope v36 source was not loaded. No additional numerical/cost run was made.
