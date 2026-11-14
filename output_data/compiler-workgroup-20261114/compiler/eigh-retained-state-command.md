# Diagnostic: retain existing call outputs

Use the same input, Hermitian module, MPS processor/placement, environment and interpreter as eigh-native-command.md. The sole process adds no numerical operations or calls. Before the ordinary call analysis, a process-local wrapper around driver.outline appends these existing Call.outputs to captured.exported.graph_module's outer output tuple:

- prefix call0 outputs2,3,4: agenda, top, norm;
- first Step call1 outputs0,1,2,3: blocks, vectors, agenda, top;
- last Step call7 outputs2,3: agenda, top.

The existing two eigh outputs remain first. Append matching (OutputKind.USER_OUTPUT,None) roles, replace the outer out_spec with the flat eleven-element tuple spec, and recompile that existing outer GraphModule. Then the ordinary compile_module→Program→invocation runs once. The result prints the original accuracy metrics/vectors and these nine retained tensors.

This changes outer result ownership and lifetimes and therefore may change allocation or binding behavior. It is not an unchanged-client accuracy rerun. A disappearance of the original failure would point toward that boundary; it would not prove the original allocation or algorithm correct. No detached intermediate scratch is read, no standalone Step/CPU numerical evaluator is used, and no production source is modified. Logs: eigh-retained-state.stdout/.stderr. Binary/source snapshot: eigh-sites manifests (unchanged since that release).
