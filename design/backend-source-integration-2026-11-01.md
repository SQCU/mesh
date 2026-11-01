# Backend source integration — 2026-11-01

The compiler chooses implementations while it still has source equations and page relations. Execution receives prepared commands. It does not choose a backend, trace a function, load a model, or repair an unsupported choice.

## One lowering order

1. `Function.capture` records the supplied source and any backward work already requested by that source.
2. `compilation.assign_regions` applies explicit source-region assignments. `algebra` applies the configured product equation. These transforms keep original source/result identities.
3. Ownership projects the retained scalar and product relations. Placement composes their selected page views with each API's layout and storage requirements.
4. The compiler emits unbound scalar, product, or provider commands. The common dependency/effect pass consumes these declarations.
5. `Program.prepare` binds the resulting Metal, CPU, and ANE commands to the common recorder.

A backend choice is a preparation operand, like a tensor ownership selection. A changing mask or index is still an ordinary input page. It is not a request to choose an implementation during execution.

## Product and scalar assignments

`implementation_assignment.product` uses the existing `domains` / `otherwise` schema. Rules may match scalar types or explicit shape parameters: `rows`, `inner`, `columns`, `batches`, and `rank`. Parameter conditions are `equal`, `min`, `max`, or `multiple_of`. Overlapping rules which choose different implementations are configuration errors.

`metal_product`, `accelerate_product`, and `scalar_contraction` describe three ways to lower the same source product. Accelerate's declared arithmetic ABI is float32. Required source casts and result casts remain ordinary graph equations; they are not hidden inside its runtime entry.

Both bulk product backends consume the shared `product_pages` mathematical declaration and `mesh_matrix.selected_products` work relations. CPU selection does not turn a product into an opaque whole-output host boundary. A partial output request narrows its input, weight, reduction, and output domains before placement. Metal consumes indexed views directly. Each implementation declares its admitted batch rank; CPU batches split into ordinary matrix invocations before layout constraints, so a shared broadcast matrix need not be copied merely to give the batch a positive stride. Accelerate declares positive-strided matrix operands, so the common layout solve prepares any required layout conversion. A dynamic selected matrix which its API cannot directly address may therefore need a prepared materialization. This is an API constraint, not a runtime backend substitution.

Scalar commands retain the existing per-domain implementation selection. Their requirements now also expose iteration and reduction ranks and extents. A selected source-region scalar implementation propagates through normalization to those retained bodies.

## Source-region assignments

`implementation_assignment.regions` is an explicit list. A rule contains `implementation`, optional `nodes` (captured source node names), optional `when`, and a provider `configuration` when selecting a graph provider. Existing collective boundaries, source cuts, and selected region boundaries delimit these regions. Named source nodes select a configuration domain; object identity still carries values between compiler stages.

Region requirements record input/result types, input/result kinds, source operations, and shape parameters such as `input0.dim1`, `output0.elements`, and `operations`. The compiled decision retains its requirements, implementation, source records, and assignment. `Function.implementation_regions` exposes these records after declaration.

The repository's initial `regions` list is empty. It does not claim empirical superiority for a backend. A rule may also declare `alternatives`: records containing `implementation` and, for a provider, its `configuration`. Its `implementation` remains the initial choice. The same schema applies to scalar and product policies. The initial scalar policy offers Metal/CPU for their shared type domain; the product policy offers Metal, Accelerate, and the scalar equation for their shared type domain.

`Function.choices` records the executed source alternatives. Backend coordinates retain source statements and occurrences, member, typed requirements, complete alternatives, and one-unit invocation granularity. The compiler consumes `source_preferences` during source normalization and work-domain selection. A captured page-product declaration also defers this choice until source normalization, rather than fixing a backend during raw capture. An explicitly selected backend remains an explicit operand.

Fold coordinates retain the source scope and call occurrence, combine/identity equations, value schema, declared association laws, contributors, and complete legal parent/segment columns. Each segment's weights apportion its independent values. The source fold consumes these weights before its actual derivative is captured; active tree extents are results of that realization, not the identity of the choice.

`configuration.bind` and the engines example collect the actual records. `assignment.domains` appends them to the existing work/routing/layout/compiler vector, and `assignment.assign` fits the same whole-function cost. New source/backend choices can change the lowered graph's conditional axes. Each such axis is centered only on observations where it exists; it is not treated as a zero share in other graphs. The selecting source coordinate remains comparable across those graphs within the same source revision. Otherwise a separate baseline for every backend would erase its measured cost.

The next preparation consumes the resulting vector. There is no trial selector, accepted/rejected candidate mechanism, runtime backend choice, or new execution controller. Whole-function latency and useful rate share one interval and one fixed logical useful-work count. Minimizing that latency maximizes that rate for the declared workload. Late records retain their executed coordinates and importance; observation does not wait for another rank or episode.

## Generic ANE source

The `ane_graph` implementation connects assigned FX source regions to `export_coreml_graph`, which previously had no generic frontend caller. Export functionalizes the already-captured graph into ATen, then exposes state effects and aliases as page relations. Core ML conversion produces the provider contract before placement. That contract supplies actual feature types, layouts, footprints, alignment, and caller-owned output requirements to the shared layout solver.

The region's `configuration` is the existing Core ML/ANE provider configuration. It states backend, units/devices, precision, deployment, function, state input/output pairs, and ANE ABI capabilities. It may supply explicit symbol maps or `ane.symbol_names` templates for generated feature names. These are preparation operands. There is no model-name lookup or tensor-type guess in the runtime.

Provider aliases keep their original user-result positions. A forwarded input need not become an ANE output or a copy. Captured state updates retain their declared input/output pairs. Compiled artifacts have a preparation owner retained with the prepared execution; this adds no filesystem access during numerical execution.

An ANE procedure is currently an indivisible provider invocation. Its region must already describe the intended share. This integration does not make arbitrary internal Core ML operations independently partitionable, nor claim that every ATen operator is supported by ANE. Type eligibility is declared in the implementation catalog; the converter and compiled provider contract establish operator and storage compatibility. There is no hidden CPU/Metal substitute when a selected provider cannot compile the region. The older explicit tiled linear/FFN engine adapter remains an algorithm-specific source producer; generic graph eligibility does not depend on it.

## Shared CPU provider

`mesh-cpu-source.h/.m` now own the existing CPU source compiler, instruction binding, resource identity registry, and lifetime machinery. Both Python numerical declarations and native collective arithmetic consume this one implementation through `libnccl-mesh`. Core ML consumes its same source compiler and host resource identities; it is no longer a link dependency of ordinary CPU work.

The configured `collective_cpu` implementation supplies native collective source compilation. The compiler command is explicit and includes the common source-header directory. Generated numerical entries and page effects are compiled together before binding.

## Checks and limits

Source checks covered Python/JSON syntax; common CPU compilation and Core ML Objective-C syntax; fake-only capture of batched and dynamically indexed products and convolution equations; retained scalar assignment; sparse product-domain projection; and Core ML conversion of a masked graph with a forwarded input alias. The real graph-provider source capture also preserved repeated forwarded aliases without a provider invocation. The sparse product request selected two `(2,7) @ (7,2) → (2,2)` CPU invocations from a `(2,5,7) @ (2,7,3)` source. The same Metal request retained its batched indexed reader, including index-page dependencies.

Metadata-only source normalization also consumed the declared backend preference for both an ATen product and a page-product declaration. The CPU choice retained `product_pages`; the scalar choice emitted the ordinary multiplication/reduction equation. No numerical payload or fitted synthetic timing was run to validate these transforms.

The engines demo now composes ordinary Functions with one explicit relation context and output pages. The redundant Prepared executor wrapper and Operation.prepare interface are removed.

These checks did not submit numerical work, bind a live page region, run an ANE inference, or measure latency/throughput. The linked whole-program preparation and hardware performance remain unmeasured in this pass. Core ML tools also report that the installed Torch version is newer than their tested version range.
