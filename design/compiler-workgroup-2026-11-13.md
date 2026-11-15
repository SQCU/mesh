# Compiler integration workgroup, 2026-11-13

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

## Assignment

The operator, 2026-11-13:

> okay. so if there are no obvious landmines here which suggest we cannot govern or delegate tightly scoped, reviewed, and feedback-corrected work assignments to claudes (to, you know, write all of the implementation coverage needed here...?), it's time to do that. chop up the change request documents into assignmetns that induce well behaved and completed work units which in their conclusion give us modules which are *more similar to  ~/schedule-emit* than to *mesh* or to *metal-microbench*. this is a measurement which can be delegated to codex (agents or subagents) managing claude subagents; claude subagents have unfortunately demonstrated an untrustworthy character when delegated judgements of value and of structural similarity in this working scope.

This is an implementation assignment. Claude Opus 5.5 workers write the replacements; Codex supervisors inspect
their actual data flow, give corrective assignments, review execution evidence, and account for remaining work.
The current assignment continues the operator turns in [the-compiler.md](the-compiler.md) and the
[change-request naming standard](change-requests.md). It does not add requirements to their scope.

## Correction to the first assignments

The operator, after root reported that an existing lowerer consumed CPU graphs before the proposed replacement:

> "because the existing lowerer consumed CPU graphs first." why is there an existing lowerer? how does this comport with the requirement that the structure of the written code comport more with the scheduler-lowerer demo repository than with mesh or metal microbench?

Then:

> clamp down on the lying claudes: give them narrower work assignments and no freedom to 'innovate' by reusing garbage detritus code

The first briefs made repairing inherited bodies a work unit. Codex supplied that scope; it does not establish
worker intent. That scope is withdrawn. The table below remains an inventory of responsibilities and initial
editing locations, not authorization to preserve their implementation structure.

Replacement assignments specify one concrete producer, the exact operand/result needed by its consumer, the
reference source that supplies the implementation, and the inherited body displaced. Codex settles that contract
before assigning it. Claude implements that bounded connection. It may identify a missing operand or source
fact for correction; it may not select an inherited body as an alternative design or add a compatibility path.

Read legacy source to identify callers, existing feature obligations and removal targets. It is not a source
template. In particular, renovating `lower_metal.Native`, wrapping the provider's inherited traversal, or
reordering a list of lowerers that claim residual graphs does not satisfy a replacement assignment. Target
selection belongs to the compiler's explicit operands. A target receives its assigned body and finishes it.
Runtime and signature consumers take the completed result; they do not reconstruct the caller's program.

The individual corrective assignments replace conflicting first-round directions. Source checkpoints preserve
the work for review; a committed intermediate patch is not structural acceptance. Local compiler/API checks
are evidence, not a permission gate. Coordinate genuinely shared build outputs and hardware sessions without
inventing a prohibition on ordinary local checks.

## Sources and their roles

| Source | Role in this assignment |
|---|---|
| `~/schedule-emit/schedule_emit/driver.py`, `export.py`, `placement.py`, `devices.py`, `schedule.py`, `ir.py`, `archive.py`, `runtime.py` | Runnable example of stage responsibility and explicit producer-to-consumer composition. Read the code. Exact modules, signatures, sample graph identities and its Python runtime are not prescribed implementations of mesh's native target. |
| [Compilation composition](compilation-composition-change-request.md), items 1–7, the corrections, and "Complete replacement, all in one commit workgroup" | The required handoffs, complete lowering, terminal source emission, and displacement of the old implementations. Use its reusable form for individual assignments. |
| [AOT compilation](aot-compilation-change-request.md), A1–A10; [unequal shards](unequal-shards-change-request.md) | Captured values, placement and cost operands, collective expansion, memory allocation, schedules, and reuse of solved program relations. |
| [Torch source lowering](torch-source-lowering-change-request.md) and [Torch capture](torch-capture-change-request.md) | Required frontend and numerical coverage, including collectives in compiled execution. |
| [Plan to bridge](plan-to-bridge-change-request.md), [disjoint pages](disjoint-pages-change-request.md), [page queue](page-queue-change-request.md) | Consumers of solved storage, transfer and execution relations. |
| [Integration finishing](integration-finishing-change-request.md), [14b9a757 problems](problems-of-14b9a757-change-request.md), [synchronization signature](synchronization-signature-change-request.md) | Production connection, displaced duplicate work, and relating the compiled program to execution. |
| [Implementation order](implementation-order-2026-11-09.md) | Dependency relationships between results. It does not freeze consumers or require serial work. |
| [Boundary review](compiler-boundaries-review-2026-11-13.md) | Dated source observations to investigate, not proof of present behavior or a new specification. |

Use the papers and reference implementations cited in the applicable clauses for algorithms, data structures,
and their names. In particular, the composition request names MLIR structured operations and bufferization,
Tint IR before its MSL writer, and the actual LiteRT/ML Drift producers. The compiler request names TensorFlow
Send/Receive and GSPMD. Their exact cited roles remain those in the requests; an assignment is not a license to
expand their semantics. Ordinary grammatical changes to source words are allowed; naming defects are revised
while proceeding, as the operator instructed. No formal type-system project is assigned.

## Concurrent units

The operator broadened the active work on 2026-11-13:

> try reviewing the entire codebase and identifying all mechanisms by which a compilation IR/optimized solution can be 'discarded', then removing all of those 'discard mechanisms' all at once by refactoring (of the entire data flow)

This continues the same composition requirements across mesh and metal-microbench. The workgroup is reviewing
the complete compilation data flow, including callable alternative paths, serialization, loading and actual
execution consumers. Its dated inventory and implementation status are in
[the result-retention review](../output_data/compiler-workgroup-20261113/discard-mechanisms.md).
Producer and consumer edits belong to the same replacement even when recorded in several working commits.
The [14 November clarification](compilation-composition-change-request.md#clarification-2026-11-14-one-compilation-over-retained-results)
governs this inventory: specializing a consumer does not make its input result or references disposable. Preserve
compact results as accessible values while removing repeated interpretation from execution. Reuse the cited graph,
unrolling and allocation algorithms already used for transport across the other compiler scopes. Extracting those
algorithms and connecting their results does not authorize copying the inherited executor or adding a second solve.

These are initial editing responsibilities for coordination, not walls across interfaces. When a complete
connection needs another owner's file, the supervisors agree the change immediately and one worker writes it.
An absent operand requires completing its producer and consumer together, not parking either side or supplying
a second interpretation. Source is shared on main; commits record progress. Stage only named files.

| Unit | Codex supervisor | Initial files and complete connection |
|---|---|---|
| Torch export | frontends | `compiler/mesh_compiler/export.py`, `devices.py`, `shares.py`: captured values and annotations into placement/lowering, retaining the producer's meaning. Coordinate actual consumers with compiler. |
| LiteRT export | frontends | metal-microbench `provider/` and its source build: model operations, bodies, values and effects into the same whole-program compilation responsibility. Complete the producer interface; generated source and per-shader callbacks cannot substitute. Coordinate production consumption with integration. |
| Placement and collectives | compiler | `compiler/mesh_compiler/placement.py`: explicit placement and sharding into one graph containing computation and transfers, with identities and relations consumed by scheduling/lowering. |
| Scheduling and storage | compiler | `compiler/mesh_compiler/schedule.py`: graph, costs, machine model and lifetimes as operands into schedule/storage, consumed without rebuilding the solved problem. Configuration acquisition belongs at the caller. |
| Numerical lowering | compiler | `compiler/mesh_compiler/lower_metal.py`: explicit computation/access operands into fusion, allocation and terminal emission; remove emitter invocation used merely to discover accesses. Coordinate all target calls with executor. |
| Native execution | executor | `compiler/mesh_compiler/runtime.py`, `bridge.py`, `lower_coreml.py`, `rdma/mesh_metal4.py`, `mesh-metal4.m`: completed program into native CPU/MPS/ANE/transfer/copy execution; remove Python callbacks from the compiled program. |
| Archive and loading | executor | `compiler/mesh_compiler/archive.py`, `driver.py`, `lower_inductor.py`, `__init__.py`: completed compilation results into storage/loading and invocation without scheduling again or quietly executing residual FX modules. |
| Production integration | root | Shared `compiler/mesh_compiler/ir.py`, `signature.py`, existing clients; metal-microbench `tools/prepare_e2b.py`, `tools/e2b_replay.m` and required compiler consumers: connect the above values through sanctioned entry points, displacing semantic rediscovery. Agree shared representation edits with the producers and consumers. |

Each supervisor gives its workers narrower concrete assignments using the composition request's reusable form:
applicable clauses, cited source and role, actual producer, representation, consumer, preserved property,
displaced body, files, and existing client that exercises the result. An initial unit ends with an implemented
connection and an honest account of remaining request coverage; adding unused fields or a helper without its
consumer does not finish a unit. The whole replacement's coverage and performance requirements remain in force.
Partial commits are progress, not assertions that the whole replacement is finished.

The existing implementation bodies are the replacement inventory. Do not copy them into renamed wrappers as
the design. Read the reference's composition and implement the cited algorithms in that flow. Do not recover
semantic operands from emitted text. Keep graph algorithms in `mesh-graph`/`mesh_graph`, not in each consumer.
Record concrete remaining work rather than introducing fallback execution or a smaller supported domain.

## Supervision and evidence

### Local compositional review

The operator corrected the comparison boundary during the check-in:

> rather than this lets ask whether the modules which were worked on have been migrated locally wrt their inputs and outputs to match the sort of structure the work requests impose, and that is demonstrated by schedule-emit

Then:

> if every single function, header, or module worked on by this swathe of subagents locally matches the right data flow, and we have evntually refactored every function/module which is required for the overall codebase's role, we know by construction we have actually finished that refactoring scope without bothering to check the entire thing 'end to end' (this is a proof strategy used in real formal methods; what is it?)

The review strategy is compositional verification: establish the changed unit's input assumptions, assigned
transformation and output guarantees, then establish that the connected consumer's assumptions are met.
[Hoare, 1969, section 3.3, Rule of Composition](https://www.cs.cmu.edu/~crary/819-f09/Hoare69.pdf) supplies the
sequential composition principle; [Abadi and Lamport, Composing Specifications, 1993](https://lamport.azurewebsites.net/pubs/abadi-composing.pdf)
supplies the component-specification perspective. These are cited roles for the review method, not requirements
to add formal type machinery, runtime assertions, or a proof assistant.

Assess the functions, headers and modules actually changed. Untouched backends or runtime responsibilities
remain uncompleted units; their existence does not invalidate a completed local migration. A missing operand
at the changed boundary is a concrete local obligation. An unchanged numerical implementation may remain the
right producer; its size or age does not establish a boundary defect.

When all required units and their connecting contracts satisfy the specified data flow, the structural refactoring
claim follows by composition. An end-to-end execution is not an additional proof obligation for that structural
claim. Numerical and performance measurements answer their own stated questions. Do not substitute a whole-repo
similarity verdict for this local accounting, or infer global completion before the required units are covered.

The supervising Codex agent checks a changed connection against the runnable reference by answering:

1. What value is produced, and which actual consumer uses each needed field? Trace identities, access/effect,
   storage, and dependency facts through transformations to their last consumer.
2. Where is each decision made? Which previous re-analysis, implicit conversion, side-effect discovery, callback
   or duplicate implementation has disappeared from the production calling path?
3. Which parameters invalidate which results? Is changing placement/routing forcing unrelated numerical work
   to be reconstructed? Explain from the implemented dependencies, not from a claim in a document.
4. What remains at execution after abstract operations are lowered? Account for dispatches, copies, waits,
   synchronization, and native calls from the compiled result and observed execution.

These are review questions derived from the assigned handoffs, not a new parser, score threshold, or test suite.
Codex makes the structural comparison; a worker's self-evaluation is not its own acceptance decision. Source
inspection can establish a source finding. Numerical coverage and performance claims require the sanctioned
end-to-end calling path and stated inputs. Use existing clients and `metal-microbench/docs/measurement.md`;
do not add mock suites, graph-fingerprint gates, benchmark grids, or a parallel evaluator. A historical note or
passing import alone does not establish present execution behavior.

Record worker session IDs, model identifiers actually reported by the harness, concrete steering, commits,
client commands/results and unresolved work in this workgroup's output records. Preserve raw logs locally;
commit compact evidence and reviewed source. Treat hook findings as feedback to inspect, not new authority.

## Harness and coordination

Use the operator's `claude-personal` profile. Its shell alias calls
`/Users/mdot/.local/bin/claude-profile personal`; that executable is available without interactive shell setup.
Select Opus 5.5 explicitly and inspect the session's reported model. Do not silently use the profile's default
model or another family. Use persistent session IDs and resume them for corrections. Prompts and steering to
Claude are file directions: `Read /absolute/path/to/assignment.md and act on it.`

The three Codex supervisors launch three (compiler), two (executor), and two (frontends) Claude workers.
Root supervises the eighth integration worker. Start implementation after writing concrete assignments;
keep supervisors available for corrective feedback and cross-owner changes. Workers report changed paths,
connections completed and remaining, and evidence; supervisors review and commit named paths regularly.
Do not spawn additional uncontrolled worker trees or assign Claude final judgments of structural adequacy.

Read each repository's applicable instructions. Work in the existing repositories. Never reset, clean, stash,
copy source backups, transfer uncommitted source, or touch unrelated working changes. Do not modify global
harness configuration. Never kill processes by name/pattern, or terminate user terminals; supervise only exact
owned worker sessions. Existing user services and hardware sessions are not this workgroup's disposable jobs.

Environment observation from the preceding review: `TORCH_DEVICE_BACKEND_AUTOLOAD=0` avoids an obsolete
installed Torch plugin while invoking the existing compiler client. This is per-process environment setup,
not a compiler restriction. Runtime Metal compilation uses the API; absence of the `metal` command-line tool
does not establish that it cannot compile. Check actual build and client paths before declaring a blocker.

## Immediate graph algorithm consolidation and result retention

The operator, continuing this workgroup:

> is 'dropping' information from a serializable data tructure evidence of inlining of a graph algorithm where the data structure is a graph? consider eliminating all inlining of all graph algorithms by placing all graph algorithms in a single header module related to graphs to handle this. (this has already begun in this codebase)

> we should also consider that, effectively, there are no semantically valid reasons to 'drop data' from computation results which are already present by wit of being solved, and can be accessed zero-copy by view from existing paged memory, and that consolidating inlined code (and removing redundant inlined code) is in effect a blanket prohibition on the behavior involved in subagent claudes attempting to 'remove' data from interfaces

> " Those callers are centralized only partway." then this must happen immediately

This continues composition requirements 3 and 6; it does not introduce a new graph framework. The immediate
assignment is to consolidate the remaining graph algorithms into the existing `rdma/mesh-graph.h` module and its
implementation. Python and native callers use that owner's results. The Python graph module currently contains
algorithm bodies as well as bindings; replacing only their call sites is incomplete. Directed/undirected semantics,
edge causes, node identities and transformation correspondence remain part of the producer/consumer review.

Retain solved results and their ownership through actual consumers. Bindings expose views with that ownership;
they do not reduce a result to selected fields, free it, then reconstruct it for another consumer. Calls that
analyze the same graph share its retained representation and correspondence. Domain-specific operand construction
and numerical emission remain with their producers; graph traversal, ordering, closure and contraction do not
acquire additional implementations there. This changes the interfaces and their callers together, including
deleting displaced algorithm bodies.

The current source-relation omission is noncompliance with an existing requirement, not a newly discovered
requirement: composition lines 312–314 have required the value/effect/transfer relation of each edge since
`9df08de71` (10 November). Root's earlier wording and review did not make that distinction adequately. Specific
worker drafts replacing relations with operation-number comparisons or task-to-task all-to-all expansion were
rejected. These observations establish defects and review corrections, not intent.
