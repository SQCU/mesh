# Memorandum: unsupported restrictions and claims in the compiler documents

Style guide: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights),
for clear language, descriptive links, and code formatting. Operator quotations retain their spelling.

## Commission and scope

The operator, 2026-11-13:

> "I removed the unsupported constraints and aggregations, corrected the reporting," cool thanks. lets do a similar review over the memoranda and change request documents for similar suspicious defects i suppose

The preceding instruction on excess constraints was:

> if this is an attempt by claudes to add overconstriants, it has to be removed without replacement, no matter what 'breaks' from the absence of the excess constraint.

Codex reviewed the request prose and memorandum conclusions starting at mesh `cc9e7a54`, their relevant introduction
commits, the operator turns they quote, and selected primary sources. The scope included the eleven indexed change
requests, the November 9–10 memoranda, `data-flow.md`, `the-compiler.md`, the change-request format, and the two
November 13 review memoranda. The long embedded transcripts and agent reports were used as attributed records;
not every report or external quotation was independently verified. This was a documentary review, with no compiler
or runtime execution. The existing change to `compiler/mesh_compiler/placement.py` was outside this work.

## Restrictions removed

| Passage and provenance | Problem | Revision |
|---|---|---|
| [The compiler](the-compiler.md), added by `c50fc92f`: one item assigned at a time, alone, with its text the whole assignment. | The commit quotes a request for literal assignments. It does not supply authority for serial work or excluding other applicable requirements. | Removed the restriction. |
| [Implementation order](implementation-order-2026-11-09.md), `cd098202`: closure by deleting named functions; several consumers frozen until item 7 closed. | The quoted commission requests dependency order and complete replacement. Name disappearance does not establish replacement coverage, and data dependencies do not prescribe a freeze on consumer work. | Removed the closure rule and freeze; retained the producer/consumer dependencies. |
| [The compiler](the-compiler.md), `07676c44`: the same modules, signatures, arguments, results, and two types as the example. | The quoted assignment requires the reference's data flow and corresponding responsibilities. The literal interface restriction would also constrain the added sharding and transport operands. | Removed the literal cloning requirement and quoted the actual data-flow assignment. Removed unverified `Scheduled.devices` and `Scheduled.engines` implementation claims. |
| [Requests before the example](requests-before-the-example-2026-11-10.md): omit requirements with no stage in the example's existing mechanism. [Torch capture](torch-capture-change-request.md), `57d9f06e`: four items apply only if recording precedes the static schedule. | An obsolete mechanism was being used to discard the property it had been assigned to implement. The quoted instruction asks to restate all outstanding requirements. | Removed the exclusions. Changed the claimed completed restatement into the requested assignment. |
| [AOT A7](aot-compilation-change-request.md), `7c1e068f`, and [plan-to-bridge R8](plan-to-bridge-change-request.md), `940cb181`: one program only; remove any rule for several programs. | The cited operator turn rejects ad hoc rules and dynamic allocation. It does not impose a single-program limit. A7 itself quotes Pathways' ordering across programs. | Removed the limit and kept the cited sequencer and compile-time allocation requirements. |
| [Change-request format](change-requests.md): repository code and Claude-written documents cannot be sources. | This blanket exclusion conflicts with the format's own composition of requests in stated roles and the commissioned executable counterexample. | Removed the exclusion. Existing requirements for sourced methods and verbatim turns remain. |

The first group of corrections was committed as `71de11eb`. The original passages remain in Git history.

## Algorithm and citation corrections

In [integration-finishing](integration-finishing-change-request.md), items 2 and 3, the original `90b88505` text
combined completion times with dependency levels as scheduling keys and imposed task lengths as weights. Both
passages acknowledged that their extra rule had no cited source. Item 2 now uses R5's completion time for every
instruction and names R5's compilation scheduler as the producer. Item 3 consumes the weighting operand already
required by A3. The uncited substitutions were removed.

In [torch-source-lowering](torch-source-lowering-change-request.md), item 6, `4b9f2b90` turned JAX's rewrite of `cond`
under `vmap` into an unconditional requirement that every `cond` and `switch` evaluate alternatives and select.
That commit explicitly says it came from reviews of `ddfc524d` and `a1f45513`. The
[JAX `cond` documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html), read on 2026-11-13,
distinguishes conditional branch execution from conversion to selection under batching. The universal rule was
removed; the cited rule retains its batching condition.

The same item said no source supplied a compile-time bound for QDWH's Halley continuation. The linked
[JAX source](https://github.com/jax-ml/jax/blob/main/jax/_src/tpu/linalg/qdwh.py), read on 2026-11-13, makes
`max_iterations` static. The continuation counter starts at the number of coefficient steps and increments once per
iteration, with `k < max_iterations` in its condition. The revision distinguishes this bound from the data-dependent
early termination count and from guaranteed convergence. The source is a moving link; these observations belong
to this reading.

Items 5 and 6 also generalized the static-program restriction into bans on iteration inside any mesh-written
lowering, while item 5 permitted iteration inside a CPU supplier function. The operator's later item 6 turn asks
for the full processor scope and expressly counters earlier limitations. Those universal bans and the blanket
off-CPU refusal were removed. The static exported-program scope, plain errors outside it, and the requested NaN
and runtime-error behavior for excessive matrix-exponential squarings remain. This edit does not claim that GPU or
ANE numerical coverage has been implemented, nor that the remaining QDWH-eig lowering questions are answered.

The [page-queue request](page-queue-change-request.md) said the replacement was "finished until" it covered the old
calls. The missing "not" was restored, consistent with the same paragraph's coverage-before-removal order.

## Claims that exceeded the recorded evidence

The [subproblem inventory](already-solved-subproblems-2026-11-09.md) defined a problem as solved when any commit
contained a function or table for it. Its own method said it searched names without reading or running bodies.
The revision removes that definition and the resulting completion totals. The tables remain dated candidate
locations. Unchecked hardware claims and supplier-only restrictions were also removed; the quoted operator turn
explicitly permits either reference source or source written here.

The [interface review](interface-boundaries-2026-11-09.md) treated a cited line as fact and classified a module that
performed several stages as a responsibility inversion. Its preserved reports include calls composing stages and
device compilation operands classified as runtime products. The revision identifies these as limitations of that
review prompt. A source location does not establish the interpretation, and calling stage functions does not by
itself duplicate their algorithms. The historical classifications remain attributed to their readers.

The [function-body reports](subproblem-function-bodies-2026-11-09.md) and
[ragged-computation inventory](ragged-computation-sources-2026-11-09.md) now state the limits of their source readings.
The latter no longer calls a search summary confirmation from the source itself. These edits do not certify the
remaining reports, counts, missing-caller claims, or quotations.

The [synchronization request](synchronization-signature-change-request.md), item 5, inferred exact agreement with the
compiler's signature from playback reaching completion. That inference was removed. The explicit count comparison,
signature comparison, and kernel-synchronization requirements remain; completion alone does not supply their
observations. [The index](../INDEX.md) also no longer presents its linked prose as an account of how the system works
now or names the obsolete `.git/review-commentary/` output path.

## Verification and limits

The edits were checked against the original passages, introduction commits, quoted turns, and the primary sources
identified above. A comparison of blockquoted lines found no removed operator quotations. `git diff --check` was
clean. These are checks of the documentary edits, not evidence of compiler behavior or completion.

The November 13 compiler-boundary and hook memoranda already identify their inspected revisions and limits; they
were not promoted to requirements. Some requests still describe old concrete interfaces, and some source mappings
remain incomplete. This review removes the identified excess restrictions and false deductions; it does not
establish that every remaining requirement is sound or that every implementation obligation is closed.
