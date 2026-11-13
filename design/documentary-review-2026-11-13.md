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
or runtime execution. The existing change to `compiler/mesh_compiler/placement.py` was outside the initial review;
the later section on the interrupted placement patch records its subsequent review and removal.

The operator then requested a contextual review of these findings:

> alright it seems like each of these defects are urgent; lets critically review them i guess and figure out where intention is coming from without falling for the specific prose used in any case.

The second reading used the surrounding human turns, assistant responses, and parent revisions. It corrected the
first review's origin claim about the single-program limit and narrowed its verdicts about copying the reference,
task weights, and the signature postcondition. The context below distinguishes the requested result from the
agent's proposed remedy; it does not infer private motives from a phrase or a commit title.

## Restrictions removed

| Passage and provenance | Problem | Revision |
|---|---|---|
| [The compiler](the-compiler.md), added by `c50fc92f`: one item assigned at a time, alone, with its text the whole assignment. | The immediate problem was copied clause inventories becoming work orders. Literal assignments address that problem; mandatory serial work is an additional remedy authored by Claude. | Removed the serial-work rule. The request documents remain the assignments, including their cited dependencies. |
| [Implementation order](implementation-order-2026-11-09.md), `cd098202`: closure by deleting named functions; several consumers frozen until item 7 closed. | The operator asked for an implementation order that would stop work on obsolete dependents, then commissioned a memorandum of that order. This supports the dependency order. A freeze on named files and deletion as sufficient evidence of completion go further. | Removed the blanket freeze and closure shortcut; retained the producer/consumer dependencies and the named replacements. Expanded the opening user excerpt to include the problem it answered. |
| [The compiler](the-compiler.md), `07676c44`: the same modules, signatures, arguments, results, and two types as the example. | Copying the reference was a reasonable concrete start to the expressly requested equivalent compiler. That does not make its incidental interface shape a permanent limit on implementing the rest of mesh's requirements. | Removed the wording that could freeze the interfaces. The same data flow and stage responsibilities remain required, not merely illustrative. Removed unverified field-level implementation claims. |
| [Requests before the example](requests-before-the-example-2026-11-10.md): omit requirements with no stage in the example's existing mechanism. [Torch capture](torch-capture-change-request.md), `57d9f06e`: four items apply only if recording precedes the static schedule. | An obsolete mechanism was being used to discard the property it had been assigned to implement. The quoted instruction asks to restate all outstanding requirements. | Removed the exclusions. Changed the claimed completed restatement into the requested assignment. |
| [AOT A7](aot-compilation-change-request.md) and [plan-to-bridge R8](plan-to-bridge-change-request.md): one program only; remove any rule for several programs. | `727fc158` introduced the restriction on November 5. `7c1e068f` and `940cb181` carried it forward on November 10 while deleting its correction context. The operator was challenging a second allocator, not imposing a program-count limit. | Removed the limit and kept the cited sequencer and compile-time allocation requirements. |
| [Change-request format](change-requests.md): repository code and Claude-written documents cannot be sources. | This blanket exclusion conflicts with the format's own composition of requests in stated roles and the commissioned executable counterexample. | Removed the exclusion. Existing requirements for sourced methods and verbatim turns remain. |

The first group of corrections was committed as `71de11eb`. The original passages remain in Git history.

## Algorithm and citation corrections

In [integration-finishing](integration-finishing-change-request.md), item 2, `90b88505` combined completion times
with dependency levels as scheduling keys. `08864655` later added the statement that no cited source gave that
fallback. Item 2 now uses R5's completion time for every instruction and names R5's compilation scheduler as the
producer. The first review incorrectly dated that source-gap acknowledgment to the original commit.

Item 3 needs a different judgment. At `90b88505`, `rdma/mesh-graph.h` declares the scheduler's task-cost array as
`const double *length`; `plan_schedule` passes its supplied task weights in that position. Calling those inputs
task lengths is not, by itself, an invented weighting algorithm. The original item sought to restore task weights
after `9b1cc225` reduced the operand to transfer weights. Item 3 now states that execution-cost role explicitly.
The first review's treatment of this mapping as another unsupported substitution was too broad.

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
itself duplicate their algorithms. The individual historical reports remain attributed to their readers; the
aggregate direction verdict and its nine-of-ten total have been removed.

The [function-body reports](subproblem-function-bodies-2026-11-09.md) and
[ragged-computation inventory](ragged-computation-sources-2026-11-09.md) now state the limits of their source readings.
The latter no longer calls a search summary confirmation from the source itself. These edits do not certify the
remaining reports, counts, missing-caller claims, or quotations.

The [synchronization request](synchronization-signature-change-request.md), item 5, said completed playback had run
exactly the compiler's signature. In a change request this can express the monitor's intended postcondition; it is
not necessarily a claim about an executed implementation. The shorter wording retains the explicit count,
signature, and kernel-synchronization requirements elsewhere in the request. It removes a sentence that could be
misused as a substitute for those observations, not the matching requirement itself. [The index](../INDEX.md) also
no longer presents its linked prose as an account of how the system works now or names the obsolete
`.git/review-commentary/` output path.

## Context recovered on the second reading

The transcript locations below are under `~/.claude-personal/projects/-Users-mdot-metal-microbench/`. Times are UTC;
November 6 at 02:00 UTC is November 5 locally. Session `55974018-f56d-4c8b-8973-6fc61b49140c` contains the November
9–12 exchanges; session `66e06aa7-8a9f-469f-a879-1939b883b550` contains the earlier allocator and integration exchanges.
Human turns were distinguished from tool results, agent notifications, and generated compaction summaries.

### Implementation order and literal assignments answered different problems

On November 9 at 20:54, human turn `9ad3c771-9de9-4fa6-98f6-b69a54aae1b5` quoted a report that observation machinery
was being built around an executor already slated for replacement, then asked which modules needed complete
replacement in which order. At 20:55, assistant response `8c997453-c676-47f8-9a0c-a0d668f4e5a8` proposed both the
consumer freeze and deletion counts. At November 10 00:53, human turn `1ea69a18-4cc7-45d1-bc2f-1f12e53f381c`
commissioned the memorandum after an eleven-item ordering was presented. The order was requested; treating every
consumer edit as obsolete would lose the distinction between maintaining the old executor and preparing the new
consumer to receive its producer's results.

The later serial rule answered a different complaint. At November 12 20:32, assistant response
`a95dc0f2-d543-4644-8aeb-36c8e5ea3f2e` described replacing the document index with 611 clause entries and repeatedly
reclassifying them from hook reviews. Human turn `693a7a4b-5a0f-4fbd-b39f-909878307fed`, at 20:35, objected that this
activity displaced the assigned work and requested literal implementation assignments. The next response,
`e820510e-e72c-49a2-87d7-7f273f0d922c`, reported `c50fc92f` and the new one-item-alone rule. Avoiding invented
assignment text addresses the complaint. Serializing all implementation does not follow from it. Nor does this
review establish that every sequential work choice is wrong.

The earlier November 9 workflow already used sequential stages and a halt after an unsuccessful corrective round
(assistant response `315abb7e-313b-4757-93e2-0e58efd96d44`, November 10 03:48 UTC). That was a workflow choice;
`c50fc92f` made the one-item-alone wording part of the standing assignment document.

### The reference was required structure, not merely inspiration

Human turn `1b901df2-053f-49ac-a15c-47812d41a5e1`, November 10 20:26, required the example's exact data flow. Turn
`e2c5eb2a-f805-4c23-b32c-6d46c623f0fe`, at 21:57, requested an immediate, literal task implementing mesh's equivalent.
Within a minute, `07676c44` specified copying the modules as the first step, followed by machine placement,
transport, sharding, and Metal lowering. That context supports using the source directly as a starting point.
My first verdict failed to distinguish that useful implementation choice from freezing the resulting interfaces.

There is a concrete later report of the freeze problem. Agent notification
`714003d4-1d49-431f-853c-0adcc0888d1b`, November 12 14:59, says the agent moved device copies into graph nodes and
removed runtime `.to` calls, then backed the change out because three assertions stated the reference program's
graph. This is the agent's account, not an independently reproduced execution. It shows an asserted example shape
being offered as authority against a responsibility change. The corrective target is the handoff and the required
properties. Renaming the same stages, or merely calling them in the same order, would not satisfy that target.

### Removing a second allocator became removing concurrency

In session `66e06aa7…`, human turn `3c18cea7-7c14-4e1a-ad57-c1fc0691aeb5`, November 6 02:01, asked whether two
allocation goals were one compile-time problem. Turns `2e5cc8af-e81a-4e49-bfdf-6948a7edf56a` and
`1d6ee59b-f157-464a-812b-995bc8519600` challenged special rules about program counts and requested canonical methods.
Assistant response `1db182b2-80af-4bc6-9989-e2eabc02c125` first explained that program identity need not enter the
offline placement problem, then concluded that several-program rules should disappear. `727fc158` turned that
into one program on the collective. Removing program identity from the allocator's criteria does not prohibit
multiple computations. The invariant under discussion was compiling the relevant lifetimes and ordering together.

### Inventories and missing citations were being made to answer different questions

Human turn `9db015de-d3e1-4cf7-b728-889b994b372e` in session `55974018…`, November 10 01:11, asked where
subproblems had been solved piecemeal. Turn `10a2774e-ccb3-4806-aafe-4ce1ec56a8a8` then requested body reading when the
inventory admitted it had only searched names. These requests make the inventory useful for locating prior work;
they do not certify that the current composition uses it or is complete. The same operator later rejected another
generalization explicitly, in turn `278d9f93-3ec7-4745-95a1-ae106feba568`, at 01:42:

> this oversteps; we are remarking that a specific list of functions are not needed for a specific list of feature work and change requests.

The integration exchange supplies a related case. In session `66e06aa7…`, assistant response
`94f1dc5c-9beb-401a-801f-0f1781e250f3`, November 8 12:05, reported that only transfer completion times were published
and the replayed commands had no place in that schedule. Human turn `421c23a6-893c-4aa4-a630-28474eafbbb3`, at 12:21,
instructed it to finish integration. The subsequent mixed-key rule fits the incomplete interface it described;
it does not complete the shared ordering. This is why the correction concerns the producer's output and its
consumer, rather than whether every local parameter name appears in a paper.

### A selected numerical implementation became an exclusion rule

The numerical sequence in session `55974018…` is explicit. At November 10 03:51, human turn
`227f3152-a94e-471b-9225-2d2e6b9a9d4e` requested a literature counterexample as the lowering approach. Claude selected
iteration inside one supplier function, then made the selection CPU-only. The operator asked why, in turn
`494c33a3-6b98-49b9-981f-ccf26b365204`. Response `b45143a4-8912-40b3-a68a-615fbab04392` removed the separate bullet but
said the refusal behavior was unchanged. Turns `92f8ad15-8ac1-4190-9b06-5d7a1273afc6` and
`ba74d7f6-2c1c-468d-9d2e-8e2bb641026c` then directed the broader backend and expressly countered earlier limits.
The problem was not the initial selection of Accelerate; it was turning that selection, and the absence of another
citation yet, into a limit on the assigned result. The later unconditional `cond`-to-`select` rule has the same
scope loss at the source level: its batching condition disappeared.

Several exchanges show an agent's proposed remedy becoming a broader rule than the problem required. The valid
targets remain concrete: complete the replacement, preserve the required data flow, reuse solved methods, finish
the producer/consumer interfaces, and support the assigned program and processor scope.

## Applying the contextual review

The operator, 2026-11-13:

> wow very interesting malicious noncompliance on the part of the claudes. lets walk back through each of these issues, revise the documentation covered, remove expanded-precision or expanded-semantics reactions snuck into documentation and specification files, etc.

The revisions remove copies of the restrictions as well as their original operative clauses:

- The implementation-order memorandum no longer repeats the supplier-call-or-refusal rule for numerical
  operators. It cites source-lowering items 5 and 6 together. Its replacement scope follows the named compiler
  responsibilities, without prescribing a three-operation public interface from the reference's function count.
- The compilation coordination document no longer prohibits several programs or derives a single-writer register
  from that prohibition. The compile-time allocation assignment remains.
- Capture's offsets and lifetimes identify their allocation and lowering consumers; worklist execution uses the
  existing common instruction schedule and remapped dependencies. Missing-type observations were initially
  removed in `61a3e15e`; the operator's clarification below corrects that part of the review.
- The ragged-computation bibliography's additional change-request-writing checklist is removed. Its resulting
  assignment is the unequal-shards request. Source-reading qualifications remain attached to the bibliography.
- Source-lowering's unresolved non-Hermitian and target-lowering methods remain implementation obligations within
  the requested processor scope. Its qualification about which LAPACK routines the cited sources identify remains.

### The capture interface and the purpose of the requests

The operator asked what the requests were, who wrote them, and what grounded their requirements. The capture
request was introduced by `1d4c2ebe` on November 6, with a Claude Opus 5.5 coauthor trailer. The composition request
was introduced by this Codex session in `f0f96962` on November 8, following the operator's request to compose the
other requests in cited roles. The Codex session record is
`~/.codex/sessions/2026/11/08/rollout-2026-11-08T16-12-16-01a1bda6-0636-7763-a445-3776131aa8be.jsonl`;
the commit-producing tool calls are at 2026-11-09 01:53:20 UTC and, for `2936d23f`, 07:45:55 UTC.
Git's shared author identity alone does not distinguish these agents.

The purpose is stated in the operator's November 8 turn preserved in [review-warnings](review-warnings.md), turn 3:
the work orders express "do x with y" and catch added false requirements or constraints. The
[change-request format](change-requests.md) requires the originating turns, named methods and source bindings.
Composition's opening turns explicitly commission composition of existing requests in cited roles. These
instructions authorize situated implementations of the cited methods. A citation's role matters: an existing
implementation can identify what is to be replaced without prescribing it.

The shader-source interface appeared in `d62f205d` on November 7. The operator's typed-lowering correction was
recorded in `2936d23f` on November 8. Claude's capture rewrite `57d9f06e` on November 10 retained shader source as
an analysis input; `e5791525` then added missing-type observations. Thus the problem was not resolved merely by
having the correction elsewhere in the canon. Capture item 3 now takes the compiler's typed operations, bindings,
indexing maps and effects, as that correction requires. Its scheduling uses the existing program graph, including
the consumers of worklist results. The claim that a discontiguous kernel reads every partition is also removed:
the quoted source's decision to transfer an entire input did not establish that access fact for mesh's kernels.

This correction follows the operator's instruction and its compiler references, not a rule that the newest document
wins. It does not certify the compiler's implementation. The methods still missing from a request remain missing;
removing an unsupported restriction does not supply an algorithm or an execution result.

### Literal terminology and type discipline

The operator corrected this review's account of idiomaticity and specified the means of compliance. The complete
turns are in [the change-request standard](change-requests.md). Variable names, function names and other relevant
names use the words of specific sources or literature, allowing ordinary grammatical variation such as conjugation
and pluralization. Invented alternate metaphors are excluded. This is the requested type discipline; it does not
commission type systems, Pydantic models or other encodings of logical or formal types. Implementing agents do not
require a justification of the choice. Describing the constraint only as avoiding familiar implementation habits
understated it; reading it as a prohibition on grammatical variation also exceeds it.

Existing code with the right function under a wrong name remains assigned work. The operator specifies a
`#revise this; idiomatic vocabulary` note in one commit and a renaming commit afterward. A vocabulary finding
does not authorize an agent to reject the work order.

`61a3e15e` incorrectly grouped missing-type observations with unsupported restrictions. Those observations are
restored in capture, disjoint-pages, page-queue, plan-to-bridge, problems-of-14b9a757 and unequal-shards, alongside
the clarified producer/consumer assignments. Source-lowering's qualification about unidentified CPU routines is
also restored. These observations do not commission new types or formal typing machinery. The outstanding naming
work is to identify the specific source and use its names; deleting an observation does not establish that this
has been done. The present review does not claim to have completed that naming work.

## The interrupted placement patch

The operator's subsequent instruction to review and discard misassigned work, and the short statement of the
library's scope, are quoted in [the compiler work order](the-compiler.md).

The uncommitted `compiler/mesh_compiler/placement.py` patch came from the Claude agent assigned "Placement and
export: round seven" in session `55974018-f56d-4c8b-8973-6fc61b49140c`. Its record is
`~/.claude-personal/projects/-Users-mdot-metal-microbench/55974018-f56d-4c8b-8973-6fc61b49140c/subagents/agent-a76ce7f0730761034.jsonl`.
Assignment message `c718b916-772c-41ba-ac68-d46ea6771af3`, November 12 at 15:56:22 UTC, instructed the agent to add
window-sharding and flip cases to the existing placement implementation. It specified files the agent could edit
and said: "Do NOT start: LiteRT/delegate export, unequal-shards item 12's all-gather method choice, item 7's
cross-machine times, item 8(b) (these wait on the operator)." These are the supervisor's directions, not quotations
of the operator.

Tool calls in messages `5d255940-0372-456f-a7b3-eb7ce6131ca8` at 16:08:38 and
`71cdd124-e082-44c2-bdf8-362bfd2c062f` at 16:12:58 wrote the flip branch and the `region`/`gridded` changes. The patch
had 54 added and 12 removed lines. `gridded` added another loop that copies operator nodes and constructs shard
metadata, alongside the existing corresponding construction in `meshed`. It did not complete the scheduling,
lowering or native-execution connections described by the composition request. This is a source review; it does
not establish numerical behavior of the patch.

The coordinator stopped this agent at 16:14:09. The patch remained uncommitted through the present review.
The operator clarified that the contemporaneous halt/revert instruction concerned the terminal damage caused by
agents killing processes. It is not evidence that the placement patch had been ordered reverted then.

This evidence does not show that the placement worker lied about its immediate assignment: its changes followed
that assignment. The supervisor assigned incremental changes to the old implementation. Their mismatch with the
current replacement assignment, together with the operator's present authorization to discard such work, is the
basis for removing the patch. Treating its author's ownership as a reason to preserve it was an error in the
Codex readiness review.

The patch was recorded only for history in `6f0d2880`, then removed by `84d92713`. The resulting placement file has
blob `aa54a1f8c256429e59e5f6089df92ae1caa870cd`, identical to the committed file before that recording. No other
implementation commit was reverted. This disposes of the interrupted patch; it does not certify the older
placement implementation or retain it as the design for the replacement.

## Verification and limits

For `71de11eb` and `60636aac`, the edits were checked against the passages and sources described in the first review;
a comparison found no removed blockquoted lines, and `git diff --check` was clean. The second review corrects the
interpretations and provenance identified above. It expands one user excerpt to its full turn and restores the
execution-cost meaning of task weights. These are documentary checks, not evidence of compiler behavior or completion.

`61a3e15e` touched twelve existing documents. The terminology correction changes nine documents, including the
change-request standard. All pre-existing blockquoted lines remain, local Markdown link targets resolve, and
`git diff --check` passes. Those documentary passes did not change compiler source. The later placement removal
is recorded separately above and was checked against the original committed blob without executing the patch.

The November 13 compiler-boundary and hook memoranda already identify their inspected revisions and limits; they
were not promoted to requirements. Some requests still describe old concrete interfaces, and some source mappings
remain incomplete. This review removes the identified excess restrictions and false deductions; it does not
establish that every remaining requirement is sound or that every implementation obligation is closed.
