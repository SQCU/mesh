# Collective goals — the list that concludes

Current operator requirements:

1. Higher-order functions over partial tensors.
2. Collective communication over those partial tensors, with each verb's actual semantics.
3. Zero-copy asynchronous execution of the call graph realized ahead of time.

The inherited numerical execution stack, expression compiler, Python package,
demonstration callers, Xonotic solver/planner and engine mesh adapter have been
deleted at the operator's direction. No replacement collective implementation is
claimed. The RDMA bridge, canonical pages and buffer ownership remain.

The six-goal wording and implementation inventory below describe the implementation
before deletion. They are historical cross-references, not authority to restore it.
In particular, a static function scan is not a requirement and has been deleted.


The [asynchronous collective contract](async-collectives.md) is the scope. This file is
the terminal-condition list for that scope: six goals, each with the artifact that
finishes it and the static check that shows it finished, plus the symbol allowlist and
the deletion list. Every normative sentence is a verbatim operator quote; the goals are
derived from those quotes and from code.

The [algorithms analysis](collective-algorithm-analysis.md) cross-references G1–G6
to the current execution and dataflow, derives the work introduced by contraction
lowering, and distinguishes local source deletion from completed integration.
Its [check analysis](collective-algorithm-analysis.md#9-what-the-goal-checks-establish-and-what-they-miss)
records the limits of the text searches below; they do not replace the semantic
source obligations.

## Operator quotes (normative)

Session `01a09bf3`, 2026-09-13T18:21:35Z:

> how do we implement the streaming producers and streaming consumers we want, which
> explicitly take chunks of a tensor which can be partially processed elementwise, and
> begin that partial processing without waiting for a big operand (which would have been
> fed into a matmul implementation one 'tile' at a time anyways once it was ready)?

Session `01a09bf3`, 2026-09-13T18:25:15Z:

> mesh can give us a way to use the rdma link through generalized linear algebraic
> operations and scatters and allgathers and reduces and stuff like that; a dnn ops
> library can then implement dnn modules using compositions of that linear algebra

> "I would make the first acceptance program streamed producer → peer sum → elementwise
> transform → contraction. Deliberately delay an unrelated extent" yeah so true bestie i
> agree, this is a good starting scope

Session `01a09bf3`, 2026-09-13T19:06:48Z:

> we should have a trivially deducible ring-of-buffers structure where we use MORE of
> the shmem on our peer computers by passing DIFFERENT OUTPUTs into DIFFERENT TARGET
> BUFFERS ... remember that we're here to implement the pallas async tp collective code,
> not to rewrite or reaffirm whatever ended up in mesh.

Session `01a09bf3`, 2026-09-13T19:27:14Z:

> it's time to add this inner 'mesh send function use' inside of the linear algebra
> kernels we use for our collectives so we never give users the impression they have the
> *option* to make or use mesh implementait[ions that stall]

Session `01a0a0ca`, 2026-09-14T16:42:04Z:

> we're workign on implementing a pallas like interface for distributed collective
> operations which don't block, don't expose low level synchronization or memory table
> implementation details to callers, and induce streaming operations by default for both
> producers and consumers of linear algebraic work being passed along meshes for work
> sharing. you're descirbing a whole lot of code which isn't part of hte mesh repo as it
> was last bashed out in specification, and also isn't necessary for the pallas like
> interface, suggesting most of the code you're 'fixing' incrementally is totally
> superfluosu

Session `01a0a0ca`, 2026-09-14T16:49:23Z:

> why are there cost models lol we're talking about a scituation where we specified
> asnyc producers, async consumers, and all responsibilty lands on the CALLER to supply
> a MESH AND TENSOR PLACEMENT CONFIG which is USEFUL instead of USELESS.

Session `01a0a0ca`, 2026-09-14T16:51:47Z:

> pallas is to provide an api header style and calling syntax which expose these
> qualities and nothing else. no features outside of htis narrow and tangible scope are
> required or *allowed*. strict dependencies of this featureset are *required* and are
> to be written with *urgency* instead of procrastination and hand-wringing. use
> established literautre or canonical implementatinos of this sort of code instead of
> inlining or inventing algorithmic code.

Current session, 2026-09-14:

> a definite list of goals that conclude in streaming tensor partial collective ops
> that, trivially, interleave and don't have a bunch of wrapping abstractions or control
> flow that stop them from being literally used

> it is so much simpler to literally implement distributed collective code instead of
> doing all of these other interventions though. the goal is really not open ended at
> all; it's *literally* 'there is a need for overlapped streaming tensor partials.
> implement overlapped streaming tensor partials. no, seriously, still streamed, still
> tensor, still partials, still overlapped.'

Current-session clarification:

> successful compilation is not important compared to the literal textual goals,
> which are all about producer async streaming consumer async streaming tensor ops
> that finish tensor parallel operations (the trivial obvious case of async
> producer/consumer meshing) faster than the same operations run standalone at
> worldsize 1.

> we do not even need to run the programs to know whether they satisfy our
> distributed collective async properties

The source obligation is the actual producer/transport/consumer dataflow: show
which partial each numerical function writes, how publication makes that partial
available, which configured transfer moves it, and which downstream computation can
use it while independent work continues. Known operands and indices describe these
relationships. A successful build, completed full tensor, or observed timing number
cannot replace that derivation.

Tensor parallelism serves the useful-work objective above. Compare the same tensor
algebra and numerical precision with world size one; the caller supplies useful
placement. Expose the work divided across participants and the communication or
extra launches on its critical path. Do not manufacture a gain by weakening the
single-participant implementation or hiding added work. This analysis does not
require a placement cost model, a benchmark framework, or running programs to
establish their asynchronous properties.

Current-session constraint on implementation work:

> we are talking about RDMA CODE ON WEAK PROCESSORS! we cannot HIDE any LATENCY
> behind SOME OTHER OPERATION and ASSUME we can WASTE A BUNCH OF OPERATIONS OR
> CONTEXT SHIFTS. that is why the requirements are SIMPEL and mention SIMPLICITY SO MUCH

Do the integration through the imported collective API. Prior demonstrated
pointwise tensor-parallel gains are not a reason to invent another acceptance
program. Remove avoidable hot-path work directly; overlap does not excuse it.

## The six goals

Each goal names its finishing artifact and a check that is read from source, not from a
run. A goal is done when its check holds; there is no further work under it.

### G1. Collectives are library functions on Refs

Artifact: `python/mesh/collective.py` implements the [collective verbs](collective-verbs.md)
through indexed transfers and numerical kernels. The operator's September 14
clarification requires each verb's actual semantics, rather than treating every
collective as a reduction, gather or scatter. Reduction trees process available
contributions without a preceding whole-tensor gather. The caller supplies placement.

Check: `nn.ffn` supplies local numerical contributions; the streaming example
selects `reduce` because its downstream consumer is on the root. Other callers
select their appropriate movement or reduction verb. There is no mandatory
reduce-scatter/all-gather pair or required count of collective names.

### G2. Partial sums carry their contribution sets

Artifact: a `Partial` marker on the Refs produced by K-split `dot` (DTensor `Partial`,
Legion `reduce` privilege). `kernel_call` refuses a `Partial` input to any expression
that is not `add`/`reduce_scatter` at setup, with an error naming the Ref.
Check: `nn.ffn` swish binds only non-`Partial` inputs; the refusal is one `raise` in
`kernel_call`; no runtime state.

### G3. Blocks publish when their writes are visible, and only then

Artifact: unchanged mechanism (`mesh_kernel_publication`, `publish_cpu`,
`complete_part`); the [end-to-end derivation](streaming-algebra.md#end-to-end-source-derivation)
lists every publish site of the chain in G1.
Check: no function in `rdma/mesh-algebra.m` waits on a command buffer or a transfer
before calling `mesh_publish_partial`; `grep -n "waitUntilCompleted\|dispatch_semaphore_wait" rdma/mesh-algebra.m`
is empty outside setup and teardown.

### G4. Receives are posted up front on planned pages; nothing is claimed or guarded

Artifact: `realize()` prints the planned arena bytes per participant (sum of every
instance's pages) and posts every configured receive (SDF balance: fixed bytes per
firing ⇒ static bound).
Check: `mesh_issue` has no claimability branch; `grep -n "claim\|EBUSY\|writable" rdma/mesh-dataflow.c`
returns only the host-writer path of `program.write`.

### G5. Transport drains and refills immediately, independently of numerical work

Artifact: `mesh_progress` in the bridge (NCCL proxy; MSCCL++ PortChannel) drains
completions and refills available queues without software waits. The operator permits additional hardware threads and shared
global memory wherever needed for independent progress. A single poller is not a
requirement.
Check: `grep -rn "scan\|while not .*ready\|sleep" python/mesh rdma/mesh-algebra.m` is
empty outside the explicitly requested `collective.sync_on_remote_fill` function;
`rdma/mesh-flow.c` contains no arithmetic on payload bytes. The completion wait is
never called by a default collective; its counterexample is caller code.

### G6. The engine step uses the collective required by its consumers

Artifact: in metal-microbench, the gemma-4 decode/prefill step calls existing
numerical kernels and the collective required by each distributed layer's
consumer placement. The caller supplies that placement; the collective library
contains no model interpretation or automatic choice of communication pattern.
Check: trace the actual distributed layer inputs and outputs through the canonical
streaming library. Runtime measurements are not a requirement or implementation
authority, as explicitly directed by the operator and the async contract.

## Implementation inventory and order — September 14, 2026

This is a source audit, not a replacement specification or a completion claim.
The audited baseline is mesh `main` at `1ed126d` plus the current uncommitted
receive/lifetime changes, and metal-microbench `main` at `1b0e930` plus its existing
working tree. The goal remains incomplete.

The operator has additionally made these requirements explicit in the current
conversation:

> your task here isn't to introduce incompatibility with core ml or ANE

> views aren't intrinsically contiguous

> a partial tensor can literally be made of several contiguous sectinos which are
> globally contiguous but are also locally contiguous so that tehy can be
> individually consumed by different contiguous-typed function launches.

> can refcounts be computed more easily / through a more isolated and lower
> synchronization/semaphore heavy approach than completioncounters? if refcount
> style and async background garbage collection is what we need an what we crave,
> lets do that instead.

> did the transcript say to write a refcounter and garbage collector? is it
> possible to write a refcounter for buffers over a known data flow when you know
> functions are completely feed forward?

### What is actually implemented

| Requirement | Source and present state | Remaining work |
| --- | --- | --- |
| Pallas-like functions, Refs, grids, indices and masks; caller chooses placement | `Program.kernel_call`, `BlockSpec`, `Tensor`, `Ref` in `python/mesh/__init__.py`; indexed lowering in `kernels.py`. Configuration precedes numerical issue. | Use this interface in the engine step. No replacement frontend is needed. |
| All distinct movement and reduction verbs | `python/mesh/collective.py` contains send/recv, broadcast, scatter, gather, all-gather, all-to-all, reduce, reduce-scatter, all-reduce and sum/max/min forms. Movement binds copies; reduction binds numerical trees. | The bridge currently connects one peer. General rank lists in Python do not implement multi-peer routing in native transport. Bind every edge to its configured peer when extending beyond the attached two-participant link. |
| Streaming numerical producers and consumers | `kernels.py` splits K contributions and publication regions; `nn.py` composes existing contractions, activations and sums. Native functions carry their actual input/output ranges. | Finish the contiguous-section binding for every retained backend and the engine caller. |
| Correct incomplete-sum semantics | `Partial`, `_ExpressionRegions.parts` and `Program.kernel_call` track contributions and reject premature nonlinear use during setup. | Retain this mechanism. It is not a runtime synchronization protocol. |
| Publication after actual writes | CPU `publish_cpu`/`mesh_publish_partial`; Metal and Core ML completion through `complete_part`; receive completion through `mesh_receive_complete`. | Preserve these sites while changing storage ownership. Early output publication must not release a producer that is still accessing storage. |
| Canonical registered operands and page indirection | `mesh_tensor_create`, `mesh_backing_alloc`, `mesh_page`, generated indexed loads/stores, `matrix_parts`, BLAS/BNNS address resolution, MPS view selection and host array mapping exist. | Core ML now binds named Refs through `mesh_algebra_predict` and resolves private setup mappings before prediction. Complete the engine encoder allocation/binding/receive/launch trace before deciding its required change. Capturing an address alone does not prove incompatibility, and strides alone do not prove contiguity. |
| Independent RDMA progress | `mesh-flow.c` has separate spinning TX and RX threads. They post and drain SEND/RECV; numerical work is elsewhere. The program's presence thread scans configured functions. | Integrate the preposting rewrite; do not introduce another scheduler or move numerical work into transport. |
| Planned preposting without destination occupancy checks | The uncommitted `link_receive` uses the planned physical-block list independently of index-message arrival. `link_deliver` maps completed blocks to logical destinations. Reader/occupancy checks were removed from numerical issue. | Not yet deployed or exercised. Finish section bindings, frame-length and lifetime integration before claiming G4 complete. Hardware capacity is a bounded window, replenished immediately, not unlimited simultaneous WR posting. |
| Distinct live values plus buffer refcounts and asynchronous pool return | The implementation subagent added one ownership word per publication section, configured numerical/send uses, automatic completion releases, Result/NumPy finalizers and a separate pool-return collector. Whole-client physical pinning is removed. | Native build and source review passed. Pool return is implemented; repeated invocation/transport-plan installation is separate unfinished integration. No caller free/done or per-page consumer-stamp protocol. |
| Repeated decode, continued prefill and verifier-shaped work | Existing engine has `DecodeParameters`, position/KV-aware `PrefillParameters`, `fullLogits`, and a verifier caller in `saguaro.swift`. | The working mesh patch issues each configured function once and rejects republishing the same Ref. It covers a finite configured graph, not continued engine operation. Realize the required distinct value instances and their recycling; do not silently narrow the engine to one invocation. |
| Existing Core ML/ANE and engine numerical implementations | Core ML uses `coreml.call` and `coreml.matmul` through the existing configured-function interface; inline native model generation and the special Program constructor path are deleted. The engine supplies add, GELU, normalization and weight loading. | Core ML is wired into the existing chain with `--backend coreml`; current evidence is source and builds. Finish the engine calling path while retaining its numerical implementations. |
| Actual distributed engine step | `examples/streaming-chain.py` already uses local projection → activation → projection → root reduction → activation → contraction. | `bootstrap.swift` decode/prefill still uses its local command-buffer graph and global activations. Exported engine numerical functions and a successful engine build do not satisfy G6. |
| Explicit bad synchronization path | `collective.sync_on_remote_fill` and `examples/sync-on-remote-fill.py` contain stream, sync, deadlock and independent modes. Defaults never call the wait. Historical finite and deadlock demonstrations are documented. | Run the existing independent mode and chain against the finished replacement. No new evaluator or implicit completion barrier is needed. |
| No scope expansion | The scoped numerical/transport source no longer contains the named cost/profile, sparse-routing, arbitrary-expression or reader-group mechanisms from the deletion list. | Keep removal structural. Do not remove required backends, numerical behavior, or needed lifetime ownership to improve deletion totals. Historical source totals are not current completion evidence. |

### Decisions that have looped and are now fixed

1. **Numerical functions stay outside transport.** Collectives select indexed
   communication relations and the supplied numerical operation. They do not
   interpret a model graph or implement every algorithm inside mesh.
2. **Placement belongs to the caller.** No cost model, automatic ownership search,
   or universal choice of gather/scatter/reduction belongs in this work.
3. **The page table is authoritative.** A tensor may contain several contiguous
   sections, each individually consumable by a contiguous-operand backend.
   Higher-order binding applies the existing function over those sections and
   maps their outputs back to the tensor's indices. Whole-tensor contiguity is
   neither assumed nor imposed. Backend support is retained.
4. **Availability and lifetime are different facts.** Presence permits reading a
   value. Existing operation completion releases that operation's buffer reference.
   The feed-forward graph supplies reference counts; background reclamation returns
   unused storage to the pool. Reclamation is not a prerequisite for unrelated work.
5. **Preposting does not await a destination's former readers.** Distinct live
   sections provide writable storage. RX fills the available hardware window from
   the plan and refills on completion; metadata identifies values without gating
   receive posting.
6. **Only explicit callers synchronize.** Input data dependencies remain local to
   the arithmetic that actually reads them. No automatic remote-fill wait, timeout,
   or consumer-retirement barrier is added.
7. **Integration is required.** A source deletion, exported symbol, successful build,
   or old demo run is not an implemented distributed decode/prefill step.

### Implementation order

The ownership work was delegated while source/dependency documentation and the
Core ML replacement proceeded locally. Steps below record the remaining dependencies,
not a requirement to reimplement completed work.

1. **Restore a coherent build.** Completed in this audit: repaired the misplaced
   Core ML initializer and the removed native entry point still used by the engine.
   Native, package, engine library and `forward_graph` builds pass locally.
2. **Finish canonical section binding.** Core ML now binds named Refs to a compiled
   model, with its invocation's virtual mappings resolved through the page table.
   The existing streaming chain uses that binding. Trace the engine encoder's
   allocation, binding, receive assignment, launch and completion before identifying
   its necessary change. Reuse its existing numerical functions and canonical storage.
3. **Static buffer references and asynchronous reclamation are implemented.** At setup,
   count actual numerical uses, sends and external borrows per owned buffer section.
   Include an unfinished producer's ownership when it publishes early. Feed existing
   completion releases to the reclamation owner; return zero-reference backing to
   the pool without clearing payload. Account for retained results and outstanding
   WRs. Numerical functions do not emit extra consumer stamps or wait for collection.
4. **Finish receive and instance integration.** ABI 34 now integrates preposting,
   page ownership permutation and automatic section lifetime. Repeated invocation
   and subsequent transport-plan installation remain unfinished. Preserve immediate planned preposting
   and page-table assignment; integrate reference ownership with arrival, forwarding,
   completion and subsequent instances. Remove the old reuse gates rather than
   transplanting them. Audit client/plan lifetime as well as payload lifetime so a
   new setup cannot overwrite metadata still used by an old WR. Treat one-shot
   `issued` state as a finite-instance implementation detail, not a restriction on
   repeated model execution.
5. **Demonstrate the actual streaming use in source.** Trace the existing
   `streaming-chain.py` through its configured functions, partial publication,
   transfer and downstream arithmetic. The independent mode supplies an explicit
   absent source and retained results. Account for configured transfers beyond the
   hardware window through immediate draining/refill. Compare the chain's tensor
   algebra and productive critical path with its world-size-one counterpart, now
   expressed by the same example without `--peer`/`--split`.
   Runtime demonstrations may illustrate this account; running them is not needed
   to establish the asynchronous properties and is not a substitute for the trace.
6. **Wire the real engine step.** In the engine's configuration and decode/prefill
   call graph, bind existing local kernels to canonical sections, supply caller-owned
   weight/activation placement and choose the collective required by the next
   consumer. Cover attention output and FFN projection boundaries actually distributed
   by that placement. Reuse the existing continued-prefill positions/KV and full-logit
   verifier paths. Keep model interpretation in the engine.
7. **Connect the engine's existing entry points to that dataflow.** Bind the
   configured model weights, placement, continued positions and verifier-shaped
   operands. Trace a produced section through communication into its actual
   numerical consumer. Show which work is divided and overlapped relative to the
   same world-size-one operation. Runtime timing, if reported, is separate evidence;
   no additional evaluator or timing-driven control flow belongs in the tensor path.

Steps 2–4 finish storage/backend interoperability before the changed transport is
presented as usable across supported backends. Step 5 uses the library; steps 6–7
finish G6. Existing collective verbs and numerical algorithms are dependencies to
reuse, not subjects to redesign in those steps.

### Integration dependencies and verified state

- **Toolchains:** local native, Python-package, `libgemma_metal.dylib` and
  `forward_graph` builds completed during this audit. Python source parses and
  `git diff --check` passes. No runtime tests were added or run. Logs are
  `/tmp/mesh-inventory-{native,package,engine}-build.log`.
- **Python/library dependencies:** `pyproject.toml` requires NumPy and the setuptools
  build toolchain. The native implementation uses system libRDMA, Accelerate, Metal,
  MPS and Core ML. JAX/Pallas and JACCL are documented prior art, not extra runtime
  dependencies of the implemented native path. The separate MLX dependency group
  is not needed by `uv run --no-project --with setuptools --with numpy` builds.
- **Core ML conversion:** `coreml.matmul(python, cache)` setup requires a Python environment
  containing `coremltools` and NumPy, the installed `mesh.coreml` module, and writable
  compiled-model storage. `coreml.call` accepts an already-compiled model directly. System Core ML is
  present. A local converter environment built a 64×128 by 128×256 FP32 section
  into `/tmp/mesh-coreml-section.mlmodelc`; the build log is
  `/tmp/mesh-coreml-section-build.log`. The peer converter environment has not
  been rechecked in this audit. A CPU-and-Neural-Engine configuration requests
  those compute units; it is not proof that every operation executes on ANE.
- **Participants:** local node 0 and `Ms-Mac-mini.local` node 1 are reachable. Both
  bridges were idle, ABI 32, 16,384-byte pages, four pages/block, 65,536 rows and one
  payload queue. The peer reports ready, client zero and error code zero. The working
  source is ABI 34; it has not been synchronized or installed on these bridges.
  Local and peer committed mesh source are `1ed126d`; peer engine is `1b0e930`.
- **Existing demonstration inputs:** both nodes contain the `.npy` chain operands,
  FP16 input/consumer variants and small GGUF fixtures in `/tmp/mesh-tp-consumer`.
  These fixtures do not stand in for the complete Gemma model needed by the engine.
- **Actual remaining runtime integration dependencies:** compatible section bindings,
  correct instance/reference ownership, identical committed ABI and native libraries,
  matching peer/shape/dtype/layout configuration, and the actual engine collective
  calls. New cost models, algorithms, runtimes and test frameworks are not dependencies.
- **Evidence limit:** previous two-node chain runs exercised earlier transport.
  They do not demonstrate the uncommitted ABI 34 preposting/refcount/Core ML implementation.
  Current source still incurs index traffic, presence scanning and publication work;
  its full-block wire framing pads short tails. No measured JACCL overhead parity
  or zero-latency claim has been established.

### Build record after ownership and Core ML integration

The combined native, Python package and engine builds passed. The engine used
its existing `libgemma_metal.dylib` and `forward_graph` targets. Logs are
`/tmp/mesh-coreml-refcount-native-build.log`,
`/tmp/mesh-coreml-refcount-package-build.log`, and
`/tmp/mesh-coreml-refcount-engine-build.log`. Source parses and whitespace checks pass.
The native build reports existing BNNS deprecations. These are build results, not
current distributed execution or ANE-performance evidence. No bridge was restarted
and no new test/evaluation program was introduced.

## Symbol allowlist

The library's public surface is exactly:

`Program, Program.tensor, Program.kernel_call, Program.copy, Program.replicate,
Program.export, Program.write, Program.constant, Program.realize, Program.close,
Tensor, Ref, BlockSpec, ShapeDtypeStruct, Result, kernels.arguments,
kernels.expression, kernels.dot, kernels.add, kernels.maximum, kernels.minimum,
collective.send, collective.recv, collective.recv_like, collective.broadcast,
collective.scatter, collective.gather, collective.all_gather, collective.all_to_all,
collective.reduce, collective.reduce_scatter, collective.sum_scatter,
collective.all_reduce, collective.all_sum, collective.all_max, collective.all_min,
collective.sync_on_remote_fill, nn.linear,
nn.ffn, nn.rmsnorm, nn.embedding`

The explicitly requested `examples/sync-on-remote-fill.py` is a counterexample,
not a default execution path or a new acceptance harness.

A function not reachable from these symbols and from these examples is
not part of the library. MLX's whole distributed surface is eight functions
(`all_sum, all_max, all_min, all_gather, send, recv, recv_like, sum_scatter`); this is
the size class.

## Deletion list (not goals)

The following exist in `python/mesh/kernels.py` (2,869 lines at `6368860`) and
`rdma/mesh-algebra.m` and are outside the contract. Delete with their callers; port no
caller onto the collective library.

- The scalar expression compiler beyond `add`, `dot`, cast, and the pointwise
  functions `nn.ffn`/`nn.rmsnorm` use: `argsort`, merge-path ordering, `philox4x32`,
  `random_normal`, `arange`, `logaddexp`, `asinh`, `expm1`, `log1p`, `power`,
  `floor_divide`, boolean/modular integer contractions (≈1,580 lines).
- Sparse routing, segment reductions, indexed products, routing directories,
  `indexed_add`, reader groups, route holds — the Xonotic solver's operations
  (≈920 lines; the solver is a separate caller and keeps its own operators).
- Cost profiles, function timing moments, compiled-code snapshots, "setup environment
  facts", plan metadata export, observation traces (`mesh_algebra_report` stays).
- Per-commit "evidence"/"provenance" JSON under `measurements/lowering-2026-09-13/`
  and any commit whose diff is documentation plus JSON only.
- Any `algorithm-sources.md` section that is not keyed to a symbol in the allowlist.
  The bibliography is one page: source per construct, not per commit.

## Process rules that stop the recurrence

Operator, 2026-09-14: "no features outside of htis narrow and tangible scope are
required or *allowed*". Derived:

1. An autonomous goal ("continue working toward the active thread goal") is admissible
   only if it names one of G1–G6 and its check. "FULL JAXIFICATION", "secure the bag",
   "lowering", "cost", "profile", "evidence" are not goals.
2. A commit is progress only if it moves a G-check from false to true or deletes from
   the deletion list. A commit that adds a bibliography section, a JSON record, or a
   trace field without changing a check is reverted.
3. Citations are per construct (§ allowlist), one line each, added when the construct
   is added. The [literature review](lit/review-2026-09-14.md) holds the rest.
4. Caller migration (Xonotic, or any other program) is never done inside the
   collective library. A caller uses the allowlist or does not use the library.
5. Static checks above are run by reading; there is no test suite. A "gold" harness is
   a test suite and is deleted (`examples/streaming-algebra.py`, 2,200 lines, removed
   at `4540397`).
