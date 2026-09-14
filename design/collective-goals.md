# Collective goals — the list that concludes

The [asynchronous collective contract](async-collectives.md) is the scope. This file is
the terminal-condition list for that scope: six goals, each with the artifact that
finishes it and the static check that shows it finished, plus the symbol allowlist and
the deletion list. Every normative sentence is a verbatim operator quote; the goals are
derived from those quotes and from code.

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

## The six goals

Each goal names its finishing artifact and a check that is read from source, not from a
run. A goal is done when its check holds; there is no further work under it.

### G1. Collectives are library functions on Refs

Artifact: `python/mesh/collective.py` exporting `send`, `reduce_scatter`, `all_gather`,
`all_reduce`, each a composition of `program.copy` and `kernel_call(add)` over blocks,
nothing else. `all_reduce = all_gather ∘ reduce_scatter` (Rabenseifner 2004;
Patarasuk & Yuan 2009). `send` is `program.copy`. Every block of every instance moves as
its own transfer edge on its own pages (memory mapping, D4–D6).
Check: `nn.ffn` has no `exchange=` parameter; its down-projection partials go through
`reduce_scatter` and its output through `all_gather`; `grep -n "exchange" python/mesh`
is empty. `examples/streaming-chain.py` calls `reduce_scatter`/`all_gather` by name.
Status 2026-09-14 11:13 (`0c5a4a3`): done.

### G2. Partial sums are a type, not a convention

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
completions and refills available queues without software waits; the algebra fires
from `mesh_events`. The operator permits additional hardware threads and shared
global memory wherever needed for independent progress. A single poller is not a
requirement.
Check: `grep -rn "scan\|while not .*ready\|sleep" python/mesh rdma/mesh-algebra.m` is
empty; `rdma/mesh-flow.c` contains no arithmetic on payload bytes.

### G6. The engine step uses G1 at the two Megatron points and is measured publicly

Artifact: in metal-microbench, the gemma-4 decode/prefill step calls existing kernels
and exactly `reduce_scatter` + `all_gather` per layer (Megatron g, ledger D11); solo
and two-node times are taken through the public server endpoint; the report states
Karp–Flatt e per run alongside the Amdahl bound from measured r and crossing cost.
Check: `grep -c "reduce_scatter\|all_gather" <caller>` equals 2 × layers; no benchmark
imports engine internals.

## Symbol allowlist

The library's public surface is exactly:

`Program, Program.tensor, Program.kernel_call, Program.copy, Program.replicate,
Program.export, Program.write, Program.constant, Program.realize, Program.close,
Tensor, Ref, BlockSpec, ShapeDtypeStruct, Result, kernels.arguments,
kernels.expression, kernels.dot, kernels.add, collective.send,
collective.reduce_scatter, collective.all_gather, collective.all_reduce, nn.linear,
nn.ffn, nn.rmsnorm, nn.embedding`

A function not reachable from these symbols and from `examples/streaming-chain.py` is
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
