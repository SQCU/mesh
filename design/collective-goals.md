# Goal — the prepared machine

Operator, 2026-09-17, verbatim:

> we need to write code which has certain effects within a certain deadline. the
> lazy post hoc just in time assembly of the data structure needed to produce
> those effects is incompatible in latency and bandwidth w/ the latency deadline.
> therefore you must write, explicitly, what the layout of the computer MUST
> ALREADY BE LIKE IN RUNTIME for the rdma related code to have the required
> effects on the required schedule. if the runtime state of the computer MUST
> ALREADY EXHIBIT A CERTAIN STRUCTURE, the runtime memory of the computer and the
> corresponding pointers and offsets and related structures allowing linear
> algebra program execution MUST OCCUPY THE CORRECT STATES AHEAD OF TIME, and be
> ALREADY PREPARED TO OPERATE UPON RECEIVED DATA at the INSTANT that a polled
> result is submitted or received.

That paragraph is the whole requirement. D0, D1 and D2 below are its only
deliverables. Nothing else in this repository is a requirement, a task, or
evidence.

## Standing of every other document

Frozen — readable, not requirements, not tasks, not evidence:
`deliverables.md` (its I25/E1d/E8 objective survives as D2, its 29 rows do not),
`collective-goals-2026-09-14.md`, `async-collectives.md`, `h-audit-2026-09-16.md`,
`e2b-structural-latency-2026-09-17.md`, `instruction-adherence-2026-09-17.md`,
`e2b-crossover-2026-09-16.md`, `pages-and-functions.md`, `function-chain.md`,
`algorithm-sources.md` (citations only), `AGENDA.md`, `RELEASE-CLOSURE.md`,
every other file in `design/`, and `metal-microbench/docs/*`.

No agent edits this file, adds a link to it, or annotates it. A commit that
does is reverted and the turn is void.

## D0. The layout, as memory

One file, `design/prepared-machine.md`, committed before any source edit. It is
a table of memory, not an explanation. One row per object that must already
exist when `start()` returns, for every (rank, invocation slot, layer, peer) of
the E2B decode step on the pair (M5 Max + M4 Pro, B = 1):

| column | content |
| --- | --- |
| object | producer output / SEND record / receive backing / RECV record / consumer operand view / availability word / ring |
| address | a formula in `start()`-time integers only (`base + slot·S + layer·L + peer·P`) |
| bytes | the extent, and the `_Static_assert` that pins it |
| constructed at start() by | `file:line` |
| read by which event | exactly one of: value-ready, TX post, RX completion, consumer read |
| loads to reach it at runtime | `0` (address already held by the thread) or `1` (the ring integer). No other value is admissible. |

Then the two runtime transitions, each written as the exact sequence of loads
and stores, one line per instruction, with every address taken from the table
above:

1. value ready → SEND posted
2. RECV completion → consumer's first read of the operand

Pass value for each transition: one ring read, one 32-byte record line, the
payload. No load whose address depends on any runtime-loaded value other than
the ring integer. No allocation, no address rewrite, no lookup, no match, no
decision, no notification traversal.

A row whose address formula needs a value not known at `start()` is not a row.
It is a defect in the design, and the design changes until it is a row.

## D1. The source is D0, and nothing else runs on the hot path

Every D0 row cites the line in `rdma/mesh-*.c`, `rdma/mesh.h`, `swift/Mesh.swift`,
`metal-microbench/mesh_layer.swift` or `metal-microbench/matrix_shaders.swift`
that constructs it at `start()`.

Every load and store executed between a producer's last store and a consumer's
first read appears in a second table, against the D0 row it reads. A load that
reads nothing in D0 is deleted in the same commit. Code that constructs nothing
in D0 and executes nothing in the two transitions is deleted in the same commit
— not explained, not moved to another thread, not renamed, not documented as a
follow-up. Documentation of code that D1 deletes is deleted with it.

## D2. The step, on that machine

35 layer pushes + 1 vocabulary push per rank per step; each rank's partial is
pushed directly to every consumer and summed on arrival in any order; the plan
is printed at `start()`. Attention, norms, residual and PLE are duplicated on
every rank. The objective (`S ≥ 1.10` at B = 1 decode, overhead ≤ 0.5 ms per
forward) is unchanged, and is reached by D0 + D1 — not by separate work.

## Rules of action

- Order: D0 committed → source edits, each citing a D0 row → D1 tables. A source
  edit citing no row is reverted.
- `prepared-machine.md` and the D1 tables are the only documents this goal
  produces. Forbidden: analyses, audits, inventories, incident notes, follow-ups,
  scope corrections, verdicts, adherence or compliance records, counts of the
  operator's instructions, diagrams of code that exists, and explanations of why
  existing code exists. Reporting on your own noncompliance is not work; deleting
  the noncompliant lines is.
- The current source has no standing. Its jobs, threads, cadences, records and
  helpers are not preserved because they exist. A line survives only where a D0
  row cites it as that row's constructor.
- No subagent is spawned to measure, count, review or audit. Subagents write
  source.
- Runtime testing and benchmarks remain out of scope. Build, commit and push
  every buildable state.
- Each turn ends with exactly these lines and nothing else:
  `D0 rows: <constructed>/<total>` · `hot-path loads listed: <n>, deleted: <n>` ·
  `dependent-load depth T1/T2: <n>/<n>` · `commits: <hashes>`
- A turn that ends with more lines under `design/` or `docs/` than it began with,
  and no line deleted from a runtime source, has failed, and is reported as
  failed.
