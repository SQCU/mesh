# Goal — Megatron TP2 over one simple collective

> Historical: the two-node targets of September 2026. The task now is [GOAL.md](../GOAL.md): any number of Macs,
> any cabling, no ordering or routing committed to ahead of the hardware.

Operator, 2026-09-22, verbatim:

> try to avoid inventing distractions or reasons to focus on anything besides
> whether there is distributed collective code implementing the world's simplest
> tensor parallelism pattern (megatron) with interleaved execution of work (rather
> than inventing reasons to synchronize or do non-overlapping non-interleaved
> execution of work on different cpus with different clock domains etc.). we are
> clearly running into issues of measurement and theory seen in the last round of
> implementation: someone keeps thinking there is some extra goal or extra problem
> which supersedes the basic telic purpose of the code, which is unambiguously
> megatron-lm style simple tp2 thru an incredibly simple distributed collective
> operation.

## The task

- **Megatron-LM TP2** (Shoeybi et al. 2019): gate/up column-split, the activation
  product local, down row-split; attention split by heads as query/output
  projection pairs (respecting GQA grouping); vocabulary tiles split. Placement is
  supplied by the caller.
- **One simple collective per projection pair.** Each rank sends its
  codomain-shaped row-split partial once, directly to its peer, and partials are
  summed on arrival in any order. Head-split attention adds its own o_proj
  collective: one per attention block and one per MLP block, as in Megatron.
  Norms, residual and PLE may be duplicated on every rank. Duplicated or
  head-split attention is the caller's placement choice. RS+AG is used only for
  large tiles.
- **The mechanism:** one SEND into a posted RECV; presence is the receive
  completion; dedicated TX/RX threads that only drain and post.
- **Interleaved execution.** Producers publish when their writes are visible and
  return; no guard, sync or wait withholds ready work; partials stream and are
  consumed as they arrive, so the two nodes' work overlaps. Operator, 2026-09-14:
  "OF COURSE THE WHOLE THING HAS TO BE ZEROCOPY AND ASYNC ONCE WE'VE FIGURED OUT THE
  CALLGRAPH AOT", and "the *runtime* is necessarily extremely ismple, ebcause any
  artificially imposed control flow puts more cache reads and loads in between
  'work is ready on peer' and 'work has arrived to this meshnode from peer and is
  being used'".

## The two targets

- **Goal A.** One gemma-4-12B FFN layer split M5 (11,904 neurons) / M4 (3,456
  neurons) with a streaming reduce-scatter/all-gather exchange: median
  completed-result period ≤ 20.40 ms, against the recorded 26.12 ms for the M5
  alone. Any build that meets the period counts.
- **Goal B.** gemma-4-E2B decode (any rows per forward: batch, sessions, draft-verify width; B=1 is one point, never the target), attention heads [7,1] and FFN 7:1: pair
  decoded tokens/s ≥ 1.035× solo decoded tokens/s (recorded best 1.035–1.045×).

Pair and solo tokens may differ (row-split reduction order). Decoded-token
identity with solo is not a requirement, criterion or diagnostic target; never
report it as a finding (operator 2026-09-08, 2026-09-18, 2026-09-22).

## Design notes

Operator, 2026-09-17, verbatim, on latency:

> we need to write code which has certain effects within a certain deadline. the
> lazy post hoc just in time assembly of the data structure needed to produce
> those effects is incompatible in latency and bandwidth w/ the latency deadline.
> therefore oyu must write, explicitly, what the layout of hte comptuer MUST
> ALREADY BE LIKE IN RUNTIME for the rdma related code to have the required
> effects on the required schedule. [...]

Operator, 2026-09-18, verbatim:

> numerical checks... do not matter... other htan the numerical check... of
> tokens/sec decoded... [...] that of the persistent kernel, and that of collective
> code which does literally nothing but pass operands to a device resident
> function [...]

Operator, 2026-09-18, verbatim:

> idk you coach the codex through running this, measuring performance,
> etcetera. the m4p machine is there to be used by this research agenda;
> there's no inhibition against runtime testing if what's being tested is
> following the design documents

Ops notes: the Mini builds from pushed source; run outputs go to `output_data`.
