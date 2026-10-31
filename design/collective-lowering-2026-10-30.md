# Collective lowering, logical steps to wire entries: the path as it is, and three forms that derive it once (2026-10-30)

Written while tracing c56f21a9's regression: rank 0 reaches a GPU wait deadline during the 16 MiB all-reduce,
then peers lose its link. Later output comparisons read unfinished results
([failure record](measurement-regressions-2026-10-30.md)). The path below is read at
0cd7af62. Each stage is named with what it takes, what it makes and what it re-derives.

## The path as it is

1. **Logical steps** (mesh-collective.c mesh_program_steps_cut, nccl-mesh.c routed_steps, counted_steps): each rank's
   SEND, COPY and REDUCE in that rank's order, every rank's computed on every rank.
2. **Pieces** (mesh-routed.c mesh_routed_compile): each matched SEND/receive becomes a transaction, routed over the
   pair's paths (mesh_write_compile), each path fragment one piece per rank it touches: (rank, local position,
   transaction, hop, offset). The generator makes both endpoints of a hop together, then forgets that they are a pair.
3. **Pairing, the first time** (number_pieces): every piece of every rank sorted on (round, hop, transaction, cable,
   offset); adjacent SEND and receive checked against each other and given one `round`, a pair number; then sorted
   back into rank order.
4. **Cuts** (refine_pieces, mesh-fabric.c mesh_fabric_refine): per rank, every SEND's latest overlapping receive found
   again by an O(n²) scan of byte overlaps, a quantum check per cut, both endpoints split by pair number, repeated
   until relays stop changing. An unrepresentable cut keeps its SEND whole (since c56f21a9; before it, the rank's
   every SEND stayed whole).
5. **Retirement proofs** (retirement_proofs, mesh-graph.h mesh_graph_mark): the dependency graph built from byte
   overlaps, its ancestors walked for each overwrite.
6. **Order** (dependency_order, mesh_transfer_order, mesh-graph.h mesh_graph_schedule since 1fa47bfc, page units filled highest first; mesh_write_order before it): per rank, the dependency
   edges derived again from byte overlaps (O(n²) a row); endpoints paired again, by sorting on
   (group, source, destination, cable, round, type, width, elements); a list schedule over the cables' capacities
   gives each transfer its wire order (`round`) and its receive its consumption order.
7. **Priors** (nccl-mesh.c schedule_group): each step's overlapping earlier receives found again, O(n²) per call.
8. **Forwarding** (forwards, send_dependencies): a SEND's last prior taken, its containment checked again, its unit
   offset found by walking the landing's spans; where the units do not line up, the dependency is dropped without a
   word (`if (!same) continue`), and the SEND then waits on nothing finer than its holding's publication.
9. **Barriers** (uses_add, nccl-mesh-metal.m metal_accesses, metal-recording.m MetalEncoderOrder): before each
   dispatch, its byte ranges declared and compared with a frontier of earlier ranges (dictionaries of 1 MiB buckets of
   NSData spans, keyed by allocation base or by page), a barrier where they meet. The same dependency graph a fifth
   time, now while encoding.

The dependency graph between page units is derived from byte overlaps five times, in four representations, and the
SEND/receive pairing three times. Each derivation has its own failure: EPROTO (3, 6), a SEND kept whole (4), a
dependency dropped (8), a barrier missed or added (9). A regression in one derivation shows up as a disagreement
with another, far from its cause.

## The regression's state

Eliminated by reading: the pairing in 6 (3's pair number is in its key, so it is exact); 6's graph against the Kahn
order it replaced (the same edges: a pair's publish before its consume, a receive's consume before a later SEND's
publish, a SEND's publish before a combine over its bytes; only the topological order chosen differs); data placement
(place_data's displacement is the old one for collectives); issue laying (a collective's `frames` is 1); the
frontier's aliasing in 9 (keys are allocation bases and pages). Found by two builds of main with one part undone
(design/measurement-regressions-2026-10-30.md): 4's cut owners. c56f21a9 made a REDUCE's units own cuts; with COPY-only
owners every case passes. Which later stage loses a combine-owned piece's data (8's summed forwarding is the first to
read) is the open question, and is itself an instance of the pattern above.

A further concrete mismatch was the progress operand for combining relays.
Consumption may differ from wire order, but the channel's shared summed word
claimed a completed physical prefix. A later receive could publish past an
unfinished earlier receive, which could then store a smaller prefix. Each
produced receive now has its own prepared progress word. Its relays name that
word; the existing kernel consumes it. The words are allocated and initialized
during placement, held with the issue's descriptor pages, and retain increasing
packet values across uses. Execution adds no allocation, reset, or serialization.
Actual-source preparation confirmed distinct producer words and exact relay
dependencies across reordered frames; the library builds. The saved failed run
lacks its prepared plan, so this defect is not established as that run's first
cause, and no numerical run of this change has been made. REDUCE-owned cut
refinement remains unfinished; COPY-only refinement is not the intended final
capability.

Preparation also reports missing page bindings as an error. Returning success
without preparing the group served the removed eager path; all calls now supply
their bindings through the same declaration.

## Three forms

**A. One graph, made where the facts are made.** Stage 2 emits each hop as a pair record, never as two pieces to be
matched: {transaction, hop, offset, bytes, sender, receiver, cable}, and with it the edges it knows (a relay's receive
produces what its SEND forwards; a combine consumes a local publication). Cutting is a function on that graph: split
an edge's interval at a page-cut set and its incident edges with it; pairs stay pairs by construction. Stages 5-9 read
the graph's edges: priors, forwarding (a forward is an edge, its unit offset stored on it), retirement and barriers
(each dispatch carries the indices of the dispatches it follows, decided here). Compile time: one O(E log E) build
instead of five overlap scans, three of them O(n²). Recording can then issue a
barrier from a precomputed bit, with no repeated range lookup per dispatch.
In the current prepared caller, recording itself precedes data-function execution.

**B. Page units as the representation.** Lower every step to the transport's real grain, page units: each unit one
producer event and its consumers, as compressed rows. Cuts disappear (every unit boundary is a cut), and so does the
fixed point over relays; forwarding is a unit's producer link; a wire entry is a run of consecutive units with the same
producer and consumer, coalesced last. More nodes than A (about bytes / page: 4096 for 64 MiB at 16 KiB), every pass
linear over arrays; absence is an empty row, never a sentinel value.

**C. Typed stages, each a pure function.** Keep the stages but give each its own type and make each a function of
the previous stage's immutable arrays: steps -> pairs (struct pair {send, receive}, the pair number its index) -> cut
pairs -> ordered pairs -> lists -> dispatches with their predecessors. No stage re-pairs, re-scans overlaps or drops
what it cannot represent: an unrepresentable cut is a compile error naming the step. The smallest change from today;
it removes the re-derivations (3's second and third pairing, 7, 8's walk) without changing the transport's grain.

All three replace sentinels (a parent of itself for "outside the tree", depth `n` for "unreached", `UINT32_MAX` for
"none") with explicit membership and explicit results, and settle the graph once at compile time, which rubric C7
(symmetric knowledge) and M9 (addressing from what is known) rank above re-deriving it where it runs.
