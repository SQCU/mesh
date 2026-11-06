/* The bridge's data path, in a port's child (mesh-child.c): one cable and its queue pairs (the tag's lanes, the link's
   `plan`), serving this node's share of the region's page lists (mesh.h struct mesh_hop) by the rule both ends compute
   (design/page-queue-change-request.md, "Claude's proposal": "The page lists (the queue)", "What the bridge decides",
   "The rule both ends compute", "The functions"; and the operator's "The page queue" and "Interleaving every
   transfer": every inbound link used, a ready page waiting only behind earlier pages of its own queue pair), as
   design/disjoint-pages-change-request.md, Appendix C, C3, changes it (design/disjoint-pages-revision-coordination.md,
   "The bridge"): a plan is u copies of one execution's hops, its template, execution t running copy t mod u, every
   page placed once by the plan's one allocator (C3.3, modulo variable expansion: M. Lam, Software pipelining, PLDI
   1988), so it runs any number of executions, the client's start saying how many (the plan's state word `executions`,
   which only rises: the contract's D2); and its state lives apart from its pages, in the plan's state section (mesh.h
   struct mesh_state; C3.7), no word of it written to an earlier value: a landing flips its receive's 2-bit Gray-coded
   phase (PTX's mbarrier phase bit, widened to two) and adds one to its range's count of its copy (counted completion,
   C3.1: Portals 4's counting events; the contract's D1), and a cancellation arrives on that count with no phase written
   (C3.1).  A send of execution t is released by its range's supplied word reading at least t + 1, or, forwarding, by
   its receive's phase of execution t.  Nothing here writes a data page.
   Its native forms are the d6dceac transport's (MESH_VERSION 104, restored 2026-10-01: the source of the fastest
   recorded E2B decode, metal-microbench docs/measurement.md): SENDs prepared once a plan, each posted in its queue
   pair's order as its word releases it; receives prepared once a plan, posted in their queue pair's order as it has
   room, a receive record again only once its last posting has landed; a record of either posted once for each
   execution of its copy (link_prepare, link_receive_drain); completions polled through the
   provider's own functions (mesh-verbs.h struct mesh_verbs `poll`, struct mesh_queue `send`, `receive`); traced
   variants under MESH_LEDGER.  Their inputs are the page lists, which replaced d6dceac's per-link, per-queue transfer
   lists and its --link form's link_configure (both since removed).  The progress thread, the guard around its device's
   verbs calls and a port's halts are tag pre-return-20261103 rdma/mesh-link.c's, verbatim but at interfaces.  No main
   (the binary's is mesh-link.c's, the child's mesh-child.c's) and no verbs teardown: a halt clears `served` and moves
   `pairing`, and the child's exit is the teardown (mesh-child.c child_leave, lanes_flush; down_pair stays deleted).
   2026-11-04, sub-assignment 6: reversing this note's 2026-11-03 form (every execution's entries unrolled into the
   list, a landing's bit ORed into its execution's landed record, a cancellation UINT64_MAX in that record's cancel
   word), which unrolled by the run's length and gave the landed record five writers (C1; C4).  What must stay: the
   k-th SEND on a queue pair fills the k-th receive posted on it; nothing is written in or beside a page; no request is
   taken back but by the child's exit.  mesh-dataflow.h is no longer included: mesh_cables, the one thing this file
   took from it, is mesh-map.h's (through mesh-bridge.h).
   2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, D.3.5 and D.4; design/disjoint-pages-rulings-
   coordination.md, R1): reversing this note's receives "posted ahead as their queue pair has room", under which a
   receive record was posted again while an earlier posting of it might be outstanding: a received page is now named
   once per landing.  What must stay: as above.
   2026-11-04, owner B1 (design/plan-to-bridge-change-request.md R5 and R8; design/aot-compilation-change-request.md
   A7; design/compilation-coordination.md, T7, Q4 and Q5): reversing this note's one plan of the region (hdr `hops`,
   `plan`, `state` and `client`: a plan a port loaded while its one client held the region).  Any number of plans stand
   at once, each in its claim (its head's publication `state`, mesh.h struct mesh_buffer's note), and a port's child
   finds them by hdr `plans` and a walk of the claims' heads (e6bd7db2's section 6, "Discovery", as the contract's
   ruling 1 and Q5 take it); it loads each with records of its own (struct flow, one a plan), tells that it holds it in
   the plan's `held` or that it refused it in `refused`, serves its executions t < `executions` and t < `released`
   (Q4: the order the island's fixed sequencer gave, which every node delivers in sequence-number order: X. Défago, A.
   Schiper and P. Urbán, Total order broadcast and multicast algorithms: taxonomy and survey, ACM Comput. Surv. 36(4),
   2004, section 7.1, Fig. 5; first in, first out, P. Barham et al., Pathways: asynchronous distributed dataflow for
   ML, MLSys 2022, section 4.4), tells what it has served in `served`, and ends each plan whose claim closed.  Every
   hop moves its `bytes` (T7), where every hop moved its whole page.  mesh-dataflow.h is included again, for the walk
   (mesh_head_after, mesh_head_state), reversing sub-assignment 6's note above.  Why: "nothing happens when a program
   joins or leaves because a program is a sequence of like, function calls" (R8).  What must stay: the k-th SEND on a
   queue pair fills the k-th receive posted on it; a cable carries each execution whole, executions in order; nothing
   written in or beside a page; no request taken back but by the child's exit.
   2026-11-04, owner B3 (design/compilation-coordination.md, "The appendix's revisions", O2): "a cable carries each
   execution whole, executions in order" above, the hold that serves one plan's batch at a time on this port's cable
   (link_run; the parent's orders_served, mesh-records.c), stays, for O2's reason and not the first contract's ruling 1
   ("A cable never interleaves two executions", Claude's words, which design/aot-compilation-change-request.md,
   appendix item 2, withdraws): on each queue pair the k-th SEND fills the k-th receive posted (Hilland et al., RDMA
   Protocol Verbs Specification, section 8.2.2, design/prepared-machine.md#M08), so both ends of the cable must post the
   released executions of several plans on each queue pair in one order; this child learns of its parent's deliveries
   only each plan's `released`, a count, so the order of deliveries across plans does not reach it; and carrying it
   needs a structure the change requests do not name.  No behavior changed.  What must stay: as above, until the
   operator names what carries the delivery order to the ports (O2, open).
   2026-11-04, owner B3 (design/aot-compilation-change-request.md, appendix item 5; design/compilation-coordination.md,
   "The appendix's revisions", S4 to S7): a hop's `set` (mesh.h struct mesh_hop) is read, where every hop was one of a
   sequence: each execution of a set is posted on the cable in a permutation of its members on that cable (the
   positions the deal gave it: S7') drawn by Knuth's Algorithm P (set_draw) from Threefry-4x64-20 (threefry4x64_20)
   keyed by the plan's name and counted by (execution, set, block, the cable's place in key order), so both ends draw
   the same one with nothing exchanged (link_prepare, position_record).  What must stay: as above; a sequence served
   as before.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", OP and LR): one permutation of a set's
   every position on the pair each execution, counted by (execution, set, block) alone, each cable posting its own
   members in its order, and each of Algorithm P's integers by Lemire's rejection (set_draw), where each cable drew its
   members' own permutation counted by its place in key order (S7′) and took the product's high word alone.  Why:
   design/aot-compilation-change-request.md, appendix item 5.  What must stay: as above.
   2026-11-04, owner B4 (design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node";
   design/compilation-coordination.md, "The documents' revisions of edd4b583", DL1 to DL5), reversing the hold the two
   notes above keep ("a cable carries each execution whole, executions in order", one plan's batch at a time,
   `served` told once a batch was served whole): the port serves every standing plan at once, interleaved.  The bridge
   parent publishes each released execution's sequence number in its plan's sequence word of its copy, then raises the
   plan's `released` (mesh-records.c orders_deliver: L. Lamport, Concurrent reading and writing, CACM 20(11), 1977,
   "Message Buffer", the sender); each lane of this port, each way, merges the plans it carries by those numbers
   (lane_take: the item's k-way merge, the least first, each sequence read as Lamport's receiver reads, in the opposite
   order of the writing, the conclusion's "writing data elements in one order and reading them in the opposite order"),
   posting each execution it takes in order as before, then the next; each record names its plan (`flow`), so the
   completions of several plans on one queue pair count apart (DL5); and the port tells, in each plan's `taken[p]`,
   the least execution a lane of it may still read (taken_tell; Lamport's mr), on which the parent's sender waits
   before it reuses a word.  So the delivery order reaches the ports in the words, which O2 found nothing to carry.  Why:
   the item: "On each cable, the pages of every released execution interleave"; "Each port serves the released
   executions of every standing plan in sequence-number order: a k-way merge"; "Both ends of a cable read the same
   sequence numbers, so they merge alike.  Nothing holds."  What must stay: the k-th SEND on a queue pair fills the
   k-th receive posted on it (both ends hold the same plans and records on each lane and read the same numbers); each
   plan's executions on a lane in order; nothing written in or beside a page; no request taken back but by the child's
   exit */
#include "mesh-bridge.h"
/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5): mesh-dataflow.h
   is no longer included, reversing owner B1's note above ("included again, for the walk"): the walk of the claims'
   heads (mesh_head_after, mesh_head_state) is gone, and the program's plans are read at their places through mesh.h
   (mesh_program, mesh_program_plan; mesh-bridge.h plan_from).  What must stay: nothing of the library's client
   interface here */
#include "mesh-graph.h"
/* 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-8: "a
   gone holder is read from the heartbeat's words"; "The revisions of 14b9a757", R-8): mesh-holders.h, for the
   holder's one read (mesh_algo_holder) and its test (mesh_holds), which plan_stands takes.  What must stay: the holder
   read there alone */
#include "mesh-holders.h"

/* design/prepared-machine.md#M29: d6dceac's prepared SEND (its span and request, one cache line's worth of what the
   post reads), its release its entry's `word` and `field` (mesh.h struct mesh_hop; design/disjoint-pages-change-
   request.md, C3.7): from its range's source (`field` MESH_ABSENT), execution t's send released where the range's
   supplied word reads (acquire) at least t + 1, its producer's release of t; forwarding (a relay's, or a destination's
   that sends the page on), where its receive's phase field, bits [field, field + 2) of `word`, reads (acquire)
   mesh_phase(t / u + 1), that receive's landing of t; t the execution its lane's cursor serves (struct cursor).  The
   word is the plan's state section's, the record the child's.  One record a template entry, posted once for each
   execution of its copy (link_prepare).
   2026-11-04, sub-assignment 6: `mask` (2026-11-03: the bits of its word its release needed, MESH_SUPPLIED or its
   receive's bit of its execution's landed record) is `field`, the landed records and MESH_SUPPLIED gone (C4).  What
   must stay: the word and field in the line the post reads.
   2026-11-04, owner B1 (design/compilation-coordination.md, T7): its span its hop's `bytes` long, where it was the
   whole page (page_bytes()), and its frames frames_of(that length), posted and retired through its lane's gate alike
   (link_send_drain: the completion names the record, so the record knows its frames), where every SEND took a page's
   frames.  Why: R5, each hop with its bytes.  What must stay: one 256-byte cell, its span and request where the post
   reads them.
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL5: "each record
   names its plan (struct flow), so a pass's completions are counted to their own plans while several interleave on
   one queue pair"): `flow`, the plan whose record it is, in the cell's room after the request (offset 160: the
   request, struct ibv_send_wr, is 128 bytes from 32), so the SEND completion, which names the record (`wr_id`),
   counts its plan's `sent`; no field the post reads moves.  What must stay: as above */
struct flow;
struct prepared_send {
  _Alignas(128) _Atomic uint64_t *word;
  uint32_t field,padding;
  struct ibv_sge span;
  struct ibv_send_wr request;
  struct flow *flow;
};
_Static_assert(sizeof(struct prepared_send)==256 && _Alignof(struct prepared_send)==128 &&
  offsetof(struct prepared_send,field)==8 && offsetof(struct prepared_send,span)==16 &&
  offsetof(struct prepared_send,request)==32 &&
  offsetof(struct prepared_send,request.send_flags)+sizeof(unsigned int)<=128 &&
  offsetof(struct prepared_send,flow)==160,"M29");
/* design/prepared-machine.md#M08: d6dceac's prepared receive, its request and span verbatim, and what its landing
   writes (design/disjoint-pages-change-request.md, C3.1 and C3.7; design/disjoint-pages-revision-coordination.md, "The
   bridge"): its phase field, bits [field, field + 2) of `word`, and `count`, its range's count word of its copy, which
   takes the range's `pages` arrivals an execution, NULL where its entry's `count` is 0 (a relay's own receive: its
   forwards are released by its phase and nobody waits on a count of it, so it adds, wakes and is raised for nothing;
   2026-11-04, the coordinator's ruling on review 15's finding 1); `landings` the record's completions since the load, the progress
   thread's alone (its j-th is execution copy + j u: link_prepare), which also holds back its next posting: its round
   j is posted only once `landings` reads j, so at most one posting of it is outstanding and its page is named by at
   most one posted receive (link_receive_drain; 2026-11-04, sub-assignment 1: design/disjoint-pages-change-request.md,
   D.3.5, design/disjoint-pages-rulings-coordination.md, R1, reversing its posting again while an earlier posting
   might be outstanding; what must stay: the record's request never changed once written); and `copy` its copy.  Every
   field its landing reads is in its second 64 bytes, the 128-byte line its request starts.
   2026-11-04, sub-assignment 6: reversing 2026-11-03's `input`, `mask` and `cancel` (its bit of its execution's landed
   record, ORed in, and that record's cancel word, stored UINT64_MAX by link_close: C4); `next` (its queue pair's chain
   of receives each posted once) and `pair` gone, the lane's cursor walking its records (struct cursor) and the queue
   pair the lane's.  What must stay: the record one line, its request never changed once link_prepare has written it.
   2026-11-04, owner B1 (design/compilation-coordination.md, T7): its span its hop's `bytes` long, where it was the
   whole page: a landing of another length is refused (EMSGSIZE, the link's failure: link_receive_drain) and its frames
   are frames_of(bytes), so a landing reads its span's length too, in the record's one 128-byte line.  What must stay:
   as above.
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL5): `flow`, the
   plan whose record it is, in the line's room after `padding` (offset 104), so a landing, which names its record
   (`wr_id`), counts its own plan's `landed` and writes its own plan's trace while several plans' receives interleave
   on one queue pair; no field the post or the landing reads moves, and the record stays one 128-byte line.  What must
   stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", PM): `round` where `landings` was, the
   same 8 bytes: the round j = t / u of the record's posting, stored as it is posted and read by its landing, which
   flips bit field + (j & 1) and wakes where the count it made is (j + 1) x `pages`; a record of round j is posted only
   once its page's phase field reads mesh_phase(j), its round j - 1 landed (link_receive_drain).  Why:
   design/aot-compilation-change-request.md, appendix item 5: a set's member is posted at any of the set's positions
   on the pair's cables, each port's record of it a view of its page, so a record's own completions no longer count its
   page's landings; the phase field is the page's own (mesh.h, "Phases"), one for every port's child.  What must stay:
   at most one posting of a page outstanding; the record one line, its request never changed once written */
struct prepared_receive {
  _Alignas(128) struct ibv_recv_wr request;
  struct ibv_sge span;
  _Alignas(64) _Atomic uint64_t *word;
  _Atomic uint64_t *count;
  uint64_t round;
  uint32_t field,pages,copy,padding;
  struct flow *flow;
};
_Static_assert(sizeof(struct prepared_receive)==128 && _Alignof(struct prepared_receive)==128 &&
  offsetof(struct prepared_receive,span)==32 && offsetof(struct prepared_receive,word)==64 &&
  offsetof(struct prepared_receive,count)==72 && offsetof(struct prepared_receive,round)==80 &&
  offsetof(struct prepared_receive,field)==88 && offsetof(struct prepared_receive,pages)==92 &&
  offsetof(struct prepared_receive,copy)==96 && offsetof(struct prepared_receive,flow)==104,"M08 M09");
/* design/prepared-machine.md#M27 */
struct mesh_trace { _Alignas(32) uint64_t identity; uint64_t begin,middle,end; };
_Static_assert(sizeof(struct mesh_trace)==32 && _Alignof(struct mesh_trace)==32,"M27");
/* A lane's place in the executions it serves one way (design/disjoint-pages-revision-coordination.md, "The bridge":
   each lane in execution order, execution t's records on it copy t mod u's, in assignment order): execution t
   (`execution`), its copy t mod u (`copy`) and t / u (`round`), and the records of that copy on the lane still to post
   in t, indexes [at, end) of its direction's records.  Kept in registers through a pass (link_receive_drain,
   link_send_drain), so a post costs no division.  2026-11-04, owner B4 (design/compilation-coordination.md, "The
   documents' revisions of edd4b583", DL3): a lane has one cursor for each plan it carries each way, each plan's
   executions there in order, and the merge (lane_take) takes the next of them by sequence number; a cursor is kept in
   registers while its plan's execution is posted.  What must stay: each plan's executions on a lane in order */
struct cursor { uint64_t execution,round,at,end; uint32_t copy,padding; };
/* One standing plan as a port's child serves it, the progress thread's alone (design/compilation-coordination.md,
   Q5: records per plan per port): the claim it stands in (its head row `head`, and the head's `owner` word and
   `generation` as its load read them: plan_stands); its state (`words`, its `copies` u); its records by lane, copy and
   assignment order (lane q's copy c at [first[q (u + 1) + c], first[q (u + 1) + c + 1]) of its direction's), each
   lane's cursor each way, the lanes that carry each way (`carrying`, `lanes` either); the value this port last stored
   in the plan's `taken[p]` (`told`: taken_tell); its completions (`sent`, `landed`, each counted through its record's
   `flow`) and the requests it has posted either way (`posted`); and the traces (MESH_LEDGER).  Served (FLOW_SERVING),
   each lane it carries taking its executions by the merge (lane_take); or cancelled (FLOW_CANCELLED: its `serial` no
   longer this node's `map_serial`, so the bridge parent has cancelled it, plan_stale, D.3.6), served no more, its
   `taken[p]` following its `released` (DL4).
   2026-11-04, owner B1 (Q5; e6bd7db2's section 6): reversing the one flow a port (`flows`: the region's one plan, by
   hdr `hops`, `plan` and `client`, with its states none, refused and ended).  A port holds every standing plan, a flow
   each, listed (struct port_plans); its refusal of a plan is that plan's `refused` word, not a state here; and a plan
   ended with requests outstanding is the port's (`ended`).  Why: any number of plans stand at once (R8).  What must
   stay: a record's request unchanged once written; each lane's executions in order.
   2026-11-04, owner B3 (design/aot-compilation-change-request.md, appendix item 5; design/compilation-coordination.md,
   "The appendix's revisions", S4 to S7): beside the records, each direction's sets as link_prepare finds them (a set
   the direction's picks within one copy with the same nonzero `set`, wherever they lie in the list: 2026-11-05,
   design/compilation-coordination.md, "The revisions of 67ca86f8", SP, where it was a maximal run of them, S3, S7), in
   parallel arrays, the
   records' layout (M29, M08) unchanged: `key` the plan's name zero-padded, the generator's key (S4), and `place` this
   cable's place in the pair's key order, the counter's fourth word (S7'); `sets[d]` each set's `set`, its size m on
   this cable and the index in `members[d]` of its first member (struct flow_set); `members[d]` each set's members on
   this cable, their records in list order, X_1 .. X_m (S5, S7': the members whose positions the deal gave this
   cable); `position_set[d][r]` the set whose position
   record r is (UINT32_MAX: a sequence's position) and `position_index[d][r]` its index in that set, from 0 (NULL both
   where the direction has no set); and per lane, `drawn[d][q]` the permutation of (`drawn_execution`, `drawn_set`)
   its cursor last met, room for the largest set the lane carries a position of, taken at the load (S7: nothing
   allocated at run time).  Why: each execution of a set is posted in a uniformly random permutation of its members,
   both ends drawing the same one (item 5).  What must stay: as above; a sequence's positions posted as before.
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL3 and DL4: "What
   goes: ... link_run's serving of one plan at a time and its batch bound per plan (`released` as a count of a batch
   ...); the ports' `served` as "served whole""): `executions` (the batch bound, the lesser of `executions` and
   `released`), `send_total` and `receive_total` (the completions that bound asked) and FLOW_DONE (every record done
   to the bound, which `served[p]` told) are removed.  A loaded plan not cancelled (FLOW_SERVING) is a candidate of the
   merge on every lane it carries each way (lane_take), its cursors there its next executions, each taken by the
   lane when its number is the least (DL3); `told` is the value this port last stored in the plan's `taken[p]`
   (taken_tell, its starting value the word's at the load); `sent`, `landed` and `posted` count to the plan through
   its records' `flow` (DL5).  FLOW_CANCELLED as before, its `taken[p]` following `released` (DL4).  Why: the item:
   "Each port serves the released executions of every standing plan in sequence-number order: a k-way merge";
   "Nothing holds" (design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node").  What must stay:
   a record's request unchanged once written; each lane's executions of a plan in order.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", OP): reversing S7′'s per-cable draw
   (`place`, this cable's place in the pair's key order, the counter's fourth word, and each set's size m its members
   on this cable): each set's `size` is now its every position on the pair in the direction, M, its `local` its m on
   this cable, and `ranks` the index in `ranks[d]` of M words, the j-th the index among this cable's members of the
   set's j-th position on the pair in list order (UINT32_MAX: a position the deal gave another cable); `drawn[d][q]`
   room for the largest M the lane carries a position of; `place` gone.  Why: design/aot-compilation-change-request.md,
   appendix item 5: "The counter is (execution, set), so a set is drawn once per execution.  Every edge and every cable
   of the set's tree carries its share of the set's pages in that one permutation's order"; "A draw for each cable
   departs from the counter."  What must stay: every member's records this port's own; both ends drawing from the same
   key, counter and size.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", PM): `members[d]` each set's every
   member on the pair, M records from its `first`, a record of each whose own position the deal gave another cable
   among them (after every lane's records), and `position_index` a position's index among the set's M positions;
   struct flow_set's `local` and `ranks`, and `ranks[d]`, gone.  Why: appendix item 5, "Each port holds a record for
   every member of every set it carries a position of".  What must stay: both ends drawing from the same key, counter
   and size.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PQ): `deliveries` and `room`, a view of
   the plan's own delivered calls' queue in its state section (mesh.h struct mesh_state `deliveries`, `room`), read in
   place, and `at[d][q]`, the index in it of the entry lane q reads next way d (lane_take), 0 at the load: a plan's
   queue holds its every message from its first (its barrier, A7), each until every lane carrying it has taken it (the
   room's bound, appendix item 2).  Why: item 2, "Each plan's delivered calls reach the bridge at a node through a
   queue of the plan's own ... The bridge reads it in place, by view, and nothing copies it".  What must stay: the
   progress thread's alone.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): a flow is the plan at place `function` among the program's plans (mesh.h struct
   mesh_program `table` word k, function k + 1's: mesh_program_plan) of the program's root of generation
   `generation`, as its load read them (plan_stands), where it was the claim at head row `head` with its `owner` word
   and generation; `owner` gone.  Why: point 5, "These go: `mesh_claim_take`'s first fit and the claim's head record";
   the root's generation is minted once a program from hdr `serial` (mesh.h struct mesh_program), so (place,
   generation) names one plan of one program as (head, owner, generation) named one claim's.  What must stay: the
   progress thread's alone.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K1 and G4;
   "The revisions of 55b9cbad"): `generation` the operand's `published` (mesh.h struct mesh_algo) as its load read it;
   `next` gone, the port's flows held by place (struct port_plans `loaded`).  What must stay: the progress thread's
   alone.
   2026-11-06 (design/aot-compilation-change-request.md A7, corrected 2026-11-06: "A plan then stands while its view
   does, and its view while the sequencer's map does (D.3.6)"; "The revisions of 14b9a757", A7): FLOW_CANCELLED where
   its `serial` is no longer hdr `view`, the installed view's identifier (link_run; mesh-bridge.h plan_stale), where it
   was this node's `map_serial`.  What must stay: a cancelled flow served no more */
enum { FLOW_SERVING, FLOW_CANCELLED };
struct flow_set { uint32_t set,size; uint64_t first; };
struct flow {
  int state,ledger;
  uint32_t function,lanes,carrying[2],copies;
  uint64_t generation,send_count,receive_count,sent,landed,posted;
  struct mesh_state *words;
  struct prepared_send *sends;
  struct prepared_receive *receives;
  uint64_t *first[2];
  struct cursor cursor[2][MESH_LINK_LANES];
  struct mesh_trace *trace[2];
  size_t traced[2],trace_capacity[2];
  uint64_t key[4];
  struct flow_set *sets[2];
  /* 2026-11-06 ("The revisions of 14b9a757", the coordinating implementer, on sub-agent D1's finding): `position_set`
     and `position_index`, where they were `slot_set` and `slot_position`, and link_prepare's `record_of`, where it was
     `slot`: a position record's set and its index among the set's positions (Fisher and Yates's shuffle draws a
     permutation of a set's positions, Knuth vol. 2, §3.4.2, Algorithm P), and each pick's record.  Why: the operator's
     "Canonical names, not inherited idiom: no 'lease', 'claim' or 'slot' from old code without the canonical structure
     and citation".  What must stay: every use as before; a rename alone */
  uint32_t *position_set[2],*position_index[2];
  uint64_t *members[2];
  uint64_t *drawn[2][MESH_LINK_LANES],drawn_execution[2][MESH_LINK_LANES];
  uint32_t drawn_set[2][MESH_LINK_LANES];
  const struct mesh_delivery *deliveries;
  uint64_t room,at[2][MESH_LINK_LANES];
};
/* A port's plans, its progress thread's alone (design/compilation-coordination.md, Q5): those it holds, in the order
   it loaded them (`loaded`); hdr `plans` as this pass read it (`plans`: port_progress) and as its last walk of the
   heads read it (`walked`, 0 at the child's start: `plans` is 0 only where no plan was ever published, mesh.h struct
   hdr), and whether the next pass walks whatever it reads (`walk`: its plans ended, link_close); and whether a plan of
   it ended with requests outstanding on the link's queue pairs (`ended`), which only the child's exit releases
   (LINK_STREAMS_ENDED), so that nothing more is loaded or served in the child's life, as the one plan's FLOW_ENDED was
   (link_close's note, 2026-11-03).
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL3): `current`,
   for each direction d and lane q, the plan whose execution at its cursor there the lane has taken by the merge and
   is posting (lane_take; NULL where none is taken: the lane's next post begins with a merge).  What must stay: the
   progress thread's alone */
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH and M5): the port's copy of the
   deliveries' channel (mesh.h struct mesh_delivery), deliveries [`base`, `copied`) at `orders` (room for `room`), in
   its child's own memory, which grows as deliveries come (G. Kahn, IFIP 1974: an unbounded FIFO queue), entries before
   every served lane's place let go (deliveries_copy); `joined` once the child has read hdr `delivered` at its first
   pass; and `at`, for each direction d and lane q, the delivery its lane reads next (lane_take).  Why:
   design/aot-compilation-change-request.md at mesh 0e56d8da, appendix item 2, "Each port reads the channel in order".
   What must stay: the progress thread's alone.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", CH2), reversing M5: the copy (`orders`,
   `base`, `copied`, `room`) is gone; `at[d][q]` is the index of the delivery lane q reads next way d, read in place in
   the region's channel, entry `at` mod hdr `deliveries_room` (lane_take).  Why: design/aot-compilation-change-
   request.md, appendix item 2: "Each port reads it in place, by view, and nothing copies it.  Only the driver moves
   bytes"; "(M4 and M5 go)".  What must stay: the progress thread's alone */
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PQ): `at` and `joined` gone with the
   one channel for every plan, each lane's place now in each plan's own queue (struct flow `at`).  Why:
   design/aot-compilation-change-request.md, appendix item 2, "This replaces ... the one channel for every plan (the
   implementation's CH2, with M3's room)".  What must stay: the progress thread's alone.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K1: "the
   held-flow search (mesh-flow.c 1696): a flow table indexed by place"; "The revisions of 55b9cbad"), reversing
   `loaded` as a list in load order, which the walk searched for a flow of (place, generation): `loaded` a table of
   `places` flows indexed by place, entry k the port's flow of the plan at place k among the operand's `plans` (mesh.h
   struct mesh_algo), NULL where it holds none, grown to the operand's `plans` where a walk meets a place past it
   (plans_walk), in this child's memory.  One flow a place: a flow is held only while the operand of its generation is
   published, since no release frees an operand while a port holds a plan of it (mesh-dataflow.h
   mesh_retired_release), and the walk ends each flow that no longer stands before it loads any.  Why: K1, as
   MSCCL-IR's steps name their dependences by index (`depid`, `deps`); A9, "searches for nothing".  What must stay: the
   progress thread's alone.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-6:
   "Each cable's processor executes its list of instructions in the compile-time order, firing each on its dependency
   word (aot A9, "What the bridge runs").  It scans no table of plans"; W8: "The bridge's take and poll ... visit every
   place to `plans`, empty ones included.  The places with flows toward each adjacent node are fixed when the plan is
   made, so they are a list the plan writes"; "The revisions of 14b9a757", R-6): `generation`, the operand (mesh.h
   struct mesh_algo `published`) whose places toward this port's peer the walk read, 0 none; `adjacent`, those places
   (mesh.h mesh_algo_adjacent: the plans with an entry toward the peer's map index, ascending, in the operand's own
   pages, read in place); `listed`, their count while this port holds a flow at one of them, else 0; `holding`, the
   flows the table holds.  Every loaded flow is at one of them (plans_walk loads no other place), so the take, the
   poll, the busy test, the walk's ending pass and link_close visit those places alone, each by its table entry
   (`loaded`, the store by place).  The list is read only while it is the operand's: under the walk's announcement, and
   while a flow of the operand holds its `held` bit, which keeps the operand from its release (mesh-dataflow.c
   mesh_retired_release).  So a pass that ends flows reads no entry once `holding` is 0 (its last `held` bit cleared,
   the operand may be released and its pages written again), and `listed` is 0 once no flow is held.  What must stay:
   the progress thread's alone; no place read but the operand's own list, and none of it once no flow holds it */
static struct port_plans { struct flow **loaded; uint32_t places; uint64_t plans,walked; int walk,ended;
  struct flow *current[2][MESH_LINK_LANES]; uint64_t generation; const uint32_t *adjacent; uint32_t listed,holding; }
  port_plans[MESH_PORTS];

/* tag pre-return-20261103 rdma/mesh-link.c:56-65 port_halt (3f72cbde, 306b22be), verbatim but not static */
/* design/transport.md: a port halted from any thread (a peer interface's termination heard, a provider's fault
   absorbed; atomics alone): its pairing lost, so no job of it makes a further call (B10), then no lane of its link
   served, both sequentially consistent (link_turn's connect stores served and reads the pairing again), and its bit set
   in `which` for the event loop's conclusion (terminations_turn, ports_stalled) */
_Static_assert(MESH_PORTS<=32,"a bit a port");
void port_halt(struct port *port,_Atomic uint32_t *which){
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  atomic_fetch_or(which,UINT32_C(1)<<port->index);
}
/* tag pre-return-20261103 rdma/mesh-link.c:71-81 lane_lost (d556370e), verbatim */
/* a lane's failure, found by a progress thread: served no more, and the link lost by the event loop on reading
   `broken` (link_turn) */
static void lane_lost(struct port *port,uint32_t lane,int64_t code){
  struct link *l=&port->link;
  if(atomic_fetch_and_explicit(&l->served,~(UINT32_C(1)<<lane),memory_order_acq_rel)>>lane&1)
    mesh_say("bridge","link %u (%s): lane %u failed (code %lld): the link toward node %u lost",port->index,port->name,lane,
             (long long)code,port->peer);
  atomic_store_explicit(&l->lost,code,memory_order_relaxed);
  atomic_fetch_or_explicit(&l->broken,UINT32_C(1)<<lane,memory_order_release);
  wake();
}
/* tag pre-return-20261103 rdma/mesh-link.c:275-286 link_failed (d556370e), verbatim */
/* design/transport.md: the link's first failed completion, on any lane, or a failed poll: no lane of the
   link served from here, so nothing is posted on it again, and the link lost by the event loop on reading `broken`
   (link_turn), the stream's words ~0 (port_lost) */
static void link_failed(struct port *port,int64_t code){
  struct link *l=&port->link;
  const uint32_t served=atomic_exchange_explicit(&l->served,0,memory_order_acq_rel);
  if(served)mesh_say("bridge","link %u (%s): a completion or its poll failed (code %lld): no lane served, the link toward node %u lost",
                     port->index,port->name,(long long)code,port->peer);
  atomic_store_explicit(&l->lost,code,memory_order_relaxed);
  atomic_fetch_or_explicit(&l->broken,served,memory_order_release);
  wake();
}

/* tag pre-return-20261103 rdma/mesh-dataflow.c:79-103 mesh_records_read (e3c4c986), adapted: the region the bridge's
   own (M), the map's serial read inside the same reading, an error value in place of -errno, and its deadline the
   bridge's clock (main's mesh.h has no struct mesh_deadline); the bridge that writes the map is this child's parent, so
   a write that stays open past MESH_PLATFORM_NS is ETIMEDOUT, not the tag's ESRCH */
/* the map's records as one serial left them, the seqlock's reader (mesh_map_whole): a copy in *out the caller frees,
   its count and the installed view's identifier with it, or an error value; the read within its deadline,
   MESH_PLATFORM_NS from its start.
   2026-11-06 (design/aot-compilation-change-request.md A7, corrected 2026-11-06: "The map a view is created from is the
   sequencer's, and the view's first message carries it ... Every member plans on those records"; mesh.h hdr
   `view_map`: "written under `view_map`, a seqlock as `map` is"; "The revisions of 14b9a757", A7), reversing map_read,
   which read the map alone: records_read reads either seqlock's records whole, `lock` its sequence word, `nodes` its
   count, `off` its records' region offset, and the installed view's identifier read with them (mesh.h
   mesh_view_installed): the map (hdr `map`, `map_nodes`, `map_off`), whose cables link_prepare joins, or the
   installed view's (hdr `view_map`, `view_nodes`, `view_off`, and `view`, `view_sequencer`, `view_boot`: mesh-map.h
   mesh_view_records), whose places give a plan's map indices (plans_walk, streams_cancel).  The reader's test is
   mesh_map_whole's (an acquire fence, then the sequence word even and unchanged), written here over `lock` since
   mesh_map_whole reads hdr `map` alone; `before` is even where it is reached.  Why: "Refactor by pattern analysis:
   merge duplicated patterns", one reader of the two seqlocks.  What must stay: the reading whole or retried, within
   MESH_PLATFORM_NS; the reader's errors (ETIMEDOUT, EINVAL, ENOMEM) */
static int records_read(_Atomic uint64_t *lock,const _Atomic uint64_t *nodes,uint64_t off,struct mesh_record **out,
                        uint64_t *count,struct mesh_view_id *view){
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  for(;;){
    const uint64_t before=atomic_load_explicit(lock,memory_order_acquire),at=clock_ns();
    if(at>=until)return ETIMEDOUT;
    if(before&1){
      const uint64_t left=until-at;
      os_sync_wait_on_address_with_timeout(lock,before,sizeof *lock,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                           OS_CLOCK_MACH_ABSOLUTE_TIME,left<MESH_PAUSE_NS?left:MESH_PAUSE_NS);
      continue;
    }
    const uint64_t n=atomic_load_explicit(nodes,memory_order_relaxed);
    const int fits=off<=M->length && n<=(M->length-off)/sizeof(struct mesh_record);
    struct mesh_record *copy=fits?malloc((n?n:1)*sizeof *copy):NULL;
    if(copy)memcpy(copy,(char *)M+off,n*sizeof *copy);
    const struct mesh_view_id s=mesh_view_installed(M,memory_order_relaxed);
    if(mesh_seqlock_whole(lock,before)){
      if(!copy)return fits?ENOMEM:EINVAL;
      *out=copy;*count=n;*view=s;return 0;
    }
    free(copy);
  }
}
/* the installed view's records whole (records_read), `view` its identifier */
static int view_read(struct mesh_record **out,uint64_t *count,struct mesh_view_id *view){
  return records_read(&M->view_map,&M->view_nodes,M->view_off,out,count,view);
}
/* a node's index in the map: its place in ascending node-id order among the map's records (mesh.h hdr `map_serial`);
   UINT32_MAX where the map holds no record of it.
   2026-11-06 (design/aot-compilation-change-request.md A7, corrected 2026-11-06: "Every member of a view plans on the
   view's map"; mesh-collective.h: "The map planned on is the installed view's"; "The revisions of 14b9a757", A7): the
   records given are the installed view's (view_read), so the index is the one the plan's entries name (mesh.h struct
   mesh_hop `node`), where they were this node's own map's.  What must stay: ascending node-id order, as
   mesh-dataflow.c mesh_nodes_read orders the records the plan is made on */
static uint32_t map_index(const struct mesh_record *records,uint64_t count,uint32_t node){
  uint32_t index=0,held=0;
  for(uint64_t r=0;r<count;r++){index+=records[r].node<node;held|=records[r].node==node;}
  return held?index:UINT32_MAX;
}
static inline _Atomic uint64_t *region_word(uint64_t offset){ return (_Atomic uint64_t *)(void *)((char *)M+offset); }
/* 2026-11-04, sub-assignment 6: landed_cancel (2026-11-03: an execution's cancel word, its landed record's word 0,
   found from a word of the record and hdr `landed_stride`) is gone with the landed records and the cancel word (C4);
   a cancellation is count_raise's */
/* the executions t < `executions` with t mod u == `copy` (u `copies`): those a record of that copy serves */
static inline uint64_t copy_executions(uint64_t executions,uint32_t copies,uint32_t copy){
  return executions>copy?(executions-copy-1)/copies+1:0;
}
/* A cancellation's arrival (design/disjoint-pages-change-request.md, C3.1: "Landings and cancellations both arrive on
   the count, as arrivals do on an mbarrier"; design/disjoint-pages-revision-coordination.md, "The bridge"): the count
   word raised to `final` and never lowered, by a compare-and-swap (release) that stores only above what it read and is
   tried again only where another arrival moved the word between (so it ends within the arrivals still to come, a
   range's P an execution), then the word's waiters woken (os_sync_wake_by_address_all, 8 bytes, region-shared: the
   client's waiter's os_sync_wait_on_address_with_timeout, mesh-metal.m) where this call raised it.  A count at or past
   `final` is left as it is, so several records or entries of one range raise its word once.  No phase is written:
   the first execution the receive did not land reads its field one phase behind (mesh.h, "Phases") */
static void count_raise(_Atomic uint64_t *count,uint64_t final){
  uint64_t seen=atomic_load_explicit(count,memory_order_relaxed);
  while(seen<final && !atomic_compare_exchange_weak_explicit(count,&seen,final,memory_order_release,memory_order_relaxed)){}
  if(seen<final)os_sync_wake_by_address_all(count,sizeof *count,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* 2026-11-04, owner B1 (e6bd7db2's section 6: "A plan's pages are never rewritten while it stands, so the old
   re-checks for torn reads go"; design/compilation-coordination.md, Q5): plan_refused (EAGAIN where hdr `hops` or
   `plan` moved since a load read them, the region's one plan written again in place) is gone.  A plan's entries and
   state are in its own claim, written before it stands and never again, and a load reads its claim's head again after
   its announcement and before it reads them (plan_load, plan_stands).  What must stay: no refusal of a plan the load
   did not read as it stands */
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): plan_sequences_inside (whether a
   plan's sequence words lay inside the region, DL8) is gone with the words: design/aot-compilation-change-request.md at
   mesh 0e56d8da, appendix item 2, "in place of the sequence words (`sequences`), `taken[p]` and the parent's wait".
   What must stay: nothing of it */
/* whether a plan's entries lie inside the region (its state section's `hops` and `entries`, mesh.h mesh_hops), as the
   start judges them (mesh-call.h mesh_transfers_start: EINVAL where they lie outside it) and as map_read judges the
   map's records.  2026-11-04, owner B4: and its sequence words (plan_sequences_inside), which every port reads, so a
   plan whose words lie outside the region is refused (link_prepare, EINVAL) and names no count to cancel
   (plan_cancel), as one whose entries do.  What must stay: the entries' test as it was.
   2026-11-05 (CH): the sequence words' test gone with the words.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K2: "`plan_inside`
   and `deliveries>M->length` (mesh-flow.c 715, 719, 2053); link_prepare's room and entry tests"; "Answered in the
   documents", "Run-time checks left"; "The revisions of 55b9cbad"): plan_inside is removed, with its tests in
   link_prepare and plan_cancel.  Its producer, one hop up, is mesh-collective.c mesh_plan (sub-agent C's): a plan's
   entries are laid out in its state section (plan_layout, `hops`), placed in the region's pages by the one allocation,
   `entries` its template's entries times `copies` u.  What must stay: nothing of it */
/* whether the plan read at claim head `head` stands there still: the head's `owner` word and generation (its page
   entry's `device`) as read before, and a plan standing at it (mesh-dataflow.h mesh_head_state: `owner` not 0,
   `closed` 0 and its publication `state` not 0, each read with acquire) in the same state section.  It is D3's head
   read again (e6bd7db2's section 6, "A load", step 2, as design/compilation-coordination.md, Q5 takes it: the hazard
   pointer's announce-and-recheck, M. M. Michael, Hazard pointers: safe memory reclamation for lock-free objects, IEEE
   TPDS 15(6), 2004), and the test by which a walk ends a loaded plan whose claim closed or was reused (plans_walk).  A generation is minted once from hdr
   `serial` (mesh-dataflow.h mesh_claim_take), so a head a later claim reused reads another.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): whether the plan read at place `function` of the program's root of generation
   `generation` stands there still: the plan standing at that place (mesh.h mesh_program_plan: the root's `owner` not
   0 and `closed` 0, the place below its `plans`, the plan's `closed` 0, each read with acquire) the same state
   section, and the root's `generation`, read after them, the same, where a claim's head was read again.  It is the
   root read again after the announcement (plan_load; M. M. Michael, IEEE TPDS 15(6), 2004, as above), and the test by
   which a walk ends a loaded plan the program retired, alone (its `closed`) or with itself (the root's), or a
   released root's next program replaced.  Why: point 5, "Nothing walks heads to find it"; the root's generation is
   minted once a program from hdr `serial` (mesh.h struct mesh_program), so a place a later program's plan holds reads
   another.  What must stay: the plan read again after the announcement, before the load reads it.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4; "The
   revisions of 55b9cbad"): the operand's `published` (mesh.h struct mesh_algo, acquire) `generation`, then the plan
   standing at place `function` (mesh_algo_plan) the same state section, where the root's `generation` was read after
   the plan.  `published` is compared first, the implementer's (sub-agent D's) order: a place a load or a walk read of
   generation `generation` is then read only in that operand, which no release frees while this port announces it or
   holds a plan of it (mesh-dataflow.h mesh_retired_release), where read after it a place of a next operand with fewer
   plans would be read past its table (mesh_algo_plan tests no bound: K2).  Why: G4, the generation the ports announce
   is `published` (mesh.h struct mesh_port_info `prepared`); "The publication's readers": the ports read nothing of an
   operand before `published` (acquire).  What must stay: the operand read again after the announcement, before the
   load reads the plan.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-8: "One
   writer a word (W11; Lamport's single-writer registers): the program writes `retired`, the bridge writes only
   `released`, and a gone holder is read from the heartbeat's words"; R-3: "every field of the published operand is
   read through the acquired publication, never before it (McKenney and Slingwine)"; "The revisions of 14b9a757", R-8
   and R-3), reversing the forms above: plan_stands(generation) is the walk's one re-read of the operand after its
   announcement (plans_walk): `generation`, the operand it announced, where mesh_algo_open reads it again and its
   holder holds the region (mesh-holders.h mesh_algo_holder, the holder's one read after the publication's acquire,
   then mesh_holds), else 0.  A plan stands where its operand does so and mesh_algo_standing finds it at its place (the
   walk's ending pass, by its flow).  The bridge writes no `retired` for a gone holder (R-8), so a gone holder's
   operand stays open until its release, and this test is how a walk ends its plans; the bridge parent raises hdr
   `plans` once it sees the holder gone (mesh-records.c), so the ports walk.  Its cost: one proc_pidinfo
   (PROC_PIDTBSDINFO: mesh-holders.h mesh_started) a walk, on the port's progress thread, in plans_walk alone, which
   runs where hdr `plans` moved since the last walk or the port's plans ended (link_run); none on a page's path.  The
   holder read after mesh_algo_holder's own test is this operand's: mesh_algo_open reads `generation` again after it,
   and an operand once not open is never open again under its generation.  What must stay: the operand read again
   after the announcement, before any field of it; the holder read after the publication; no word written */
static uint64_t plan_stands(uint64_t generation){
  const struct mesh_algo_holder holder=mesh_algo_holder(M);
  return mesh_algo_open(M)==generation && mesh_holds(holder.pid,holder.start)?generation:0;
}
/* a lane's cursor at the next execution, on that execution's copy's records on the lane (`first` the lane's u + 1
   bounds, struct flow) */
static inline void cursor_step(struct cursor *at,const uint64_t *first,uint32_t copies){
  at->execution++;
  if(++at->copy==copies){at->copy=0;at->round++;}
  at->at=first[at->copy];at->end=first[at->copy+1];
}
/* 2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL4): flow_executions
   (the executions to serve, the lesser of the plan's `executions` and `released` as link_run read them, Q4, and the
   completions they asked each way) is removed with the batch bound it computed.  Why: no lane serves a plan to a bound
   now; it takes each execution by the merge (lane_take).  What must stay: copy_executions, which plan_end and
   plan_cancel count a copy's executions by */
/* Threefry-4x64-20 (J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, Parallel random numbers: as easy as 1, 2,
   3, SC 2011, section 4.2: "Threefry-N×W-R follows the same pattern of S and P boxes as Threefish, but with R rounds";
   "The key insertion schedule is the same as in Threefish, except that Threefry ignores the 'tweak'"; "For W = 64 and
   N ≥ 4, Threefry's rotation constants are taken from Threefish"; Table 2: its key and counter (4+4)×8 bytes; section
   5: "Without AES-NI, we recommend Threefry-4×64-20 on CPUs", and this runs on a port's progress thread), so
   Threefish-256 (N. Ferguson et al., The Skein hash function family, v1.3, 2010, section 3.3) with Nr = 20 and the
   tweak t_0 = t_1 = t_2 = 0 (design/compilation-coordination.md, "The appendix's revisions", S4): the plaintext p the
   counter, the key k_0 .. k_3; for d = 0 .. 19, e_{d,i} = v_{d,i} + k_{d/4,i} mod 2^64 where d mod 4 = 0, else
   v_{d,i}; (f_{d,2j}, f_{d,2j+1}) = MIX_{d,j}(e_{d,2j}, e_{d,2j+1}), section 3.3.1: y_0 = x_0 + x_1 mod 2^64, y_1 =
   (x_1 <<< R_{d mod 8, j}) xor y_0; v_{d+1,i} = f_{d,pi(i)}, pi = (0, 3, 2, 1) (Table 3, Nw = 4); the output c_i =
   v_{20,i} + k_{5,i} mod 2^64.  R_{d,j} for Nw = 4 (Table 4): d = 0: 14, 16; 1: 52, 57; 2: 23, 40; 3: 5, 37; 4: 25,
   33; 5: 46, 12; 6: 58, 22; 7: 32, 32.  The key schedule (section 3.3.2): k_4 = C240 xor k_0 xor k_1 xor k_2 xor k_3,
   C240 = 0x1BD11BDAA9FC1A22; k_{s,i} = k_{(s+i) mod 5} for i = 0, 1, 2 (their tweak terms t_{s mod 3} and
   t_{(s+1) mod 3} are 0), k_{s,3} = k_{(s+3) mod 5} + s mod 2^64.  Every rotation is by 5 to 58 bits, so neither shift
   is by 64 */
static void threefry4x64_20(const uint64_t key[4],const uint64_t counter[4],uint64_t out[4]){
  static const unsigned char rotation[8][2]={{14,16},{52,57},{23,40},{5,37},{25,33},{46,12},{58,22},{32,32}};
  const uint64_t k[5]={key[0],key[1],key[2],key[3],UINT64_C(0x1BD11BDAA9FC1A22)^key[0]^key[1]^key[2]^key[3]};
  uint64_t v[4]={counter[0],counter[1],counter[2],counter[3]};
  for(unsigned d=0;d<20;d++){
    if(d%4==0)for(unsigned i=0;i<4;i++)v[i]+=k[(d/4+i)%5]+(i==3?d/4:0);
    uint64_t f[4];
    for(unsigned j=0;j<2;j++){
      const uint64_t x0=v[2*j],x1=v[2*j+1],y0=x0+x1;
      const unsigned r=rotation[d%8][j];
      f[2*j]=y0;f[2*j+1]=((x1<<r)|(x1>>(64-r)))^y0;
    }
    v[0]=f[0];v[1]=f[3];v[2]=f[2];v[3]=f[1];
  }
  for(unsigned i=0;i<4;i++)out[i]=v[i]+k[(5+i)%5]+(i==3?5:0);
}
/* D. E. Knuth, The Art of Computer Programming vol. 2, section 3.4.2, Algorithm P, as design/compilation-
   coordination.md, "The appendix's revisions", S5 takes it ("P1. [Initialize.] Set j <- t.  P2. [Generate U.] Generate
   a random number U, uniformly distributed between zero and one.  P3. [Exchange.] Set k <- floor(jU) + 1. ... Exchange
   X_k <-> X_j.  P4. [Decrease j.] Decrease j by 1.  If j > 1, return to step P2."): t the set's size m on this cable
   (S7': "the set's members whose positions the deal gave that cable are permuted among those positions (S5, with t
   their count)"), X_1 .. X_m those members in list order (`members`, copied into `x`, x[i - 1] holding X_i); the s-th
   U (s = 0, 1, ... as j runs m, m - 1, ..., 2) the (execution, set, cable) stream's word s read as r / 2^64, so k =
   floor(j r / 2^64) + 1, the high word of the 128-bit product j r, plus 1 (integer arithmetic, so both ends compute
   the same; k is not a remainder modulo j).  The stream (S4; Salmon et al., section 5.1: "concatenate the object
   identifier, i, and time, T, into a p-bit counter"): word s is word s mod 4 of threefry4x64_20 of the key (the plan's
   name zero-padded, struct flow `key`) and the counter (execution, set, s / 4, place), `place` the cable's place in
   the pair's key order (S7'), so no (key, counter) serves two blocks of a stream, nor two cables of one pair (section
   5: "(key, counter) tuples must never be improperly re-used").  The counter names no node pair, so every edge of
   tree j (`set` j + 1) whose cable has the same place in its own pair's key order draws the same stream: one
   permutation along the tree where those edges' member counts agree (S2).  After it x[i - 1] is X_i, the member
   position i carries in that execution (S5).  2026-11-04, owner B3 (S7'): the counter's fourth word the cable's place,
   where it was 0, and t the set's members on this cable, where it was the set's every position on the pair.  What
   must stay: both ends drawing from the same key, counter and size.
   2026-11-04, owner B3 (reviews W1's finding 2 and W3's finding 7): this note's first S7' form said no (key, counter)
   serves two cables; that was false (the paragraph above), and what S7' rules out is two cables of one pair sharing a
   stream.  And a replan keeps its plan's name (N1: (group, function)) and starts its executions again at 0, so it
   draws the cancelled plan's streams again: open for the operator.  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", OP and LR), reversing S7′'s form
   above: t is the set's every position on the pair in the direction, M (`size`), X_1 .. X_M those positions' indices
   0 .. M - 1 in list order (x[i - 1] holding X_i), and the counter (execution, set, w / 4, 0), with no cable's place,
   so every cable of the pair, and every edge of the tree whose pair lists the set's M positions, draws one stream and
   one permutation (design/aot-compilation-change-request.md, appendix item 5: "The counter is (execution, set), so a
   set is drawn once per execution"; "A draw for each cable departs from the counter"; Salmon et al., section 5:
   "zero-pad a smaller input").  P3's k - 1 is the uniform integer in [0, j) of D. Lemire, Fast random integer
   generation in an interval, ACM Trans. Model. Comput. Simul. 29(1), 2019, section 4, Algorithm 5 (arXiv 1805.10941v4),
   with L = 64 and s = j: "x <- random integer in [0, 2^L); m <- x × s; l <- m mod 2^L; if l < s then t <- (2^L - s)
   mod s; while l < t do x <- random integer in [0, 2^L); m <- x × s; l <- m mod 2^L"; "return m ÷ 2^L", which "generates
   unbiased random integers in [0, s) for any integer s ∈ (0, 2^L)" (item 5: "With Lemire's rejection step, which the
   code must add, the integer is exactly uniform"), where the high word alone was taken.  Each x is the stream's next
   word w (w = 0, 1, ... over the whole draw), so a rejection takes words the next j's would have taken, alike at both
   ends; (2^64 - j) mod j is (0 - j) % j in uint64_t.  After it x[i - 1] is the index of the position whose member is
   the i-th of the permutation; position_record gives each cable its own members in that order.  What must stay: both
   ends drawing from the same key, counter and size.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", PM): x after the draw is σ, x[j] the
   index among the set's M members on the pair, in list order, of the member at its position j (the j-th of its M
   positions, in list order), where it was read the other way and restricted to this cable's members (M1).  Why:
   appendix item 5, "In each execution, the member at each position is the one σ puts there".  What must stay: both
   ends drawing from the same key, counter and size */
struct set_stream { const uint64_t *key; uint64_t execution,word,block[4]; uint32_t set; };
static uint64_t set_word(struct set_stream *s){
  if(s->word%4==0){const uint64_t counter[4]={s->execution,s->set,s->word/4,0};threefry4x64_20(s->key,counter,s->block);}
  return s->block[s->word++%4];
}
static void set_draw(const uint64_t key[4],uint64_t execution,uint32_t set,uint32_t size,uint64_t *x){
  for(uint32_t i=0;i<size;i++)x[i]=i;
  struct set_stream stream={.key=key,.execution=execution,.set=set};
  for(uint64_t j=size;j>1;j--){
    unsigned __int128 m=(unsigned __int128)set_word(&stream)*j;
    uint64_t l=(uint64_t)m;
    if(l<j){
      const uint64_t t=(0-j)%j;
      while(l<t){m=(unsigned __int128)set_word(&stream)*j;l=(uint64_t)m;}
    }
    const uint64_t k=(uint64_t)(m>>64)+1;
    const uint64_t held=x[k-1];x[k-1]=x[j-1];x[j-1]=held;
  }
}
/* the record a lane's post takes at its cursor `at`, direction d, lane q (design/compilation-coordination.md, "The
   appendix's revisions", S7: "A member's record (its page, its words, its bytes) is the one link_prepare prepares,
   posted at the position its execution's permutation gives it, on that position's line"): a sequence's position its
   own record; position i of a set member X_i's, of the set's permutation in the cursor's execution, drawn once a lane
   where its cursor meets the set in an execution other than the one drawn last (set_draw into the lane's room, struct
   flow `drawn`; S7: "Each lane computes an (execution, set)'s permutation once, into room taken at the load").  A
   lane's positions of one set may lie between its positions of other sets or of a sequence (a set's pages keep their
   places in the list: design/plan-to-bridge-change-request.md R5), so a lane whose cursor meets a set again after
   another in one execution draws it again, the same permutation (set_draw a function of the key, the counter, the
   members and the size alone).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): this note said a lane's positions
   of one set are consecutive among its records of a copy, so the lane met each set once an execution.  Why: R5, "A
   set's pages ... keep their places in that order.  The set's random permutation is drawn over those places."  What
   must stay: the member each position carries in an execution the same at every draw, at both ends.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", OP): the set's one permutation of its
   M positions on the pair (set_draw), then this cable's members in its order: x walked from its first, each index
   whose position the deal gave this cable (`ranks`, its index among this cable's members) taken in turn, so position
   r of this cable (its r-th member in list order, `position_index`) carries the r-th of this cable's members in the
   permutation; x is compacted in place, each index read before its word is written (r <= i).  Where it was the
   permutation of this cable's members alone, drawn on a stream of its own.  Why: appendix item 5, "Every edge and
   every cable of the set's tree carries its share of the set's pages in that one permutation's order"; the share is
   the members the deal gave the cable, whose records this port's child holds (design/compilation-coordination.md,
   "The revisions of 46ae8fdf", M1).  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", PM), reversing M1's share: position j
   of the set on the pair (this cable's position record's `position_index`) posts member x[j] of σ, its record among
   the set's every member (`members`), whichever cable the deal gave that member's own position.  Why: appendix item 5,
   "So the deal fixes positions, not members ... So any member can be posted at any of the set's positions on its
   cables."  What must stay: the member each position carries in an execution the same at every draw, at both ends */
static inline uint64_t position_record(struct flow *f,int d,uint32_t q,const struct cursor *at){
  const uint32_t s=f->position_set[d][at->at];
  if(s==UINT32_MAX)return at->at;
  const struct flow_set *set=f->sets[d]+s;
  if(f->drawn_set[d][q]!=s || f->drawn_execution[d][q]!=at->execution){
    set_draw(f->key,at->execution,set->set,set->size,f->drawn[d][q]);
    f->drawn_set[d][q]=s;f->drawn_execution[d][q]=at->execution;
  }
  return f->members[d][set->first+f->drawn[d][q][f->position_index[d][at->at]]];
}
/* every table of a flow's sets freed (link_prepare's failure, plan_end) */
static void flow_sets_free(struct flow *f){
  for(int d=0;d<2;d++){
    free(f->sets[d]);free(f->position_set[d]);free(f->position_index[d]);free(f->members[d]);
    f->sets[d]=NULL;f->position_set[d]=f->position_index[d]=NULL;f->members[d]=NULL;
    for(uint32_t q=0;q<MESH_LINK_LANES;q++){free(f->drawn[d][q]);f->drawn[d][q]=NULL;}
  }
}

/* The rule both ends compute (design/page-queue-change-request.md; mesh.h struct mesh_hop, "Both ends"), and this
   port's share of a standing plan built from it, into `f`, from the plan's state section `words` (mesh-dataflow.h
   mesh_head_state), which plan_load has announced and found standing.  The plan is refused (ESTALE) where its `serial`
   is not the map's serial as one whole reading of the map left it.  The cables between this node and
   the port's peer are the reach entries both records list for each other (each from its own side, the ports crossed),
   one cable an entry of each (mesh-map.h mesh_cables, the one per-cable join); a cable's key is (the lower node id's
   port index, the higher's), cables in key order; its bandwidth the lesser of the two records' and its queue pairs the
   lesser of the two (the pairing's, the same at both ends).  The pair's links are each cable's queue pairs
   0 .. qps - 1 in (key, queue pair) order, each of rate the cable's bandwidth over its queue pairs.  The plan's list
   is its template, u copies of one execution's n hops (its state's `entries` = u n at its `hops`, copy c's at [c n,
   c n + n), mesh.h mesh_hops; u its `copies`: design/disjoint-pages-change-request.md, C3.3), each cable's sends in
   the schedule's order (design/plan-to-bridge-change-request.md R5; design/compilation-coordination.md, T6).  Each
   direction's hops of the pair (the template's entries naming the peer's index, in list order: copy 0's, then copy
   1's, ...) are its units in that order, unit k its hop's `bytes` long (T7: mesh-graph.h's "unit k, bytes[k] long"),
   assigned by mesh_graph_list_schedule (Graham's list scheduling, mesh-graph.h), as R5 leaves it to the bridge ("How
   the bridge spreads a cable's sequence over what its devices give it is the bridge's own, as now"): this node's
   SENDs to the peer are the peer's receives from this node, and
   u is the collective's (design/disjoint-pages-change-request.md, C3.3; design/compilation-coordination.md, "The
   appendix's revisions", U1: every rank computes the same template, lags and u), so both ends assign one list alike
   (2026-11-04, owner B3, review W3's finding 8: this cited the coordination's D5, which U1 withdraws; what must stay:
   one u at every rank).  Kept:
   the units on this port's cable, each on its queue pair (lane q, link_plan), one record each, by lane, copy and
   assignment order (struct flow `first`).  A lane serves the executions in order, execution t's records on it copy
   t mod u's in assignment order (struct cursor), so in every execution the k-th SEND on a queue pair fills the k-th
   receive posted on it (2026-11-04, owner B4, DL3: each plan's executions so, the plans a lane carries merged by
   sequence number, lane_take, alike at both ends).  A record is posted once for each execution of its copy, and a receive record again only once
   its last posting has landed (link_receive_drain: its round j, j = t / u, only once its `landings` reads j, the
   lane's posting stopping at the first record that has not), so a page is named by at most one posted receive at any
   moment and enters its list again only once its last landing has landed (design/disjoint-pages-change-request.md,
   D.2, "How many pages are in flight at once", and D.3.5; design/disjoint-pages-rulings-coordination.md, R1).  A queue
   pair's receives complete in the order posted (Hilland et al., RDMA Protocol Verbs Specification, section 8.2.2:
   design/prepared-machine.md#M08), so a receive record's j-th completion since the load is execution copy + j u
   (`landings`).  A posting after the last landing is in time: the hardware holds a SEND until its receive is posted
   (the data model, section 3), and the peer's SEND of execution t + u follows this node's reads of t, which follow t's
   landing, by the allocator's order (C3.3), not by anything here.  A SEND sends its page once released
   (link_send_drain); a relay is no special case, its SEND naming its receive's word and field.  A receive lands in its
   page and writes its phase and its count alone (link_receive_drain).  No verbs call: the progress thread posts as it
   serves (link_receive_progress, link_send_progress).  0, or an error value: ESTALE, ETIMEDOUT or ENOMEM reading the
   map, EINVAL where the plan's copies are 0 or do not divide its entries, its entries lie outside the region
   (plan_inside), or an entry's `field`, `pages` or `bytes` is none a plan writes (mesh.h struct mesh_hop: a field's
   low bit even and below 64, the pages of a receive with a count at least 1, 1 to block x pgsz bytes), EALREADY where
   this port carries units of a plan whose run has begun (its `executions` above 0 at the load: the note at the
   refusal, below) or whose `taken[p]` is above 0 at the load (DL10, 2026-11-05: the same note), EHOSTUNREACH where the pair has units but no link of rate, EPROTO where the map gives this cable a
   queue pair its link does not hold, EMSGSIZE where a hop is more frames than a lane it is assigned can hold.
   2026-11-04, sub-assignment 6 (design/disjoint-pages-change-request.md, C1, C3.3, C4): reversing 2026-11-03's list of
   every execution's entries (executions x n, hdr `executions`), one record an entry, each posted once, so the records
   and the load's work grew with the run's length; now one record a template entry, posted once for each execution of
   its copy, so they are the template's whatever the run's length.  What must stay: both ends pick and assign the same
   u x n list in list order; each lane's executions in order; a record's request unchanged once written here.
   2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, D.3.5 and D.4; design/disjoint-pages-rulings-
   coordination.md, R1): reversing sub-assignment 6's note here, that a record was posted again while an earlier
   posting of it might be outstanding, relying on the provider taking a request at its post (its tbt_post_recv read
   from the disassembly, never documented), so that two postings could name one page.  Why: nothing required it; it
   gained nothing over the u rounds the copies give, and named one page in two transfers at once (D.2).  What must
   stay: the k-th SEND on a queue pair fills the k-th receive posted on it (both ends' one assignment, in list order);
   each lane's executions in order; no request taken back but by the child's exit.
   2026-11-04, owner B1 (design/plan-to-bridge-change-request.md R5; design/compilation-coordination.md, T7): each
   unit's length its hop's `bytes` (struct mesh_hop), the operand mesh_graph_list_schedule states ("unit k, bytes[k]
   long"), and each record's span that long, where every unit and span was a whole page (page_bytes()); a hop's frames
   past a lane's room refused per hop (EMSGSIZE).  The plan read from its own state section (`words`: its `entries` at
   its `hops`, its `serial`; mesh.h mesh_hops), where it was the region's one plan (hdr `hops`, `plan` and mesh_hops(M),
   now vacant), and nothing in it rewritten while it stands, so the torn-read refusal (EAGAIN, plan_refused) and the
   announcement here go to plan_load (e6bd7db2's section 6, "A load").  What must stay: both ends pick and assign the
   same u x n list in list order; each lane's executions in order; a record's request unchanged once written here.
   2026-11-03, sub-assignment 6 (design/disjoint-pages-change-request.md, "Claude's proposal" 1 and 3; Appendix A, A4):
   the ring went from the bridge: plan_shape (hdr `depth`, a slot's hops, execution t's page the slot t mod depth's
   and its word the cell at word + t hop_stride), the receive window (a receive of t posted once its hop's executions
   0 .. t - window had landed: struct flow_hop, landed_through, `need`) and the coordinator's forwards gate of the same
   day (a relay page's receive of t posted, from t = depth, once the page's forwarding SENDs of t - depth had completed,
   each completion adding 1 at offset 8 of its landed word's cell: `sent`, `forwarding`), with the per-page cells both
   read.  Why: they reused `depth` copies of a page under a wait before each execution, which nothing asked for; the
   plan's one allocator gives each execution's values their own pages, sharing a page only where the plan's dependence
   graph orders the last read (a consumption, or a relay's forward) before the next write (a landing), so a receive
   needs no landing or forward of another execution before it is posted.  What must stay: the k-th SEND on a queue pair
   fills the k-th receive posted on it (both ends' one assignment, in list order); a landing is never overwritten while
   it is read, by the allocator's order, not a count here; and no request is taken back but by the child's exit
   (link_close).
   2026-11-04, owner B3 (design/aot-compilation-change-request.md, appendix item 5; design/compilation-coordination.md,
   "The appendix's revisions", S3 to S7), where every unit was one of a sequence posted at its own position: each
   direction's sets found among its picks (a maximal run of consecutive picks within one copy with the same nonzero
   `set`; 2026-11-05, "The revisions of 67ca86f8", SP: the picks within one copy with the same nonzero `set`, wherever
   they lie), refused EINVAL where a nonzero `set` appears in two runs of one copy and direction (2026-11-05, SP: no
   longer), EMSGSIZE where a member is more frames than a line of one of its set's positions holds; the deal as it
   stands, over the positions in list
   order, each its own hop's bytes (S6, O3); and the tables each execution's permutation needs, with each lane's room
   for it, built once (struct flow; S7).  A set's position i on this cable then posts member X_i of its execution's
   permutation of the set's members on this cable (S7'), at both ends alike: the same key (the plan's name), counter
   (execution, `set`, block, the cable's place in key order) and size (S4, S5, S7'), the same positions on the same
   lines (the one deal), so the k-th SEND on a queue pair still fills the k-th receive.  Why: each execution of a set
   is carried in a uniformly random permutation of its pages (item 5).  What must stay: as above; a sequence's units
   prepared and posted as before.
   2026-11-04, owner B3 (design/compilation-coordination.md, S7', the coordinator's ruling on B3's finding): the
   permutation per cable, among the positions the deal gave it, and the ENOTSUP refusal of a set with positions on
   this cable and another of the pair gone (the note at the sets' test).  What must stay: each record's `landings`,
   round test and phase flip its own.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", OP): the permutation is the set's one
   over its every position on the pair, drawn per (execution, set), each cable posting its own members in that
   permutation's order (set_draw, position_record), where it was a permutation of this cable's members on a stream of
   this cable's own (S7′).  Why: design/aot-compilation-change-request.md, appendix item 5, "One permutation of the
   whole set per execution"; "A draw for each cable departs from the counter."  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", SW): no EALREADY; the refusal of a
   load after the run began is gone (its note, below).  What must stay: the other refusals.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K2:
   "`plan_inside` and `deliveries>M->length` (mesh-flow.c 715, 719, 2053); link_prepare's room and entry tests";
   "Answered in the documents", "Run-time checks left": "mesh-flow.c link_prepare's room and entry tests check facts
   the plan fixes at compile time, which the compile-time requirement forbids.  They go in the next unit"; "The
   revisions of 55b9cbad"): the EINVALs go, each a test of a fact mesh_plan fixes, its producer one hop up
   mesh-collective.c mesh_plan (sub-agent C's): `copies` u dividing `entries` (it writes `entries` as the template's
   times u), the entries and the deliveries' queue inside the region (plan_layout lays both in the plan's state
   section, placed in the region's pages), `room` not 0 (plan_graph's ENOMEM where a lag is 0), an entry's `bytes` 1 to
   block x pgsz (min(block x pgsz, the range's bytes left), the range's bytes above its page's start), a field below 63
   and even (2 x (field mod 32)) or MESH_ABSENT for a send of a supplied word, a receive's `count` its range's count
   word and its `pages` the pages this node receives of the range (at least 1).  Kept, each run-time state or a
   device's report: the map read and its serial (ESTALE, the reader's errors), a cable's queue pairs past the lanes a
   link has (EPROTO, the map's and the provider's), no link of rate (EHOSTUNREACH, the map's cables), a hop past a
   lane's room (EMSGSIZE, the device's queue), and ENOMEM.  What must stay: the kept refusals.
   2026-11-06 (design/aot-compilation-change-request.md A7, corrected 2026-11-06: "Every member of a view plans on the
   view's map ... A plan then stands while its view does"; design/compilation-coordination.md, "The root cause of the
   copies, and the revised work list", R-6; "The revisions of 14b9a757", A7 and R-6): the peer's map index is
   `index`, its place in the installed view's records as the walk read them once for the operand (plans_walk), where it
   was its place in this node's own map; and the ESTALE test of the plan's `serial` is the walk's, made once for the
   operand before that index is used (every plan of an operand is made on one view: mesh.h struct mesh_algo `nodes`,
   "the map the operand was planned on"), where it was made here for each plan against `map_serial`.  The test where
   the index is UINT32_MAX goes from the picks: the walk loads no plan where the view holds no record of the peer.  The
   cables of the pair stay this node's own map's (mesh_cables), read whole (records_read): transport is the bridge's
   alone.  Kept: the map reader's errors (ETIMEDOUT, EINVAL, ENOMEM), EPROTO, EHOSTUNREACH, EMSGSIZE, ENOMEM.  What
   must stay: both ends pick and assign the same u x n list in list order, the view's indices at both ends being one
   view's (A7) */
static int link_prepare(struct port *port,struct flow *f,struct mesh_state *words,uint32_t index){
  struct link *l=&port->link;
  const uint32_t copies=words->copies;
  const uint64_t entries=words->entries;
  f->deliveries=(const struct mesh_delivery *)(const void *)((const char *)M+words->deliveries);f->room=words->room;
  const uint64_t n=entries/copies,width=(uint64_t)copies+1;
  const struct mesh_hop *list=mesh_hops(M,words);
  struct mesh_record *records=NULL;uint64_t count=0;
  struct mesh_view_id view;
  int error=records_read(&M->map,&M->map_nodes,atomic_load_explicit(&M->map_off,memory_order_relaxed),&records,&count,
                         &view);
  if(error)return error;
  const uint32_t self=M->node,peer=port->peer;
  const struct mesh_record *mine=NULL,*theirs=NULL;
  for(uint64_t r=0;r<count;r++){if(records[r].node==self)mine=records+r;if(records[r].node==peer)theirs=records+r;}
  /* the pair's cables, the one per-cable join (mesh-map.h mesh_cables, which mesh_joined and mesh_nodes_read read too:
     C9), in key order; this port's the one whose key names it at this node's end */
  struct mesh_cable cables[MESH_PORTS];
  const uint32_t cable_count=mine && theirs?mesh_cables(mine,theirs,cables):0,end=self<peer?0:1;
  free(records);
  double rate[MESH_PORTS*MESH_LINK_LANES];
  uint32_t on[MESH_PORTS*MESH_LINK_LANES],queue[MESH_PORTS*MESH_LINK_LANES],links=0;
  for(uint32_t c=0;c<cable_count;c++){
    if(cables[c].qps>MESH_LINK_LANES)return EPROTO;
    for(uint32_t q=0;q<cables[c].qps;q++){
      rate[links]=(double)cables[c].bandwidth/(double)cables[c].qps;on[links]=c;queue[links]=q;links++;
    }
  }
  /* each direction's hops of the pair, its units: the template's entries naming the peer, picked once over the whole
     list, in list order (copy order) */
  uint32_t *pick[2]={NULL,NULL};uint64_t units[2]={0,0};
  uint64_t *bytes=NULL,*carried=NULL;uint32_t *assigned[2]={NULL,NULL};
  /* each direction's sets (S7, below; 2026-11-05, "The revisions of 67ca86f8", SP): set r's picks
     member[d][bound[d][r] .. bound[d][r + 1]) in list order, `runs` of them, `local` of them with positions on this
     cable; each pick's set (`set_of`, UINT64_MAX for a sequence's), each set's first pick while they are found
     (`lead`); and each pick's record (`record_of`), where the direction has such a set */
  uint64_t *member[2]={NULL,NULL},*bound[2]={NULL,NULL},*set_of[2]={NULL,NULL},*lead[2]={NULL,NULL},runs[2]={0,0},
           local[2]={0,0},*record_of[2]={NULL,NULL},*kept[2]={NULL,NULL},extra[2]={0,0};
  for(int d=0;d<2;d++){
    if(!(pick[d]=malloc((entries?entries:1)*sizeof *pick[d]))){error=ENOMEM;goto failed;}
    for(uint64_t j=0;j<entries;j++)
      if(list[j].node==index && list[j].direction==(uint32_t)d)pick[d][units[d]++]=(uint32_t)j;
  }
  const uint64_t most=units[0]>units[1]?units[0]:units[1];
  if(most){
    bytes=malloc(most*sizeof *bytes);carried=malloc((links?links:1)*sizeof *carried);
    if(!bytes || !carried){error=ENOMEM;goto failed;}
  }
  /* the assignment, and this cable's units on each of its queue pairs: unit k of a direction its hop's `bytes` long
     (T7), 1 to a page's bytes, none a plan writes else (EINVAL); and a hop a lane's queue cannot hold refused, not left
     unposted: its frames past the lane's receive room (lane_reset's frame_capacity) or past its SEND gate's
     (link_connect), EMSGSIZE.  The bytes' EINVAL is design/compilation-coordination.md, B1-1 ("a hop whose `bytes`
     is 0 or past a page is EINVAL").  2026-11-04, owner B1 (T7): every unit a page (page_bytes()) and the frames test
     one page's for every lane before; what must stay: no hop assigned to a lane that cannot hold it.
     2026-11-05 ("Review of 4e78f6f4 against the requirement lines", K2; "The revisions of 55b9cbad"): the bytes'
     EINVAL goes, a hop's `bytes` 1 to block x pgsz being mesh_plan's (link_prepare's note); EMSGSIZE stays, a lane's
     room the device's.  What must stay: no hop assigned to a lane that cannot hold it */
  uint64_t lane_units[2][MESH_LINK_LANES]={{0}};
  for(int d=0;d<2;d++){
    if(!units[d])continue;
    for(uint64_t k=0;k<units[d];k++)bytes[k]=list[pick[d][k]].bytes;
    if(!(assigned[d]=malloc(units[d]*sizeof *assigned[d]))){error=ENOMEM;goto failed;}
    if(isinf(mesh_graph_list_schedule(links,rate,units[d],bytes,assigned[d],carried))){error=EHOSTUNREACH;goto failed;}
    for(uint64_t k=0;k<units[d];k++){
      const uint32_t a=assigned[d][k];
      if(cables[on[a]].key[end]!=port->index){assigned[d][k]=UINT32_MAX;continue;}
      const uint32_t q=queue[a];
      if(!(l->plan>>q&1) || !l->provider.queues || (int)q>=l->provider.count || !l->provider.queues[q].pair){error=EPROTO;goto failed;}
      const uint64_t frames=frames_of(bytes[k]).n;
      if(d==MESH_RECEIVE?frames>l->lane[q].frame_capacity:frames>l->gate[q].capacity.n){error=EMSGSIZE;goto failed;}
      assigned[d][k]=q;lane_units[d][q]++;
    }
  }
  /* design/compilation-coordination.md, "The appendix's revisions", S3, S6 and S7 with S7' ("Amendments during the
     work"): each direction's sets, a set the picks within one copy with the same nonzero `set` (struct mesh_hop),
     wherever they lie in the list, its positions those picks in list order, each dealt above by its own hop's bytes
     (S6, O3: the deal as it stands).  The permutation is per cable (S7': "on each cable of
     the pair, the set's members whose positions the deal gave that cable are permuted among those positions (S5, with
     t their count), each execution, at both ends"): this cable's members of a set are its positions here, the deal's,
     fixed at the load; EMSGSIZE where one of them is more frames than the line of one of those positions holds (S7),
     the line's room the per-hop test's above.
     2026-11-04, owner B3 (S7', the coordinator's ruling on B3's finding), reversing this note's first form, in which a
     set with positions on this cable and on another cable of the pair was refused (ENOTSUP, not the contract's), since
     a member a permutation over the whole pair placed on another cable had no record there, or one whose `landings`
     no longer counted its executions.  Now each port's child holds records for its own cable's members only, as
     before, each posted once an execution of its copy, so each record's `landings`, round test and phase flip stay its
     own.  What must stay: no member permuted onto a cable that holds no record of it.
     2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): reversing a set as a maximal run
     of consecutive picks (S3) and the EINVAL where one `set` appeared in two runs of one copy and direction (S7).  Why:
     design/plan-to-bridge-change-request.md R5, "A set's pages ... keep their places in that order.  The set's random
     permutation is drawn over those places.  (Corrected 2026-11-05: the first text moved a set's pages to consecutive
     places, which nothing requires.)"  What must stay: a set's members in list order; the permutation per cable, over
     the positions the deal gave it (S7').
     2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", OP): the members each cable permutes
     are still the positions the deal gave it, and the frames test over them stands; their order is now the whole set's
     one permutation (set_draw), not one of their own.  What must stay: a set's members in list order; no member posted
     on a cable whose child holds no record of it */
  for(int d=0;d<2;d++){
    if(!units[d])continue;
    set_of[d]=malloc(units[d]*sizeof *set_of[d]);lead[d]=malloc(units[d]*sizeof *lead[d]);
    bound[d]=calloc(units[d]+1,sizeof *bound[d]);member[d]=malloc(units[d]*sizeof *member[d]);
    if(!set_of[d] || !lead[d] || !bound[d] || !member[d]){error=ENOMEM;goto failed;}
    /* each pick's set, the sets of a copy those of its picks (`from` its first), each set's picks counted at
       bound[d][r + 1]; then the bounds summed, and each set's picks laid out in list order, `lead` its next place */
    uint64_t from=0,copy=UINT64_MAX;
    for(uint64_t k=0;k<units[d];k++){
      const uint32_t set=list[pick[d][k]].set;
      set_of[d][k]=UINT64_MAX;
      if(!set)continue;
      if(pick[d][k]/n!=copy){copy=pick[d][k]/n;from=runs[d];}
      uint64_t r=from;
      while(r<runs[d] && list[pick[d][lead[d][r]]].set!=set)r++;
      if(r==runs[d])lead[d][runs[d]++]=k;
      set_of[d][k]=r;bound[d][r+1]++;
    }
    for(uint64_t r=0;r<runs[d];r++){bound[d][r+1]+=bound[d][r];lead[d][r]=bound[d][r];}
    for(uint64_t k=0;k<units[d];k++)if(set_of[d][k]!=UINT64_MAX)member[d][lead[d][set_of[d][k]]++]=k;
    /* 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", PM): the frames over the set's
       every member on the pair, where they were over its members on this cable, and `kept` each set's positions on this
       cable, `extra` the members whose positions the deal gave another cable of a set with a position here.  Why:
       design/aot-compilation-change-request.md, appendix item 5, "Each port holds a record for every member of every
       set it carries a position of ... So any member can be posted at any of the set's positions on its cables"; a
       member posted at a position of this cable goes on that position's line.  What must stay: no member assigned to a
       line that cannot hold it (EMSGSIZE) */
    if(!(kept[d]=calloc(runs[d]+1,sizeof *kept[d]))){error=ENOMEM;goto failed;}
    for(uint64_t r=0;r<runs[d];r++){
      uint64_t frames=0;
      for(uint64_t i=bound[d][r];i<bound[d][r+1];i++){
        const uint64_t k=member[d][i];
        kept[d][r]+=assigned[d][k]!=UINT32_MAX;
        const uint64_t each=frames_of(list[pick[d][k]].bytes).n;
        if(each>frames)frames=each;
      }
      if(!kept[d][r])continue;
      for(uint64_t i=bound[d][r];i<bound[d][r+1];i++){
        const uint32_t q=assigned[d][member[d][i]];
        if(q==UINT32_MAX)continue;
        if(d==MESH_RECEIVE?frames>l->lane[q].frame_capacity:frames>l->gate[q].capacity.n){error=EMSGSIZE;goto failed;}
      }
      local[d]++;
      extra[d]+=bound[d][r+1]-bound[d][r]-kept[d][r];
    }
    if(local[d] && !(record_of[d]=malloc(units[d]*sizeof *record_of[d]))){error=ENOMEM;goto failed;}
  }
  /* 2026-11-04, the coordinator's ruling on sub-assignment 6's finding 2 (a re-paired child loading the plan again):
     a plan whose run has begun, its state's `executions` (acquire) above 0 at this load, is refused (EALREADY, by
     link_run's refusal: `code`, phase MESH_STOPPED, `prepared` the refused generation), where this port carries units
     of it.  Why: a record's executions are counted from the load (`landings`, the cursors at execution 0), and a load
     after the run began cannot know which executions landed or were sent before it; mesh-dataflow.c
     mesh_transfers_start raises `executions` only once every port its hops name has loaded the plan, so a load that
     reads above 0 is a late one (a child re-paired on the same plan, or a port loading after the start).  Such a port
     posts nothing of the plan, so the next plan finds no request of it outstanding on its queue pairs, and its
     `prepared` is cleared by link_close once `hops`, `plan` or the client moves, so the release of the plan's claim
     is held no longer than the plan stands.  Where the port carries no unit of the plan it is loaded as before
     (differs from the ruling's text, which refuses every such load): it posts nothing either way, and a refused port
     whose cable reaches a node a hop names would fail the next start's wait (STOPPED) with nothing to serve.  A read
     torn by a plan written again is EAGAIN, not a refusal (plan_refused).  What must stay: no record served from a
     load after its run began.
     2026-11-04, owner B1 (design/compilation-coordination.md, Q5: "ESTALE or EALREADY as now"): the refusal is the
     plan's `refused` word (plan_load), where it was the port's `code`, phase and `prepared`, and holds nothing (no
     `held` bit, `prepared` 0 once the load ends), where `prepared` kept the refused generation until the plan moved;
     the start reads the code from `refused` (mesh-call.h).  What must stay: as above.
     2026-11-05, owner B4, the coordinator's ruling DL10 (design/compilation-coordination.md, "The documents' revisions
     of edd4b583", amendments): a plan this port carries units of is refused (EALREADY, the same refusal, the plan's
     `refused[p]` by plan_load) also where its `taken[p]` (acquire) is above 0 at the load, beside its `executions`.
     Why: a load after this port has taken numbers, or been told past them (plan_end raises `taken[p]` to `released`
     as the port ends a plan, a halt among the causes), cannot know them, since the parent may since have overwritten
     their words (Lamport's sender reuses a word once every port's `taken[p]` is past it: mesh-records.c
     orders_deliver), as a load after the run began cannot know which executions landed (the refusal's own reason,
     above); the parent's wait leaves a refused port out (DL2).  What must stay: no record served from a load after
     this port's numbers of the plan may have been reused */
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): DL10's test of `taken[p]` gone
     with the word (appendix item 2 at mesh 0e56d8da); the refusal where the run has begun stays.
     2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", SW): that refusal is gone too (a plan
     this port carries units of refused, EALREADY, where its `executions` read above 0 at the load), with the count of
     the units it read.  Why: design/aot-compilation-change-request.md, appendix item 2, "The code must: ... remove the
     start's wait for every port to load the plan, and the late-loading port's refusal of it (`mesh_ports_held`).
     Nothing requires them.  "a send that arrives first is held until its receive is posted (§3), and a loss cancels
     the plan anyway""; the start no longer waits for this load (mesh-dataflow.c mesh_transfers_start), so a load after
     the run's first submission is a load like any other: this port has posted nothing of the plan before it, and its
     cursors start at execution 0 (struct cursor), and each receive's round rule reads its page's phase field (PM,
     link_receive_drain), so what earlier executions landed is read where it is.  A child that pairs again after a loss
     finds the plan's `serial` no longer the map's, the loss having moved
     the map's serial (mesh-records.c map_put), and refuses it, ESTALE (above).  What must stay: the ESTALE refusal.
     2026-11-06 (design/aot-compilation-change-request.md A7, corrected 2026-11-06; mesh-bridge.h plan_stale: "a move
     of this node's own map ends no plan (its link losses cancel their receives as before)"; "The revisions of
     14b9a757", A7): the serial is the installed view's identifier, which a loss moves only once the sequencer's map
     moves and its next view is installed here, not at this node's own map_put.  So the same child loads no plan of an
     operand it loaded before (plans_walk: once an operand; link_close keeps the operand's generation), as the ESTALE
     refused it; a child spawned after a loss, before the next view is installed, finds the plan's serial the view's
     and loads it with its cursors at execution 0 (open, in this unit's report).  What must stay: no record served
     from a load after its run began, in this child */
  /* the records, by lane, copy and assignment order: first[q (u + 1) + c] lane q's copy c's first, each lane's units
     counted by copy and summed in that order (copy pick / n; a lane's units come in list order, so in copy order) */
  /* 2026-11-05 ("The revisions of f3f7cf46", PM): beside them, after every lane's, the records of `extra` (above), one
     a member, in list order (`next`) */
  f->send_count=extra[MESH_SEND];f->receive_count=extra[MESH_RECEIVE];
  for(uint32_t q=0;q<MESH_LINK_LANES;q++){f->send_count+=lane_units[MESH_SEND][q];f->receive_count+=lane_units[MESH_RECEIVE][q];}
  f->sends=f->send_count?aligned_alloc(128,f->send_count*sizeof *f->sends):NULL;
  f->receives=f->receive_count?aligned_alloc(128,f->receive_count*sizeof *f->receives):NULL;
  for(int d=0;d<2;d++)f->first[d]=calloc(MESH_LINK_LANES*width,sizeof *f->first[d]);
  if((f->send_count && !f->sends) || (f->receive_count && !f->receives) || !f->first[0] || !f->first[1]){error=ENOMEM;goto failed;}
  uint64_t fill[2][MESH_LINK_LANES],next[2]={0,0};
  f->carrying[0]=f->carrying[1]=0;
  for(int d=0;d<2;d++){
    uint64_t *first=f->first[d],at=0;
    for(uint64_t k=0;k<units[d];k++)if(assigned[d][k]!=UINT32_MAX)first[assigned[d][k]*width+pick[d][k]/n]++;
    for(uint32_t q=0;q<MESH_LINK_LANES;q++){
      for(uint32_t c=0;c<copies;c++){const uint64_t m=first[q*width+c];first[q*width+c]=at;at+=m;}
      first[q*width+copies]=at;
      fill[d][q]=first[q*width];
      f->cursor[d][q]=(struct cursor){.at=first[q*width],.end=first[q*width+1]};
      if(lane_units[d][q])f->carrying[d]|=UINT32_C(1)<<q;
    }
    next[d]=at;
  }
  f->lanes=f->carrying[MESH_SEND]|f->carrying[MESH_RECEIVE];
  for(int d=0;d<2;d++)for(uint64_t k=0;k<units[d];k++){
    const uint32_t q=assigned[d][k];
    if(q==UINT32_MAX && (set_of[d][k]==UINT64_MAX || !kept[d][set_of[d][k]]))continue;
    const struct mesh_hop *hop=list+pick[d][k];
    /* 2026-11-05 ("Review of 4e78f6f4 against the requirement lines", K2; "The revisions of 55b9cbad"): the refusal of
       a field the shifts cannot hold and of a receive with no count word or no pages goes (link_prepare's note), and a
       receive's `count` is its range's count word as given, every receive naming its range's from 114
       (mesh-collective.c mesh_plan), where a 0 named none.  What must stay: each record's words its hop's */
    /* the span its hop's `bytes` from the start of its page (T7; mesh.h struct mesh_hop: "A hop moves its `bytes` from
       the start of that page or into it").
       2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1:
       "The bridge's instructions are those blocks: aot A9, MSCCL-IR's steps with offset and count"; mesh.h struct
       mesh_hop `offset`: "a hop moves bytes [page x pgsz + offset, + bytes) of the region's arena, `offset` +
       `bytes` <= block x pgsz"; "The revisions of 14b9a757", R-1), reversing the span from its page's start: the span its hop's
       `bytes` from byte `offset` of its page, at each end its own entry's (a send's in its source's block, a receive's
       in its destination's or its relay's).  A receive's `pages` is its range's instructions this node receives in one
       copy (mesh.h), each one landing on its count, so the landing's arithmetic ((j + 1) x `pages`, link_receive_drain)
       and the cancellation's final value (`pages` x executions: plan_end, plan_cancel) are unchanged.  What must stay:
       the span's length its hop's `bytes`, which the landing's length test reads */
    const struct ibv_sge span=wire_span(&port->device,(uint64_t)hop->page*M->pgsz+hop->offset,hop->bytes);
    const uint64_t at=q!=UINT32_MAX?fill[d][q]++:next[d]++;
    if(record_of[d])record_of[d][k]=at;
    if(d==MESH_SEND){
      struct prepared_send *record=f->sends+at;
      *record=(struct prepared_send){.word=region_word(hop->word),.field=hop->field,.span=span,.flow=f};
      record->request=(struct ibv_send_wr){.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1,.opcode=IBV_WR_SEND,
        .send_flags=IBV_SEND_SIGNALED};
    } else {
      struct prepared_receive *record=f->receives+at;
      *record=(struct prepared_receive){.span=span,.word=region_word(hop->word),.count=region_word(hop->count),
        .field=hop->field,.pages=hop->pages,.copy=(uint32_t)(pick[d][k]/n),.flow=f};
      record->request=(struct ibv_recv_wr){.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1};
    }
  }
  /* the sets' tables (struct flow; design/compilation-coordination.md, "The appendix's revisions", S4, S5 and S7, with
     S7' of its "Amendments during the work"), built once here: for each direction's sets with positions on this cable,
     in list order, its `set`, its size m on this cable (the positions the deal gave it, S7') and its first member; its
     members X_1 .. X_m, those positions' own records in list order (each position's record its own hop's: page, words,
     bytes); each position record's set and index among them; each lane's room for one permutation of the largest set
     it carries a position of (S7: "nothing is allocated at run time"); the generator's key, the plan's name (struct
     mesh_state `group`, `epoch`) zero-padded (S4; Salmon et al., section 5: "zero-pad a smaller input if the full
     period length or key space is not needed"); and this cable's place in the pair's key order (mesh-map.h
     mesh_cables: "cables in key order", the same at both ends), the counter's fourth word (S7').
     2026-11-04, owner B3 (S7'): a set's members on this cable alone, where they were the set's every position on the
     pair (the ENOTSUP refusal's note above), and the cable's place kept.  What must stay: the members' records this
     port's own.
     2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", OP): each set's size its every
     position on the pair, M, beside its m here (`local`); its M words of `ranks`, its j-th position's index among this
     cable's members or UINT32_MAX; each lane's room the largest M; this cable's place gone.  Why: appendix item 5's one
     permutation of the whole set per execution (set_draw, position_record).  What must stay: the members' records this
     port's own */
  f->key[0]=words->group;f->key[1]=words->epoch;f->key[2]=f->key[3]=0;
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", PM), reversing M1 ("The revisions of
     46ae8fdf": a cable's share of a set the members whose positions the deal gave it, so its records were its own
     members' alone): each set with a position on this cable has its every member on the pair, X_0 .. X_{M-1} in list
     order, at `members` from its `first`, each member's record the one built above (a position's own where the deal
     gave it this cable, else one of `extra`), each a view of that member's page (R1); each position record of this
     cable its set (`position_set`) and its index j among the set's M positions on the pair (`position_index`); `ranks`
     gone.  Why: design/aot-compilation-change-request.md, appendix item 5: "So the deal fixes positions, not members.
     Each lane keeps the positions the deal gives it.  In each execution, the member at each position is the one σ
     puts there.  Each port holds a record for every member of every set it carries a position of, each record a view
     of that member's page (R1)"; "(Corrected 2026-11-05: ... the implementation's M1 ...)".  What must stay: the
     positions the deal's; each lane's room for one permutation of the largest set it carries a position of */
  for(int d=0;d<2;d++){
    for(uint32_t q=0;q<MESH_LINK_LANES;q++){f->drawn_set[d][q]=UINT32_MAX;f->drawn_execution[d][q]=0;}
    if(!record_of[d])continue;
    const uint64_t records=d==MESH_SEND?f->send_count:f->receive_count;
    uint64_t whole=0,room[MESH_LINK_LANES]={0};
    for(uint64_t r=0;r<runs[d];r++)if(kept[d][r])whole+=bound[d][r+1]-bound[d][r];
    f->sets[d]=malloc(local[d]*sizeof *f->sets[d]);f->members[d]=malloc(whole*sizeof *f->members[d]);
    f->position_set[d]=malloc(records*sizeof *f->position_set[d]);f->position_index[d]=malloc(records*sizeof *f->position_index[d]);
    if(!f->sets[d] || !f->members[d] || !f->position_set[d] || !f->position_index[d]){error=ENOMEM;goto failed;}
    for(uint64_t r=0;r<records;r++){f->position_set[d][r]=UINT32_MAX;f->position_index[d][r]=0;}
    uint32_t sets=0;uint64_t m=0;
    for(uint64_t r=0;r<runs[d];r++){
      const uint64_t a=bound[d][r],e=bound[d][r+1];
      if(!kept[d][r])continue;
      for(uint64_t i=a;i<e;i++){
        const uint64_t k=member[d][i],at=record_of[d][k];
        const uint32_t q=assigned[d][k];
        f->members[d][m+i-a]=at;
        if(q==UINT32_MAX)continue;
        f->position_set[d][at]=sets;f->position_index[d][at]=(uint32_t)(i-a);
        if(e-a>room[q])room[q]=e-a;
      }
      f->sets[d][sets]=(struct flow_set){.set=list[pick[d][member[d][a]]].set,.size=(uint32_t)(e-a),.first=m};
      m+=e-a;sets++;
    }
    for(uint32_t q=0;q<MESH_LINK_LANES;q++)
      if(room[q] && !(f->drawn[d][q]=malloc(room[q]*sizeof *f->drawn[d][q]))){error=ENOMEM;goto failed;}
  }
  /* design/prepared-machine.md#M27 (its traces' room taken before the plan is told loaded, plan_load, so no refusal
     follows a load's `held` bit).  Room for one event a record each way, its first
     u executions' worth (each record's first posting, or the first landings), sized here once and written while room
     lasts (link_receive_drain, link_send_drain), so a trace is bounded by the template whatever the run's length */
  const char *ledger=getenv("MESH_LEDGER");
  f->ledger=ledger && ledger[0] && !(ledger[0]=='0' && !ledger[1]);
  const uint64_t traces[2]={f->send_count,f->receive_count};
  for(int d=0;d<2;d++){
    f->traced[d]=0;f->trace_capacity[d]=f->ledger?traces[d]:0;f->trace[d]=NULL;
    if(!f->trace_capacity[d])continue;
    if(!(f->trace[d]=aligned_alloc(32,(f->trace_capacity[d]+1)*sizeof(struct mesh_trace)))){error=ENOMEM;goto failed;}
  }
  /* 2026-11-04, owner B1 (design/compilation-coordination.md, Q5; e6bd7db2's section 6, "A load"): the port's
     announcement (`prepared` the generation, a seq_cst fence, the plan read again) is plan_load's, made before this
     load reads the plan, where it was made here, after, with hdr `hops` and `plan` read again; and the undoing of a
     refusal (`code` 0, phase MESH_PAIRED) is gone with the refusal's own `code` and phase (the plan's `refused` word in
     their place).  Why: D3 announces, then reads the head again, then computes; a plan's words are never rewritten
     while it stands, so no read here can be torn.  What must stay: nothing posted before the announcement and the head
     read again; the reclamation's pairing (plan_load's note) */
  /* the trace layout, the plan's copies and executions read from its state (the executions as the load finds them;
     link_run reads them again each pass) */
  if(f->ledger)for(uint32_t q=0;q<MESH_LINK_LANES;q++)if(f->lanes>>q&1)
    fprintf(stderr,"{\"trace_layout\":%u,\"rank\":%u,\"lane\":%u,\"send_base\":%llu,\"sends\":%llu,\"receive_base\":%llu,\"receives\":%llu,"
      "\"send_record_bytes\":%zu,\"receive_record_bytes\":%zu,\"copies\":%u,\"executions\":%llu}\n",port->index,M->node,q,
      (unsigned long long)(uintptr_t)(f->sends?f->sends+f->first[MESH_SEND][q*width]:NULL),
      (unsigned long long)lane_units[MESH_SEND][q],
      (unsigned long long)(uintptr_t)(f->receives?f->receives+f->first[MESH_RECEIVE][q*width]:NULL),
      (unsigned long long)lane_units[MESH_RECEIVE][q],sizeof(struct prepared_send),sizeof(struct prepared_receive),copies,
      (unsigned long long)atomic_load_explicit(&words->executions,memory_order_acquire));
  /* 2026-11-04, owner B4 (DL4): FLOW_SERVING, a candidate of the merge, where it was FLOW_DONE until link_run gave it
     a batch; the batch's bound and totals gone; `told` is plan_load's */
  f->words=words;f->copies=copies;f->state=FLOW_SERVING;
  f->sent=f->landed=f->posted=0;
  for(int d=0;d<2;d++){free(pick[d]);free(assigned[d]);free(member[d]);free(bound[d]);free(set_of[d]);free(lead[d]);free(record_of[d]);free(kept[d]);}
  free(bytes);free(carried);
  return 0;
failed:
  for(int d=0;d<2;d++){
    free(pick[d]);free(assigned[d]);free(member[d]);free(bound[d]);free(set_of[d]);free(lead[d]);free(record_of[d]);free(kept[d]);
    free(f->trace[d]);f->trace[d]=NULL;free(f->first[d]);f->first[d]=NULL;
  }
  flow_sets_free(f);
  free(bytes);free(carried);
  free(f->sends);free(f->receives);
  f->sends=NULL;f->receives=NULL;f->send_count=f->receive_count=0;f->lanes=f->carrying[0]=f->carrying[1]=0;
  return error;
}
/* A load of the plan standing at claim head `head` (its `owner` word, `generation` and state section `words` as the
   walk read them: plans_walk), on the port's progress thread: D3 of e6bd7db2's section 6, "A load", as
   design/compilation-coordination.md, Q5 takes it.
   1. The port's `prepared` stored the claim's generation (release), then a seq_cst fence: the announcement of the
      claim this port is reading, which the bridge's release reads before it frees a closed claim (mesh-dataflow.h
      mesh_retired_release: a seq_cst fence, then every port's `prepared`, then the plan's `held`).  It is the
      announced epoch and its re-read of deferred reclamation (C3.4: RCU's call_rcu, P. E. McKenney and J. D.
      Slingwine, Read-copy update, PDCS 1998; K. Fraser, Practical lock-freedom, PhD thesis, Cambridge, 2004; the
      contract's D10) and a hazard pointer's announcement (M. M. Michael, Hazard pointers: safe memory reclamation for
      lock-free objects, IEEE TPDS 15(6), 2004): it lives in memory reclamation cannot reuse (mesh.h struct
      mesh_port_info).
   2. The head read again (plan_stands: owner, generation, `closed`, publication `state`); where it changed, `prepared`
      0 and the plan skipped, neither loaded nor refused (EAGAIN, which the walk drops).  Of the two fences one comes
      first: where the release's does, this read sees the close and the load withdraws before it reads the plan; where
      this one does, the release reads this generation and keeps the claim.  Nobody waits.
   3. The plan built (link_prepare): ESTALE where its `serial` is not the map's, EALREADY as before, and its other
      refusals.
   4. Loaded: its `held` bit p set (atomic_fetch_or, release), port p the port's index; refused: its `refused[p]` the
      code (release), port p its one writer.  A port holds every standing plan it does not refuse, its bit set even
      where its cable carries nothing of it (e6bd7db2's section 6, "Discovery"), which the start's wait reads
      (mesh-call.h mesh_transfers_start: a `held` bit or a `refused` code at every port toward a node the plan names).
   5. `prepared` 0 (release).  The `held` bit is set before it, so a release that reads `prepared` withdrawn reads the
      bit after it (mesh-dataflow.c: `prepared` read first, then `held`, both acquire), and keeps the claim until the
      port ends the plan (plan_end).
   0 and the flow in *loaded; EAGAIN, the plan no longer standing at that head; or the refusal's code.  The
   announcement is transient, set and cleared within this call (mesh.h struct mesh_port_info).
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", SW): step 3's EALREADY is gone
   (link_prepare); the start reads no `held` bit (mesh-dataflow.c mesh_transfers_start), which the bridge's release
   still reads (mesh-dataflow.c mesh_retired_release).  What must stay: steps 1, 2, 4 and 5.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): the plan at place `function` of the program's root of generation `generation` (as the
   walk read them, plans_walk), where it was the claim at head `head` with its `owner` and generation: step 1's
   announcement is the root's `generation` in the port's `prepared`, which the root's release reads (mesh-dataflow.h
   mesh_retired_release: kept while a port's `prepared` is its generation), and step 2 reads the root again
   (plan_stands).  Why: point 5, "Release at retirement stays deferred reclamation (RCU)"; "Nothing walks heads to
   find it".  What must stay: steps 1, 2, 4 and 5.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4 and K1; "The
   revisions of 55b9cbad"): `generation` the operand's `published` (mesh.h struct mesh_algo), which step 1 announces
   in the port's `prepared` and step 2 reads again (plan_stands); `loaded` the port's table entry of place `function`
   (struct port_plans), NULL where the table could not grow to it, which is refused ENOMEM as a flow that cannot be
   allocated is.  What must stay: steps 1, 2, 4 and 5.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-3:
   "every field of the published operand is read through the acquired publication, never before it (McKenney and
   Slingwine)", and R-6; "The revisions of 14b9a757", R-3 and R-6): steps 1, 2 and 5 are the walk's, around all its
   loads of one operand (plans_walk), where they were each load's: the walk reads the operand's places toward the peer
   (mesh.h mesh_algo_adjacent) before any load, and those reads, as a plan's, are made only under the announcement and
   after the operand's re-read (plan_stands), so a release cannot reuse the operand's pages while the walk reads them
   (mesh-dataflow.c mesh_retired_release: a seq_cst fence, then every port's `prepared`, then the plans' `held`).  The
   load reads its plan after that re-read, at its place (mesh_algo_standing), so its own re-read and EAGAIN go.  Steps
   3 and 4 are here: `index` the peer's map index the walk read (link_prepare), step 3's ESTALE the walk's, once for
   the operand (plans_load).  The announcement stays transient, set
   and cleared within the walk's loads.  What must stay: the announcement and the operand's re-read before the plan is
   read; the `held` bit set before the announcement is cleared */
static int plan_load(struct port *port,uint32_t function,uint64_t generation,struct mesh_state *words,uint32_t index,
                     struct flow **loaded){
  const uint32_t p=port->index;
  struct flow *f=loaded?calloc(1,sizeof *f):NULL;
  const int error=f?link_prepare(port,f,words,index):ENOMEM;
  if(error){
    free(f);
    mesh_say("bridge","link %u (%s): plan (%llu, %llu) refused (code %d)",port->index,port->name,
             (unsigned long long)words->group,(unsigned long long)words->epoch,error);
    atomic_store_explicit(&words->refused[p],(int32_t)error,memory_order_release);
  } else {
    /* 2026-11-04, owner B4 (DL1, DL3): `told` the plan's `taken[p]` as this load finds it (0 for a plan no child of
       this port held, mesh_plan's starting value; an earlier child's last otherwise), so taken_tell only raises it.
       An earlier child of this port ends a plan it carries only as its link is lost (link_close), which changes the
       map's cables and so moves `map_serial` (mesh-records.c map_put), so a later load of that plan is refused
       (ESTALE, link_prepare) and no lane reads again a number its `taken[p]` let the parent overwrite.
       2026-11-05, owner B4 (review X1 #4), the note above made exact: what guards a reload is, first, link_prepare's
       EALREADY: a plan this port carries units of whose `executions` is above 0 at the load is refused, so no lane
       of a reload reads a number once its node's run has begun; a plan the port carries nothing of reads no word
       (its `taken[p]` follows `released`); and the same child walks again after link_close (`walk`) where its link
       serves again.  A carried plan reloaded before its run began, whose `taken[p]` plan_end raised to `released`
       while the reload's cursors would start at 0, is refused too (EALREADY: DL10, link_prepare, where `taken[p]` is
       above 0 at the load; 2026-11-05, owner B4, in place of this case recorded open here), so a loaded plan this
       port carries starts with `told` 0, and `told` from the word keeps it only rising for a plan it carries nothing
       of.  What must stay: `taken[p]` never lowered */
    /* 2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): `told` gone with `taken[p]`;
       the note above is the 112 form's.  What must stay: nothing of it */
    f->function=function;f->generation=generation;*loaded=f;
    atomic_fetch_or_explicit(&words->held,UINT64_C(1)<<p,memory_order_release);
  }
  return error;
}

/* The k-way merge at a lane, one direction (design/aot-compilation-change-request.md, appendix item 2, "Delivery at a
   node": "Each port serves the released executions of every standing plan in sequence-number order: a k-way merge (D.
   E. Knuth, The Art of Computer Programming vol. 3, section 5.4.1)"; design/compilation-coordination.md, "The
   documents' revisions of edd4b583", DL3; the book not read, so the merge is the item's own statement: the least of
   the k sequences' first elements next, by comparing them, no tree of losers), each sequence read as L. Lamport's
   Message Buffer receiver reads (Concurrent reading and writing, CACM 20(11), 1977, "Message Buffer": "receiver: wait
   until mr < ms; read message in B[mr mod P]; mr := mr + 1").  The k sequences are the plans lane q carries way d
   (each loaded plan not cancelled whose records this lane carries that way, struct flow `carrying`); a sequence's first
   element is its cursor's execution n there (struct cursor), a candidate where n < its `released` (acquire, Lamport's
   ms), its number its sequence word n mod u (B[mr mod P]; mesh.h struct mesh_state `sequences`).  The readings are in
   Lamport's order, "writing data elements in one order and reading them in the opposite order" (the conclusion): the
   bridge parent publishes in sequence order within one numbering, each word before its `released` (release) and ORDER
   after ORDER (mesh-records.c orders_deliver), so the lane reads every candidate's `released` (first reading), then
   every candidate's `released` again, the number of each it finds published, and hdr `plans` (second reading).
   A number m acquired in the first reading was published after every smaller number this node publishes, so the
   second reading, after that acquire, sees each of them, and its least, at most m, is next: every later publication's
   number exceeds m.  Where the first reading acquired none, nothing is taken this pass, since a number the second
   reading alone found published is not known least (a smaller one may be published between two of its reads), and
   `again` is set so the thread passes again at once (link_run's 1) rather than wait for a wake of `plans` that came
   already (DL3′).  The premise holds within one numbering: the sequencer's `seqnum` returns to 0 at its own map move
   (Q3′) and a node keeps its pending ORDERs across its own, so at a node whose map has not yet moved an ORDER
   numbered after the reset can match a plan of the old serial and the numbers published there go back, which this
   reading does not allow for; every plan of the old serial is cancelled at that node's own move (D.3.6), and no
   handler is added (design/compilation-coordination.md, "Numbers across a reset", open; 2026-11-05, owner B4, review
   X1 #1).  A number read stays as read: the parent stores word n mod u again (for n + u) only once every port's
   `taken[p]` exceeds n (Lamport's "wait until ms < mr + P"), and this port's does not while this lane has not taken n
   (taken_tell).  Nothing is taken where hdr `plans` moved since the port's last walk (a plan standing, and so maybe
   published, that the port has not loaded would be missed): the parent raises `plans` before it publishes a plan's
   first number (DL7, mesh-records.c orders_deliver; 2026-11-05, owner B4, the coordinator's ruling: mesh_plan stands
   a plan before it raises `plans`, so without that raise a plan's first numbers could be published unseen by this
   test), so every plan that can hold a number smaller than one this lane acquired is loaded here or moves `plans`
   before the second reading.  Nor is anything taken where the least is an execution its node's program has not
   submitted (n at or past the plan's `executions`, acquire, D2): it cannot be served, and nothing after it comes
   first.  The sequencer gives each execution its own number (mesh-records.c order_sequence), so no two candidates'
   are equal.  Both ends of a cable hold the same plans and the same records on each lane (link_prepare, one deal) and
   read the same numbers, so each lane takes the same executions in the same order at both ends ("they merge alike"),
   and the k-th SEND on its queue pair fills the k-th receive (Hilland et al., RDMA Protocol Verbs Specification,
   section 8.2.2, design/prepared-machine.md#M08).  The plan taken, its cursor's execution now the lane's to post
   (struct port_plans `current`), or NULL.  2026-11-04, owner B4: new (DL3), in place of link_run's one plan served to
   its batch's bound.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH and M5), reversing the merge above
   (the k sequences' numbers in the plans' sequence words, read in Lamport's opposite order, `released`, the `plans`
   test): lane q way d reads the port's copy of the deliveries' channel in order (deliveries_copy) from its place
   (struct port_plans `at`), each delivery a plan's name and executions [first, end) (mesh.h struct mesh_delivery):
   - a loaded plan of that name not cancelled (a cancelled one of the name passed over for it): where the lane carries
     it this way (struct flow `carrying`), its cursor's execution n: past the delivery (n >= end) the lane reads on; in
     it, n is taken where its node's program has submitted it (n < `executions`, acquire, D2), and nothing is taken
     where it has not; before it (n < first, a delivery of the plan this node never made: mesh-records.c order_take's
     ENOMEM) nothing is taken, the lane posting no execution of any plan out of the delivered order; where the lane
     does not carry it, the lane reads on;
   - a cancelled plan of that name alone: read on (a cancelled plan is served no more);
   - no loaded plan of that name: read on where a plan of that name standing here on this node's map serial has this
     port's refusal (`refused[p]`, plan_refused_here); else nothing is taken (it stands and is not yet loaded, or does
     not stand here yet), and `again` is set where hdr `plans` moved since the port's last walk, so the next pass walks
     and loads it.
   Both ends of a cable deliver the same sequence (Défago et al., Fig. 5, the destinations' code; item 2: "Both ends
   of a cable deliver the same sequence, so they take the calls alike") and hold the same plans and records on each
   lane (link_prepare, one deal), so each lane takes the same executions in the same order at both ends, and the k-th
   SEND on its queue pair fills the k-th receive.  Why: item 2 at mesh 0e56d8da: "The delivered calls reach the ports,
   in delivery order, through a first-in, first-out channel"; "Each port reads the channel in order"; "Each port takes
   the delivered calls of every standing plan in sequence-number order", delivery order being sequence order.  What
   must stay: each plan's executions on a lane in order; no execution taken past its node's `executions`.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", CH2): the lane reads each delivery in
   place, entry `at` mod hdr `deliveries_room` of the region's channel, below hdr `delivered` (acquire, read once a
   call: the parent's release follows the entry's stores, mesh-records.c orders_deliver), where it read the port's
   copy.  Why: appendix item 2: "Each port reads it in place, by view, and nothing copies it"; "An entry stays until
   every lane that carries its plan has taken its call".  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PQ), reversing CH2's reading of the one
   channel for every plan (and plan_refused_here, which only that reading asked): each plan's delivered calls are its
   own queue (struct flow `deliveries`, `room`; mesh.h struct mesh_delivery), and lane q way d takes its plans' calls
   by a multiway merge on the sequence number (D. E. Knuth, The Art of Computer Programming vol. 3, §5.4.1, as appendix
   item 2 states it: "of the first untaken entries of its standing plans, the least sequence number first"):
   - hdr `delivered`, D, read first (acquire): Fig. 5 delivers in sequence-number order, so every message numbered at
     most D is in its plan's queue here or was not this node's (mesh-records.c orders_deliver, which stores D after the
     entries), and no smaller one comes later (item 2: "every entry below the node's delivery count is in its queue
     already, and no smaller one comes later");
   - then, for each loaded plan not cancelled that the lane carries this way (struct flow `carrying`), its entries
     below its `delivered` (acquire) from the lane's place `at[d][q]`, those whose executions the lane's cursor has
     passed (n >= `end`) stepped over; its first untaken entry a candidate where its number is at most D;
   - the least candidate taken, where hdr `plans` (acquire, after D) is the value the port last walked at: the parent
     raises `plans` before it stores D, and every plan stands here before its first message comes (its barrier, A7), so
     a port that acquires D and reads `plans` unmoved has loaded or refused every plan that can hold a number at most
     D; where it moved, nothing is taken and `again` is set, so the next pass walks;
   - nothing taken where the least's executions [first, end) do not hold the lane's cursor n (n < first: a message of
     the plan this node never had) or n is at or past the plan's `executions` (acquire, D2: its node's program has not
     submitted it): nothing after it comes first.
   Both ends of a cable hold the same plans and records on each lane and the same numbers (Défago, Schiper and Urbán,
   §3, Uniform Total Order), so each lane takes the same calls in the same order at both ends, and the k-th SEND on its
   queue pair fills the k-th receive.  What must stay: each plan's executions on a lane in order; no execution taken
   past its node's `executions`; nothing copied.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-6:
   "Each cable's processor executes its list of instructions in the compile-time order, firing each on its dependency
   word ... It scans no table of plans"; W8: "The bridge's take and poll (mesh-flow.c 1263, 1879, 1891) visit every
   place to `plans`, empty ones included"; "The revisions of 14b9a757", R-6): the merge's candidates are the flows at
   the places of the operand's plans with an entry toward this port's peer (struct port_plans `adjacent`, `listed`;
   mesh.h mesh_algo_adjacent), each by its table entry, where every place of the table was visited.  A plan with no
   entry toward the peer has no record on any lane of this port, so it was never a candidate.  What must stay: as
   above */
static struct flow *lane_take(struct port_plans *own,int d,uint32_t q,int *again){
  const uint32_t lane=UINT32_C(1)<<q;
  const uint64_t delivered=atomic_load_explicit(&M->delivered,memory_order_acquire);
  struct flow *least=NULL;
  const struct mesh_delivery *head=NULL;
  for(uint32_t j=0;j<own->listed;j++){
    struct flow *f=own->loaded[own->adjacent[j]];
    if(!f || f->state==FLOW_CANCELLED || !(f->carrying[d]&lane))continue;
    const uint64_t count=atomic_load_explicit(&f->words->delivered,memory_order_acquire),n=f->cursor[d][q].execution;
    uint64_t *at=&f->at[d][q];
    while(*at<count && f->deliveries[*at%f->room].end<=n)++*at;
    if(*at==count)continue;
    const struct mesh_delivery *e=f->deliveries+*at%f->room;
    if(e->sequence<=delivered && (!head || e->sequence<head->sequence)){least=f;head=e;}
  }
  if(!least)return NULL;
  if(atomic_load_explicit(&M->plans,memory_order_acquire)!=own->walked){*again=1;return NULL;}
  const uint64_t n=least->cursor[d][q].execution;
  if(n<head->first || n>=atomic_load_explicit(&least->words->executions,memory_order_acquire))return NULL;
  return least;
}
/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
/* d6dceac's receive drain (its unordered form: a completion names its record), now one pass of the port's progress
   thread over the link's receive completions (the queue pairs' shared queue, `completion`), in the tag's batches
   (rdma/mesh-link.c link_receive): each landing of a lane served, after the acquire the poll's DMA asks, is its
   record's j-th since the load (`landings`), execution t = copy + j u (link_prepare), and writes the plan's state alone
   (design/disjoint-pages-change-request.md, C3.1 and C3.7; design/disjoint-pages-revision-coordination.md, "The
   bridge"; mesh.h struct mesh_hop, a receive): its phase field flipped by one atomic XOR (release) of bit field +
   ((t / u) & 1), t / u being j, so the field reads mesh_phase(t / u + 1); then its range's count of its copy one more
   (an atomic add, release); and, where the count it made is (j + 1) x the entry's `pages` (the pages this node
   receives of the range a copy), the count word's waiters woken (os_sync_wake_by_address_all, 8 bytes, region-shared,
   as the client's waiter waits: mesh-metal.m).  That add is the one that completes round j: by the allocator's order
   every landing of a copy's round j comes before any of its round j + 1, so the arrival that brings the count to
   (j + 1) P is a landing of round j, and an atomic add returns each value once, so exactly one landing wakes for each
   round (a cancellation's raise wakes for itself: count_raise).  A receive with no count (a relay's own, `count`
   NULL) flips its phase alone.  2026-11-04, the coordinator's ruling on review 15's finding 1: in place of this note's
   first 2026-11-04 test, the count a multiple of `pages` (a 64-bit division a landing), a multiply and a compare, and
   no add for a relay's own receive.  It is counted
   (the lane's `got` and frames, which lanes_flush reads); a failed one, or one of other than a whole page, fails the
   link (link_failed), never retried.  Then, after this pass's completions, each served lane's receives posted, its
   cursor's (struct cursor: the executions in order, execution t's records copy t mod u's), each execution the lane
   takes by the merge (lane_take: below its plan's `released` and `executions`), while its queue has room (the tag's requests,
   MESH_LANE_UNITS, and its frames, frame_capacity), a record of round j (execution t, j = t / u, the cursor's `round`)
   only once its `landings` reads at least j: its last posting has landed (one load, from the line its request
   starts).  The lane's posting stops at the first record that has not, so its order stays, and a page is named by at
   most one posted receive at any moment (design/disjoint-pages-change-request.md, D.2, "How many pages are in flight
   at once", and D.3.5; design/disjoint-pages-rulings-coordination.md, R1).  A receive waits on its own record's last
   landing alone, and behind the earlier receives of its lane, whose order the peer's SENDs fill; at most
   MESH_POST_QUANTUM a pass.  0 where the link failed.
   2026-11-04, sub-assignment 6: reversing 2026-11-03's landing, its record's `mask` ORed into its execution's landed
   record (C4: the landed record's ORed bits), and its chain of receives each posted once (`next`).  What must stay: the
   XOR before the add, both after the poll's acquire; no load, fence or call a landing more but the wake at a range's
   end; each lane's executions in order.
   2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, D.3.5 and D.4; design/disjoint-pages-rulings-
   coordination.md, R1): reversing sub-assignment 6's posting "ahead", on nothing but the queue's room, which posted a
   record again while an earlier posting of it might be outstanding, so two postings named one page.  Why: it gained
   nothing over the u rounds the plan's copies already give, and rested on the provider copying a request at its post,
   which nothing documents (D.2).  What must stay: the completions taken before the posting in the same pass, so a
   landing frees its record's next round at once; each lane's executions in order, no record skipped.
   2026-11-04, owner B1 (design/compilation-coordination.md, T7 and Q4): the plan `f` the one this port serves
   (link_run), whose executions are those below its `executions` and `released`; a landing of other than its record's
   bytes (its span's length, its hop's `bytes`) fails the link (EMSGSIZE), where it was one of other than a whole page,
   and each record's frames are frames_of(its bytes), posted and retired alike, where every record took a page's; each
   posting counted in the plan's `posted` (plan_end's test of what is outstanding).  What must stay: as above.
   2026-11-04, owner B3 (design/compilation-coordination.md, "The appendix's revisions", S7), where the cursor's
   position always posted its own record: at a position of a set in execution t, the record posted is member X_i's of
   the permutation of (t, set) (position_record: i the position's index in its set, the permutation drawn once a lane
   where its cursor meets the set in t); its round test (`landings` against the cursor's round), its landing's phase
   bit and count, its frames and its trace are that member record's own, so each member, posted once an execution of
   its copy, still lands its j-th time in execution copy + j u.  Why: appendix item 5.  What must stay: as above; a
   sequence's position posting its own record.
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL3 and DL5;
   design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node"), reversing B1's note above (the
   plan `f` the one link_run serves, through the executions below its `executions` and `released`, the batch's bound):
   the pass is the port's, over every plan it holds.  Each landing counts to its own record's plan (`flow`, DL5: its
   `landed`, and its trace where traced and that plan's room lasts), so the completions of several plans interleaved on
   one queue pair count apart.  Each served lane that any plan not cancelled carries this way then posts its receives
   by the merge: where it has taken no execution (struct port_plans `current`), lane_take gives the plan whose next
   execution there has the least published, submitted number, and the lane posts that execution's records on it in
   order, exactly as above (each record of round j once its `landings` reads j: D.3.5; a set's positions its
   permutation's members, S4-S7'), then takes the next; an execution with no record on the lane is taken and passed
   alike (its cursor stepped, cursor_step).  The post's budget, MESH_POST_QUANTUM a lane, spent, `again` is set.  Why:
   the item, "On each cable, the pages of every released execution interleave"; "Each port serves the released
   executions of every standing plan in sequence-number order".  What must stay: the completions before the posting;
   each plan's executions on a lane in order, its records in assignment order, none skipped; the round rule.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", PM): the round rule reads the record's
   page's phase field (acquire): a record of round j is posted only where it reads mesh_phase(j), its round j - 1
   landed, whichever port's child landed it, and its round j is stored in it as it is posted, its landing flipping bit
   field + (j & 1), where the rule read the record's own `landings` and the landing took its round from them.  Why:
   design/aot-compilation-change-request.md, appendix item 5: "Each port holds a record for every member of every set
   it carries a position of ... So any member can be posted at any of the set's positions on its cables", so a page's
   rounds may land through several ports' children, and D.3.5's rule ("each received page is named again only after its
   last landing has landed") needs the landing every child sees.  What must stay: the round rule; the XOR before the
   add, both after the poll's acquire.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PM5), reversing PM2's test of the
   page's phase field: a record of round j is posted only where its range's count word of its copy (`count`, acquire)
   reads at least j x `pages`, the pages this node receives of the range in a copy; for round 0 always.  Why:
   design/aot-compilation-change-request.md, appendix item 5, "A page's round rule reads a count that only rises": a
   phase parity is "valid only for: the current incomplete phase ... the immediately preceding phase" (NVIDIA, PTX ISA
   9.5, §9.7.15.16.19), which nothing keeps a namer within once a set's members move among a node's cables; "For a
   reader that nothing keeps within one phase, C3 already names the structure: "Counted completion with rising
   thresholds (Portals 4 counting events; `MTLSharedEvent`'s 64-bit value, which only rises)""; "By C3.3, every landing
   of a copy's round j comes after every landing of its round j − 1 ... So a page is named for round j where its
   range's count of its copy reads at least j times the pages this node receives of that range in a copy.  Then every
   page of round j − 1 has landed, its own among them.  The test holds at any distance and adds no word"; "The phase
   fields stay, for C3.7's readers" (the landing's XOR stays).  A cancellation's raise of the count (count_raise) passes
   it, and a cancelled plan is served no more (FLOW_CANCELLED).  What must stay: the XOR before the add, both after
   the poll's acquire; the record's `round` stored as it is posted.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K2; "The
   revisions of 55b9cbad"): the landing's add has no test of the record's `count` against NULL, which "A receive with
   no count (a relay's own ...)" above named: every receive names its range's count word from 114 (mesh-collective.c
   mesh_plan, sub-agent C's), as the round rule below already reads it.  What must stay: as above */
static __attribute__((always_inline)) inline int link_receive_drain(struct port *port,uint32_t served,int traced,int *again){
  struct link *l=&port->link;
  struct port_plans *own=port_plans+port->index;
  struct ibv_wc done[MESH_COMPLETION_BATCH];
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *)=l->provider.poll;
  for(int count=MESH_COMPLETION_BATCH;count==MESH_COMPLETION_BATCH;){
    count=poll(l->provider.completion,MESH_COMPLETION_BATCH,done);
    if(count<0){link_failed(port,EIO);return 0;}
    if(count)atomic_thread_fence(memory_order_acquire);
    for(int i=0;i<count;i++){
      const uint32_t lane=link_lane(l,done[i].qp_num);
      if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
      uint64_t status_opcode;
      memcpy(&status_opcode,&done[i].status,sizeof status_opcode);
      if((uint32_t)status_opcode){link_failed(port,(uint32_t)status_opcode);return 0;}
      if(!((status_opcode>>32)&IBV_WC_RECV))continue;
      /* a landing of other than its record's bytes (its span's length, its hop's `bytes`: a peer of another block, or
         a plan the two ends do not share) is the link's failure by error, EMSGSIZE, not a page told landed with bytes
         of it unwritten (review R1, round 2; T7) */
      struct prepared_receive *record=(void *)(uintptr_t)done[i].wr_id;
      if(done[i].byte_len!=record->span.length){link_failed(port,EMSGSIZE);return 0;}
      struct flow *f=record->flow;
      const uint64_t polled=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
      const uint64_t round=record->round;
      atomic_fetch_xor_explicit(record->word,UINT64_C(1)<<(record->field+(round&1)),memory_order_release);
      if(atomic_fetch_add_explicit(record->count,1,memory_order_release)+1==(round+1)*record->pages)
        os_sync_wake_by_address_all(record->count,sizeof *record->count,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
      struct lane_state *state=l->lane+lane;
      state->got++;state->frames_retired+=frames_of(record->span.length).n;f->landed++;
      /* design/prepared-machine.md#M27 */
      if(traced && f->traced[MESH_RECEIVE]<f->trace_capacity[MESH_RECEIVE]){
        const uint64_t published=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        f->trace[MESH_RECEIVE][f->traced[MESH_RECEIVE]++]=(struct mesh_trace){(uintptr_t)record,polled,published,published};
      }
    }
  }
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): every served lane reads the
     deliveries' channel (lane_take), where only a lane some plan carries this way merged, so each served lane's place
     passes the deliveries of plans it does not carry and the port lets them go (deliveries_copy).  2026-11-05 ("The
     revisions of f3f7cf46", CH2): read in place in the region's channel, nothing copied and nothing let go */
  for(uint32_t lanes=served;lanes;lanes&=lanes-1){
    const uint32_t k=(uint32_t)__builtin_ctz(lanes);
    struct lane_state *state=l->lane+k;
    const struct mesh_queue *queue=l->provider.queues+k;
    struct flow *f=own->current[MESH_RECEIVE][k];
    struct cursor at=f?f->cursor[MESH_RECEIVE][k]:(struct cursor){0};
    uint32_t budget=MESH_POST_QUANTUM;
    for(;budget;budget--){
      if(!f){
        if(!(f=lane_take(own,MESH_RECEIVE,k,again)))break;
        at=f->cursor[MESH_RECEIVE][k];
      }
      if(at.at==at.end){
        cursor_step(&at,f->first[MESH_RECEIVE]+k*((uint64_t)f->copies+1),f->copies);
        f->cursor[MESH_RECEIVE][k]=at;f=NULL;
        continue;
      }
      struct prepared_receive *record=f->receives+(f->position_set[MESH_RECEIVE]?position_record(f,MESH_RECEIVE,k,&at):at.at);
      if(atomic_load_explicit(record->count,memory_order_acquire)<at.round*record->pages)break;
      const uint64_t each=frames_of(record->span.length).n;
      if(state->posted-state->got>=MESH_LANE_UNITS || state->frames_posted-state->frames_retired+each>state->frame_capacity)break;
      struct ibv_recv_wr *bad;
      record->round=at.round;
      const int error=queue->receive(queue->pair,&record->request,&bad);
      if(error){lane_lost(port,k,error<0?-error:error);break;}
      state->frames_posted+=each;state->posted++;f->posted++;
      at.at++;
    }
    if(!budget)*again=1;
    if(f)f->cursor[MESH_RECEIVE][k]=at;
    own->current[MESH_RECEIVE][k]=f;
  }
  return 1;
}
/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
/* d6dceac link_receive_progress, now a pass and not a thread's entry: its traced variant under MESH_LEDGER; the plan
   the one link_run serves (2026-11-04, owner B1, Q4: where it was the port's one flow).  2026-11-04, owner B4 (DL3,
   DL5): the port's pass over every plan it holds, where it was one plan's; traced where any plan the port holds has a
   trace (`traced`, link_run), each record's event written to its own plan's trace while that plan's room lasts.  What
   must stay: the traced variant under MESH_LEDGER alone */
static int link_receive_progress(struct port *port,uint32_t served,int traced,int *again){
  /* design/prepared-machine.md#M27 */
  if(traced)return link_receive_drain(port,served,1,again);
  return link_receive_drain(port,served,0,again);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
/* d6dceac's send drain (each queue pair's records from its cursor, a record posted as its release word holds, through
   the provider's own post), now one pass of the port's progress thread: the SEND completions first, on their own queue
   (`sent`, mesh-verbs.h pair_up), in the tag's batches (rdma/mesh-link.c link_send), each of a lane served retiring its
   frames through the lane's gate and counted (`done`), a failed one failing the link (no completion counts a page's
   forwards: link_prepare's note of 2026-11-03); then each served lane's SENDs, its cursor's (struct cursor: the
   executions in order, execution t's records copy t mod u's), each execution the lane takes by the merge (lane_take:
   below its plan's `released` and `executions`), each of execution t once released, one acquire load of its word
   (design/disjoint-pages-change-request.md, C3.7; design/disjoint-pages-revision-coordination.md, "The bridge"): from
   its range's source (`field` MESH_ABSENT) where the range's supplied word reads at least t + 1, its producer's
   release of t; forwarding, where its receive's phase field reads mesh_phase(t / u + 1), that receive's landing of t
   (a cancellation writes no phase, so it releases nothing); while the lane's gate holds its frames and its requests are
   under MESH_LANE_UNITS, at most MESH_POST_QUANTUM a pass.  A ready page waits only behind earlier pages of its own
   queue pair.  0 where the link failed.
   2026-11-04, sub-assignment 6: reversing 2026-11-03's release, a word holding every bit of the record's `mask`
   (MESH_SUPPLIED, or its receive's bit of its execution's landed record: C4).  What must stay: one acquire load a send,
   and each lane's executions in order.
   2026-11-04, owner B1 (design/compilation-coordination.md, T7 and Q4): the plan `f` the one this port serves
   (link_run), whose executions are those below its `executions` and `released`; each SEND's frames frames_of(its
   record's bytes), its span's length, its hop's `bytes`, held and posted through the gate and retired at its
   completion, which names its record (`wr_id`), where every SEND took a page's frames; each posting counted in the
   plan's `posted` (plan_end).  What must stay: as above; a gate's frames posted and retired alike.
   2026-11-04, owner B3 (design/compilation-coordination.md, "The appendix's revisions", S7), where the cursor's
   position always posted its own record: at a position of a set in execution t, the record posted is member X_i's of
   the permutation of (t, set) (position_record), the one the peer's receive at that position posts (both ends draw it
   alike: S4, S5); its release (its word and field against t), its frames through the gate and its trace are that
   member record's own.  Why: appendix item 5.  What must stay: as above; a sequence's position posting its own
   record.
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL3 and DL5;
   design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node"), reversing B1's note above (the
   plan `f` the one link_run serves, through the executions below its `executions` and `released`): the pass is the
   port's, over every plan it holds.  Each SEND completion counts to its own record's plan (`flow`, DL5: its `sent`).
   Each served lane that any plan not cancelled carries this way then posts its SENDs by the merge, as the receive
   drain does (lane_take: the plan whose next execution there has the least published, submitted number; that
   execution's records on the lane posted in order, each once its word releases it, as above; an execution with no
   record on the lane taken and passed alike; then the next), the same order the peer's receive lane takes, so the
   k-th SEND on the queue pair fills the k-th receive.  The post's budget spent, `again` is set.  Why: the item, "On
   each cable, the pages of every released execution interleave".  What must stay: one acquire load a send; each
   plan's executions on a lane in order, its records in assignment order, none skipped */
static __attribute__((always_inline)) inline int link_send_drain(struct port *port,uint32_t served,int traced,int *again){
  struct link *l=&port->link;
  struct port_plans *own=port_plans+port->index;
  struct ibv_wc done[MESH_COMPLETION_BATCH];
  for(int count=MESH_COMPLETION_BATCH;count==MESH_COMPLETION_BATCH;){
    count=l->provider.poll(l->provider.sent,MESH_COMPLETION_BATCH,done);
    if(count<0){link_failed(port,EIO);return 0;}
    for(int i=0;i<count;i++){
      const uint32_t lane=link_lane(l,done[i].qp_num);
      if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
      if(done[i].status){link_failed(port,(int64_t)done[i].status);return 0;}
      const struct prepared_send *record=(const void *)(uintptr_t)done[i].wr_id;
      gate_retire(l->gate+lane,frames_of(record->span.length));l->lane[lane].done++;record->flow->sent++;
    }
  }
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): every served lane reads the
     deliveries' channel (lane_take), where only a lane some plan carries this way merged, so each served lane's place
     passes the deliveries of plans it does not carry and the port lets them go (deliveries_copy).  2026-11-05 ("The
     revisions of f3f7cf46", CH2): read in place in the region's channel, nothing copied and nothing let go */
  for(uint32_t lanes=served;lanes;lanes&=lanes-1){
    const uint32_t k=(uint32_t)__builtin_ctz(lanes);
    struct lane_state *state=l->lane+k;
    struct send_gate *gate=l->gate+k;
    const struct mesh_queue *queue=l->provider.queues+k;
    int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=queue->send;
    struct flow *f=own->current[MESH_SEND][k];
    struct cursor at=f?f->cursor[MESH_SEND][k]:(struct cursor){0};
    uint32_t budget=MESH_POST_QUANTUM;
    for(;budget;budget--){
      if(!f){
        if(!(f=lane_take(own,MESH_SEND,k,again)))break;
        at=f->cursor[MESH_SEND][k];
      }
      if(at.at==at.end){
        cursor_step(&at,f->first[MESH_SEND]+k*((uint64_t)f->copies+1),f->copies);
        f->cursor[MESH_SEND][k]=at;f=NULL;
        continue;
      }
      struct prepared_send *record=f->sends+(f->position_set[MESH_SEND]?position_record(f,MESH_SEND,k,&at):at.at);
      const uint64_t word=atomic_load_explicit(record->word,memory_order_acquire);
      if(record->field==MESH_ABSENT?word<=at.execution:((word>>record->field)&3)!=mesh_phase(at.round+1))break;
      const struct frames each=frames_of(record->span.length);
      if(state->sent-state->done>=MESH_LANE_UNITS || !gate_holds(gate,each))break;
      const uint64_t observed=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
      struct ibv_send_wr *bad;
      const int error=post(queue->pair,&record->request,&bad);
      /* design/prepared-machine.md#M27 */
      if(traced && f->traced[MESH_SEND]<f->trace_capacity[MESH_SEND]){
        const uint64_t sent=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        f->trace[MESH_SEND][f->traced[MESH_SEND]++]=(struct mesh_trace){(uintptr_t)record,observed,observed,sent};
      }
      if(error){lane_lost(port,k,error<0?-error:error);break;}
      gate_post(gate,each);
      __atomic_store_n(&state->sent,state->sent+1,__ATOMIC_RELEASE);gate->requests++;f->posted++;
      at.at++;
    }
    if(!budget)*again=1;
    if(f)f->cursor[MESH_SEND][k]=at;
    own->current[MESH_SEND][k]=f;
  }
  return 1;
}
/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
/* d6dceac link_send_progress, now a pass and not a thread's entry: its traced variant under MESH_LEDGER; the plan the
   one link_run serves (2026-11-04, owner B1, Q4: where it was the port's one flow).  2026-11-04, owner B4 (DL3, DL5):
   the port's pass over every plan it holds, traced as link_receive_progress is.  What must stay: the traced variant
   under MESH_LEDGER alone */
static int link_send_progress(struct port *port,uint32_t served,int traced,int *again){
  /* design/prepared-machine.md#M27 */
  if(traced)return link_send_drain(port,served,1,again);
  return link_send_drain(port,served,0,again);
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#meshresult */
/* The end of a plan a port's child holds, on its progress thread: where its claim closed or was reused (plans_walk),
   where the link serves no lane (a halt, a loss), and as the child closes (link_close).  Each of its receives with a
   count (none for a relay's own, which nobody waits on) not landed for every
   execution it serves (its `landings` below the executions t <
   `executions` of its copy, the plan's word read now, so a rise since the last pass is cancelled too) is cancelled, the
   cancellation an arrival (design/disjoint-pages-change-request.md, C3.1: "Landings and cancellations both arrive on
   the count ... A link loss is such a cancellation"; design/disjoint-pages-revision-coordination.md, "The bridge"):
   its range's count of its copy raised to its final value, the entry's `pages` x those executions, never lowered, and
   woken (count_raise); no phase is written.  These stores land in the plan's own pages while this port's `held` bit
   still holds the plan (cleared below, after them, release), so the bridge's release of the plan's closed claim
   (mesh-dataflow.c mesh_retired_release, its fence, then `held` read with acquire) comes after them.  Its traces told
   (d6dceac link_close's lines); its records freed; its `taken[p]` raised to its `released` (taken_store, hdr `plans`
   raised and woken where it rose: DL9, 2026-11-05), so the parent's sender waits on no execution this port will never
   read (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL4: "A port ends ... a plan as
   now, its `taken[p]` then following `released`"; Q4': "before clearing its `held` bit"); and its `held` bit p cleared
   (atomic_fetch_and, release), the port's last access to the plan (e6bd7db2's section 6, "Retirement and the deferred
   release", as
   design/compilation-coordination.md, Q5 takes it; deferred reclamation, P. E. McKenney and J. D. Slingwine,
   Read-copy update, PDCS 1998; K. Fraser, Practical lock-freedom, 2004).
   2026-11-04, sub-assignment 6: reversing 2026-11-03's cancellation, UINT64_MAX stored into the cancel word of the
   execution whose landed record lacked a receive's bit (C4: the cancel word and the landed record's several writers);
   a cancellation now has the one form a landing has, an arrival on its count.  What must stay: a count only raised,
   never a phase written for a cancellation, and `prepared` cleared after every store into the plan's pages.
   Where a SEND or receive of the link's queue pairs is outstanding (the lanes' counts),
   nothing more is loaded in the child's life, and where the link still serves, it is lost as the tag loses a link whose
   holdings ended with units outstanding (rdma/mesh-link.c stream_end: `lost` LINK_STREAMS_ENDED, every lane `broken`,
   so the child tells CHILD_BROKEN and its parent loses the link): a posted receive is never taken back but by the
   child's exit (lanes_flush), so no later plan's page is posted beside it, and `prepared` stays the plan's until the
   parent reaps the child.  No verbs call: d6dceac's down_pair and listener close are the child's exit's.
   2026-11-04, owner B1 (design/compilation-coordination.md, Q5; e6bd7db2's section 6): one plan of several, its
   holding its `held` bit, where it was the port's `prepared` (the region's one plan); what is outstanding is the
   plan's own requests (its `posted` against its `sent` and `landed`), where it was the lanes' counts, since a port
   holds several plans and serves one at a time (link_run); a plan ended with requests outstanding keeps its `held` bit
   until the parent reaps the child (streams_cancel: a dead child's bit cleared in every plan not yet released), as
   `prepared` was kept, and the port is `ended` (struct port_plans), as FLOW_ENDED was.  The refused state's end goes:
   a refusal holds nothing (plan_load).  What must stay: a count only raised, no phase written for a cancellation, the
   `held` bit cleared after every store into the plan's pages, and a plan's pages held while requests may name them.
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL1 and DL4: "A
   port ends, cancels or carries nothing of a plan as now, its `taken[p]` then following `released`"), reversing the
   notes above where they say the port serves one plan at a time (link_run) and raises `served[p]` so that the
   parent's release of the next batch is not held: the plan's `taken[p]` is raised to its `released` (taken_store,
   only upward), hdr `plans` woken where it rose (raised and woken from 2026-11-05: DL9, link_run's note), so the
   parent's sender (mesh-records.c orders_deliver) waits on no
   execution this port will never read; and no lane of the port keeps the plan as the one it has taken
   (lanes_drop).  Why: the item's delivery (L. Lamport's Message Buffer; DL1: a port's mr), with no hold.  What must
   stay: the cancellation's counts, the traces, `taken[p]` raised before the `held` bit is cleared */
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): taken_store and taken_tell (port
   p's `taken[p]` of each plan, Lamport's mr, the least execution a lane of the port may still read the number of,
   raised once a pass, on which the parent's sender waited) are gone.  Why: design/aot-compilation-change-request.md at
   mesh 0e56d8da, appendix item 2, "publish them to each port through the channel above, in place of the sequence words
   (`sequences`), `taken[p]` and the parent's wait while they are full"; what a port tells is how far it has read the
   deliveries' channel (`delivery_read[p]`, deliveries_copy).  What must stay: nothing of them */
/* no lane of the port keeps plan f as the one it has taken (struct port_plans `current`): where the plan ends or is
   cancelled, its execution there is posted no further (plan_end, link_run) */
static void lanes_drop(struct port_plans *own,const struct flow *f){
  for(int d=0;d<2;d++)for(uint32_t q=0;q<MESH_LINK_LANES;q++)if(own->current[d][q]==f)own->current[d][q]=NULL;
}
/* a plan ending or cancelled with `outstanding` of its requests posted on the link's queue pairs and not completed:
   nothing more is loaded or served in the child's life (`ended`), and where the link still serves, it is lost as the
   tag loses a link whose holdings ended with units outstanding (rdma/mesh-link.c stream_end: `lost`
   LINK_STREAMS_ENDED, every lane `broken`, so the child tells CHILD_BROKEN and its parent loses the link), once: a
   posted request is never taken back but by the child's exit (lanes_flush), so no later plan's page is posted beside
   it on the same queue pairs, where the k-th SEND fills the k-th receive posted (link_close's note, 2026-11-03, as
   plan_end keeps it) */
static void plan_outstanding(struct port *port,const struct flow *f,uint64_t outstanding,uint32_t served){
  struct link *l=&port->link;
  struct port_plans *at=port_plans+port->index;
  if(at->ended)return;
  at->ended=1;
  if(!served)return;
  mesh_say("bridge","link %u (%s): plan (%llu, %llu) ended with %llu requests outstanding: the link toward node %u lost",
           port->index,port->name,(unsigned long long)f->words->group,(unsigned long long)f->words->epoch,
           (unsigned long long)outstanding,port->peer);
  atomic_store_explicit(&l->lost,LINK_STREAMS_ENDED,memory_order_relaxed);
  atomic_fetch_or_explicit(&l->broken,l->plan,memory_order_release);
  wake();
}
static void plan_end(struct port *port,struct flow *f,uint32_t served){
  const uint32_t p=port->index;
  lanes_drop(port_plans+p,f);
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", PM): each receive's count raised to
     its final value where it reads below it (count_raise, which leaves a count at or past its final as it is), where it
     was raised where the record's own landings were below the executions of its copy.  Why: a set's member is posted at
     whichever position its execution's permutation gives it, on any cable of the pair (appendix item 5), so no record
     counts its page's landings; the count is the range's own, every port's arrivals on it.  What must stay: a count
     only raised, never a phase written for a cancellation */
  const uint64_t executions=atomic_load_explicit(&f->words->executions,memory_order_acquire);
  for(uint64_t r=0;r<f->receive_count;r++){
    const struct prepared_receive *record=f->receives+r;
    const uint64_t serves=copy_executions(executions,f->copies,record->copy);
    count_raise(record->count,(uint64_t)record->pages*serves);
  }
  /* design/prepared-machine.md#M27 */
  for(int d=0;d<2;d++){
    for(size_t i=0;i<f->traced[d];i++){
      const struct mesh_trace t=f->trace[d][i];
      fprintf(stderr,"{\"native_trace\":%u,\"direction\":%d,\"event\":%zu,\"identity\":%llu,\"ns\":[%llu,%llu,%llu]}\n",
        port->index,d,i,(unsigned long long)t.identity,(unsigned long long)t.begin,(unsigned long long)t.middle,(unsigned long long)t.end);
    }
    free(f->trace[d]);
  }
  const uint64_t outstanding=f->posted-f->sent-f->landed;
  free(f->sends);free(f->receives);
  for(int d=0;d<2;d++)free(f->first[d]);
  flow_sets_free(f);
  if(outstanding)plan_outstanding(port,f,outstanding,served);
  else atomic_fetch_and_explicit(&f->words->held,~(UINT64_C(1)<<p),memory_order_release);
  free(f);
}
/* every plan the port holds ended (plan_end), where its link serves no lane (a halt, a loss: port_progress) and as its
   progress thread ends, and the next pass that serves walks the heads again whatever hdr `plans` reads (`walk`), so a
   link that serves again loads each standing plan anew (EALREADY where its run has begun and the port carries units
   of it: link_prepare).  2026-11-04, owner B1 (Q5): every plan the port holds, where it was the port's one plan; what
   must stay: no plan served on a link that serves no lane.  2026-11-05 ("The revisions of f3f7cf46", SW): no
   EALREADY; a load of a plan whose serial the loss moved is refused ESTALE (link_prepare) */
/* 2026-11-05 ("Review of 4e78f6f4 against the requirement lines", K1; "The revisions of 55b9cbad"): every flow of the
   port's table by place (struct port_plans `loaded`), where the list was.  What must stay: as above.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-6: "It
   scans no table of plans"; design/aot-compilation-change-request.md A7, corrected 2026-11-06; "The revisions of
   14b9a757", R-6 and A7): the flows at the places toward the peer (struct port_plans `adjacent`), every loaded flow
   being at one, where every place of the table was visited; then `listed` 0, no flow of the operand held.  The
   operand's `generation` is kept, so this child loads no plan of it again (plans_walk loads an operand once), where the
   next walk loaded each anew and link_prepare refused it ESTALE, the loss having moved `map_serial`: the plan's serial
   is now the installed view's, which this node's own map move does not move (mesh-bridge.h plan_stale), and a reload's
   cursors would start at execution 0 of a run that has begun (link_prepare's SW note).  What must stay: no plan served
   on a link that serves no lane; every flow ended before `listed` is 0; no entry of the list read once no flow holds
   it (`holding` 0) */
static void link_close(struct port *port,uint32_t served){
  struct port_plans *at=port_plans+port->index;
  for(uint32_t j=0;at->holding && j<at->listed;j++){
    const uint32_t k=at->adjacent[j];
    struct flow *f=at->loaded[k];
    if(!f)continue;
    at->loaded[k]=NULL;at->holding--;
    plan_end(port,f,served);
  }
  at->listed=0;
  at->walk=1;
}
/* The walk of the claims' heads (e6bd7db2's section 6, "Discovery", as design/compilation-coordination.md, Q5 takes
   it), on the port's progress thread where hdr `plans` moved since its last walk (link_run): each loaded plan whose
   claim closed or was reused (plan_stands no longer) ended (plan_end); then each standing plan (mesh-dataflow.h
   mesh_head_after, mesh_head_state) this port has neither loaded (a flow of its head and generation) nor refused (its
   `refused[p]` not 0, which port p alone writes and never resets, so a refusal by an earlier child of this port
   stands: design/compilation-coordination.md, B1-1, "`refused[p]` is never reset") loaded
   (plan_load), and listed after those loaded before it.  A plan published or closed during the walk may be met or
   missed (mesh-dataflow.h: a walk reads a moving set); each publication and close raises `plans` again, so the next
   pass walks again.  `refused[p]` is read before the announcement: a value read of a claim released meanwhile only
   skips a plan that no longer stands, and one read 0 is loaded only after plan_load's announcement and head read
   again find the plan standing.  Nothing more is loaded once a plan ended with requests outstanding (`ended`).
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): the program's plans read at their places (mesh-bridge.h plan_from: k below the root's
   `plans`, mesh.h mesh_program_plan), where the claims' heads were walked; a loaded plan ended where its place no
   longer holds it standing in the root of its generation (plan_stands); a plan standing at place k loaded where this
   port holds no flow of (k, the root's `generation`, read after the plan) and has not refused it.  Why: point 5, "the
   serialized operand's place is a constant the region's layout names once ... Nothing walks heads to find it";
   design/aot-compilation-change-request.md A9, "What the bridge runs: the self-timed schedule": "the bridge reads the
   serialized operand at a place its compilation fixed, and searches for nothing".  What must stay: ended plans ended
   before any load; nothing loaded once `ended`; a refusal by an earlier child standing.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K1: "the
   held-flow search (mesh-flow.c 1696): a flow table indexed by place", and G4; "The revisions of 55b9cbad"),
   reversing the search of the loaded list for a flow of (k, generation): the flow at place k is the port's table entry
   k (struct port_plans `loaded`), read by index; a plan at place k is loaded where that entry is NULL, into it
   (place_at, which grows the table to the operand's `plans`), its generation the operand's `published` (acquire),
   read after the plan (plan_from).  An entry left after the ending pass stands, so it is the plan at k of the operand
   published: no release frees an operand while this port holds a plan of it (mesh-dataflow.h mesh_retired_release).
   What must stay: ended plans ended before any load; nothing loaded once `ended`; a refusal by an earlier child
   standing.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-3:
   "Publication is read as RCU's publish-subscribe (W5): every field of the published operand is read through the
   acquired publication, never before it (McKenney and Slingwine)", R-6: "Each cable's processor executes its list of
   instructions in the compile-time order ... It scans no table of plans", and R-8: "a gone holder is read from the
   heartbeat's words"; W8: "The places with flows toward each adjacent node are fixed when the plan is made, so they are
   a list the plan writes"; design/aot-compilation-change-request.md A7, corrected 2026-11-06; "The revisions of
   14b9a757", R-3, R-6, R-8 and A7), reversing the forms above:
   1. The publication acquired first (mesh.h mesh_algo_open), its generation announced in the port's `prepared`
      (release), a seq_cst fence, and the operand read again with its holder (plan_stands): plan_load's steps 1 and 2,
      made once for the walk, so every read of the operand below (`plans`, `table`, `nodes`, `adjacent`, a plan's
      words) is through the acquired publication and under the announcement (M. M. Michael, IEEE TPDS 15(6), 2004: a
      hazard pointer announced, then the pointer read again; mesh-dataflow.c mesh_retired_release reads `prepared`
      after its own fence).  Where plan_stands gives 0 (not open, another operand, or its holder gone), every flow ends.
   2. The ending pass over the places toward the peer (struct port_plans `adjacent`): a flow ends where its operand is
      not the one standing or its place no longer holds its plan standing (mesh_algo_standing), and no entry is read
      once no flow is held (`holding` 0), where a retired operand, announced by no port, may be released.
   3. An operand not met before (`generation` not the port's): plans_load, once an operand.  Every plan of an operand
      stands from its publication or not at all (a plan's `closed` only rises), so the first walk of the operand meets
      every plan it will load; a later walk of it only ends.
   4. `listed` 0 where no flow of the operand is held, then `prepared` 0 (release), after every `held` bit set.
   What must stay: the publication, its announcement and the operand's re-read before any field of it; ended plans
   ended before any load; nothing loaded once `ended`; a refusal by an earlier child standing; the `held` bits set
   before `prepared` is cleared */
/* entry k of the port's table by place, the table grown where k is past it to the operand's `plans`, at least k + 1
   (realloc, its new entries NULL); NULL where it cannot grow (plan_load refuses the plan ENOMEM).
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-3; W5:
   "The count is read from the published operand after the acquire, so the index is bounded by construction"; "The
   revisions of 14b9a757", R-3): `plans` the walk's, read once after the publication's acquire (plans_load), where it
   was read here again; and the table grown to `plans`, k being below it (mesh.h struct mesh_algo `adjacent`: "each a
   place k below `plans`"), where it was the larger of `plans` and k + 1.  What must stay: the new entries NULL */
static struct flow **place_at(struct port_plans *at,uint32_t k,uint32_t plans){
  if(k>=at->places){
    struct flow **grown=realloc(at->loaded,(size_t)plans*sizeof *grown);
    if(!grown)return NULL;
    memset(grown+at->places,0,(size_t)(plans-at->places)*sizeof *grown);
    at->loaded=grown;at->places=plans;
  }
  return at->loaded+k;
}
/* The loads of an operand's plans toward this port's peer, in the walk, under its announcement (plans_walk): the
   installed view's records read whole (view_read), the peer's map index its place in them (map_index; A7, corrected
   2026-11-06: "Every member of a view plans on the view's map"), read once for the operand; the operand's view its
   plans' one view (mesh.h struct mesh_algo `nodes`: "the map the operand was planned on"; mesh_plan_view of struct
   mesh_state `serial`, `sequencer`, `boot`, compared with the installed view's identifier read with the records,
   mesh_view_compare); mesh_plan makes every function's plan on one reading and refuses a count of 0, EINVAL, so its
   plan at place 0 exists), and where it is not the installed view, none loaded: each plan of it ESTALE, the test
   link_prepare made for each plan against `map_serial`, made once here before the index is used, since the index
   names the operand's entries only in its own view (where they are one view, the index is below the operand's `nodes`
   by construction).  Where the view holds no record of the peer, no plan of the operand has an entry toward it, and
   none is loaded.  Else the places toward the index (mesh.h mesh_algo_adjacent: R-6; W8), each plan standing at one
   and not refused by this port (`refused[p]`, never reset) loaded (plan_load) into the table, grown to `plans` read
   once.  A failed view read (ETIMEDOUT, EINVAL, ENOMEM: records_read) is said and loads none of the operand, as
   link_prepare's map read refused each plan.  The flows loaded are counted in `holding`, the operand's places kept
   (struct port_plans `adjacent`, `listed`).  What must stay: only the places toward the peer loaded; the view's reading
   and its test before the index is used */
static void plans_load(struct port *port,struct port_plans *at,uint64_t generation){
  struct mesh_record *records=NULL;uint64_t count=0;
  struct mesh_view_id view;
  const int error=view_read(&records,&count,&view);
  if(error){
    mesh_say("bridge","link %u (%s): the installed view not read (code %d): no plan of operand %llu loaded",
             port->index,port->name,error,(unsigned long long)generation);
    return;
  }
  const uint32_t index=map_index(records,count,port->peer);
  free(records);
  if(index==UINT32_MAX)return;
  const struct mesh_algo *algo=mesh_algo(M);
  const struct mesh_view_id serial=mesh_plan_view(mesh_algo_state(M,algo,0));
  if(mesh_view_compare(serial,view)){
    mesh_say("bridge","link %u (%s): operand %llu planned on view (%llu, %u, %llu), not the installed view (%llu, %u, "
             "%llu): no plan of it loaded (code %d)",port->index,port->name,(unsigned long long)generation,
             (unsigned long long)serial.id,serial.sequencer,(unsigned long long)serial.boot,(unsigned long long)view.id,
             view.sequencer,(unsigned long long)view.boot,ESTALE);
    return;
  }
  const uint32_t plans=algo->plans;
  at->adjacent=mesh_algo_adjacent(M,index,&at->listed);
  for(uint32_t j=0;j<at->listed;j++){
    const uint32_t k=at->adjacent[j];
    struct mesh_state *words=mesh_algo_standing(M,k);
    if(!words || atomic_load_explicit(&words->refused[port->index],memory_order_relaxed))continue;
    if(!plan_load(port,k,generation,words,index,place_at(at,k,plans)))at->holding++;
  }
}
static void plans_walk(struct port *port,uint32_t served){
  struct port_plans *at=port_plans+port->index;
  struct mesh_port_info *info=&mesh_links(M)[port->index].port;
  const uint64_t open=mesh_algo_open(M);
  atomic_store_explicit(&info->prepared,open,memory_order_release);
  atomic_thread_fence(memory_order_seq_cst);
  const uint64_t generation=plan_stands(open);
  for(uint32_t j=0;at->holding && j<at->listed;j++){
    const uint32_t k=at->adjacent[j];
    struct flow *f=at->loaded[k];
    if(!f || (f->generation==generation && mesh_algo_standing(M,k)==f->words))continue;
    at->loaded[k]=NULL;at->holding--;
    plan_end(port,f,served);
  }
  if(generation!=at->generation){
    at->generation=generation;at->listed=0;
    if(generation && !at->ended)plans_load(port,at,generation);
  }
  if(!at->holding)at->listed=0;
  atomic_store_explicit(&info->prepared,0,memory_order_release);
}

/* The port's reading of the deliveries' channel (mesh.h struct mesh_delivery; design/aot-compilation-change-request.md
   at mesh 0e56d8da, appendix item 2, "Delivery at a node"; design/compilation-coordination.md, "The revisions of
   46ae8fdf", CH, M3 and M5): at the child's first pass (`joined` 0) every lane's place is hdr `delivered` as it reads
   it (M3: a plan this child serves stands on a map serial its own pairing moved, so every delivery of it comes after).
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", CH2), reversing M5: deliveries_copy (the
   channel copied into the child's own memory each pass, the copy grown by doubling, and `delivery_read[p]` told after
   it) is this join alone; each lane reads the channel in place (lane_take) and nothing is told.  Why:
   design/aot-compilation-change-request.md, appendix item 2: "Each port reads it in place, by view, and nothing copies
   it.  Only the driver moves bytes"; "The writer never waits and never checks for a full channel"; "(M4 and M5 go)".
   What must stay: every lane's place at `delivered` at the child's first pass.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PQ): deliveries_join is gone: a lane's
   place is in each plan's own queue (struct flow `at`), 0 at the plan's load, every message of a plan being in its
   queue from the plan's first (its barrier, design/aot-compilation-change-request.md A7).  What must stay: nothing of
   it */

/* The plans' turn on the port's progress thread, each pass while its link serves (its 2026-11-04 form, owner B1, at
   the end of this note; the rest of the note is the one plan's, before it): the plan served ended where the
   region's was written again (link_close: hdr `hops` 0, or another generation `plan`) or the client it was loaded for
   holds the region no more (hdr `client` another: mesh-dataflow.c mesh_retire); a plan written while a client holds
   the region and not yet loaded loaded (link_prepare), its generation stored in the port's `prepared` by link_prepare
   before its last check (release, then a seq_cst fence: its note of 2026-11-03; the client's start waits on it at
   every port whose cable reaches a node a hop names, so it is stored where the rule gave this cable no page too), or
   refused, the port's phase MESH_STOPPED, its `code` the refusal's and its `prepared` the refused
   generation (CONTRACT "A plan loaded", review R1 round 2), until the plan call writes again, or read torn and tried
   again on the next pass; a plan loaded served, its receives, then its SENDs (a landed relay page's SEND goes in the
   same pass), through the executions below the plan's `executions`, read (acquire) each pass the plan is served or
   done (one load a pass: the client's start raises it, design/disjoint-pages-revision-coordination.md, D2, and wakes
   `plan`, on which the idle thread waits: port_progress), until every record has served every one of them (every
   SEND completed, every receive landed: done), and served again where it rises (done to served).  1 while it serves a
   plan with work left, 0 idle.
   2026-11-04, sub-assignment 6: reversing 2026-11-03's done, every record's one posting completed with hdr
   `executions` fixed at the plan (C3.3: no input of the plan); what must stay: done only where every execution the
   word asks is served.
   2026-11-04, owner B1 (design/plan-to-bridge-change-request.md R8; design/aot-compilation-change-request.md A7;
   design/compilation-coordination.md, Q4 and Q5): reversing this note's one plan (ended where hdr `hops`, `plan` or
   `client` moved, loaded or refused here) for every plan standing at once.  Each pass: where hdr `plans` moved since
   the last walk (as port_progress read it before this pass), or the port's plans ended (`walk`), the heads walked
   (plans_walk: plans whose claims closed ended, plans standing loaded or refused).  Then for each plan held, its
   `executions` (the program's submission, Q1) and `released` (its bridge parent's, in the island's sequence order)
   read (acquire): a plan this port carries nothing of has `served[p]` raised to `released` (release, port p its one
   writer); a plan it carries is served to the lesser of the two, execution t only where t < `executions` and t <
   `released` (Q4).  The parent releases one plan's batch at a time, once every port holding the plan released before
   it has served it (Q4; mesh-records.c), so at most one plan has executions released and not served here: that plan
   is served (the first held plan with work, in the order loaded), each lane's executions in order, so a cable carries
   one plan's batch at a time, each execution whole, in sequence order (Défago, Schiper and Urbán, ACM Comput. Surv.
   36(4) 2004, Fig. 5: messages delivered in sequence-number order; Pathways, MLSys 2022, section 4.4: work enqueued in
   FIFO order): both ends post the released executions of several plans on each queue pair in one order, since the k-th
   SEND fills the k-th receive posted, and this port learns of its parent's deliveries only each plan's `released`, a
   count (design/compilation-coordination.md, "The appendix's revisions", O2).  Once every record of it has served every execution to that bound (done),
   `served[p]` is raised to it (release): the executions this port has served whole, every SEND of them completed and
   every receive landed.  1 while a plan has work left, 0 idle.  What must stay: done only where every execution asked
   is served; no execution of a plan past its `released`.
   2026-11-04, owner B1 (design/compilation-coordination.md, Q4' and Q3'): each raise of `served[p]` wakes hdr
   `plans` without raising it (served_tell); and a held plan whose `serial` is no longer this node's `map_serial`
   (read once a pass) is cancelled (FLOW_CANCELLED: the bridge parent has cancelled its receives, plan_stale, D.3.6,
   "a plan stands only while its map does", and dropped its pending ORDERs, Q3'; read so, design/compilation-
   coordination.md, B1-1: "a port reads a plan as cancelled where its `serial` is not this node's map_serial"): it is
   served no more, and its
   `served[p]` is raised to its `released` and on to every later `released` read while the port holds it, as a plan
   it carries nothing of (Q4': "raises `served[p]` to `released` for a plan cancelled").  Where requests of it are
   outstanding as it is cancelled, the port is ended and its link lost (plan_outstanding), since the parent may now
   release another plan's batch onto the same queue pairs, which only the child's exit clears (link_close's note,
   2026-11-03): the plan's end's rule (plan_end) applied here, design/compilation-coordination.md, B1-1 ("a plan
   cancelled while it has requests posted stops the port and loses its link, as link_close does for a plan's end").
   `served[p]` for a carried plan rises when the bound it serves is done whole (B1-1).  What must stay: no plan's pages
   posted beside another's outstanding requests.
   2026-11-04, owner B3 (design/compilation-coordination.md, "The appendix's revisions", O2: "Not done: Q4's hold
   stays"): the reason this note gives for serving one plan's batch at a time was "a cable never interleaves two
   executions and carries each whole, in sequence order (the contract's ruling 1)", Claude's words, which
   design/aot-compilation-change-request.md, appendix item 2, withdraws; it is O2's evidence now (the k-th SEND on a
   queue pair fills the k-th receive posted, so both ends post one order; a port learns only each plan's `released`, a
   count, so the delivery order across plans does not reach it; carrying it needs a structure the change requests do
   not name).  No behavior changed.  What must stay: one plan served at a time here, until the operator names what
   carries the delivery order to the ports (O2, open).
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL3, DL4 and DL5;
   design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node"; L. Lamport, Concurrent reading and
   writing, CACM 20(11), 1977, "Message Buffer"), reversing the notes above where they serve one plan at a time to its
   batch's bound (the lesser of its `executions` and `released`, `released` a count of a batch; FLOW_DONE; `served[p]`
   raised once the batch was done whole; "a cable carries one plan's batch at a time, each execution whole"), the
   hold O2 kept: the port serves every standing plan at once, interleaved.  Each pass: the walk where hdr `plans` moved
   (as above); each held plan whose `serial` is no longer this node's `map_serial` cancelled as above (FLOW_CANCELLED,
   B1-1; plan_outstanding where requests of it are outstanding), and dropped by every lane that had taken it
   (lanes_drop); then the receives and the SENDs of every plan, each lane each way merging the plans it carries by
   sequence number (lane_take: the parent's published numbers, read in Lamport's opposite order), each execution it
   takes posted on it in order as before, then the next (link_receive_drain, link_send_drain); then for each plan held,
   its `taken[p]` (taken_tell: the least execution a lane carrying it may still read; `released` for a plan the port
   carries nothing of or has cancelled), and hdr `plans` woken (no raise; raised from 2026-11-05, DL9, below) where
   any rose, so the parent's sender sees
   its buffer emptied.  Busy (1) while any plan has requests posted and not completed, any lane has taken an execution
   with records left to post (waiting on a word's release, a landing or room), or a lane could take a published,
   submitted execution it did not (its budget spent, or a number the second reading alone found: lane_take's
   `again`); idle (0) otherwise, and port_progress waits on hdr `plans` (bounded by MESH_PAUSE_NS), which every
   publication's wake, start and close moves or wakes.  Why: the item: "On each cable, the pages of every released
   execution interleave"; "Each port serves the released executions of every standing plan in sequence-number order: a
   k-way merge"; "Both ends of a cable read the same sequence numbers, so they merge alike.  Nothing holds."  What must
   stay: the walk before serving; a cancelled plan served no more; each plan's executions on a lane in order; the k-th
   SEND on a queue pair filling the k-th receive (both ends' one merge); no lane posting past a plan's `released` or
   its node's `executions`.
   2026-11-05, owner B4 (review X1 #5): this reversal includes B1's "What must stay: no plan's pages posted beside
   another's outstanding requests" above, which the interleaving (DL3) contradicts: several plans' requests are
   outstanding on one queue pair at once.  What stays of it: a plan cancelled with requests outstanding ends the link
   (plan_outstanding), as before.
   2026-11-05, owner B4, the coordinator's ruling DL9 (reviews X1 #7 and V1 #9): where a `taken[p]` rose, hdr `plans`
   is raised and woken (mesh.h mesh_notify), where it was woken without a raise (Q4′'s form, overturned with Q4).  Why:
   a wake with no raise that lands between the parent's watcher's read of `plans` and its wait (mesh-link.c
   control_watch) is lost until MESH_PAUSE_NS, and the delivery hands off between the parent and its ports once per
   execution past the u-th; os_sync_wait_on_address returns at once where the word has moved.  Its cost, recorded:
   each raise moves `plans`, so each port walks the heads on its next pass (plans_walk), DL3's `plans` test lets a lane
   that read the old value take nothing for one pass, and the parent's release turn runs again (mesh-records.c
   clients_turn).  What must stay: the raise once a pass, after every `taken[p]` store of it.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-6: "It
   scans no table of plans"; W8: "The bridge's take and poll (mesh-flow.c 1263, 1879, 1891) visit every place to
   `plans`, empty ones included"; design/aot-compilation-change-request.md A7, corrected 2026-11-06: "A plan then
   stands while its view does"; "The revisions of 14b9a757", R-6 and A7): the cancellation's loop and the busy test
   visit the places toward the peer (struct port_plans `adjacent`, `listed`), where they visited every place of the
   table; and a flow is cancelled where its view (mesh.h mesh_plan_view) is not the installed view's identifier
   (mesh_view_installed, read once a pass), where it was this node's `map_serial`.  What must stay: a cancelled plan
   served no more; the walk before serving */
static int link_run(struct port *port,uint32_t served){
  struct port_plans *at=port_plans+port->index;
  if(!at->ended && (at->walk || at->plans!=at->walked)){at->walk=0;at->walked=at->plans;plans_walk(port,served);}
  if(at->ended)return 0;
  const struct mesh_view_id view=mesh_view_installed(M,memory_order_relaxed);
  int traced=0;
  /* 2026-11-05 ("Review of 4e78f6f4 against the requirement lines", K1; "The revisions of 55b9cbad"): the port's
     flows read from its table by place (struct port_plans `loaded`), where they were a list.  What must stay: as
     above */
  for(uint32_t j=0;j<at->listed;j++){
    struct flow *f=at->loaded[at->adjacent[j]];
    if(!f)continue;
    if(f->state!=FLOW_CANCELLED && mesh_view_compare(mesh_plan_view(f->words),view)){
      f->state=FLOW_CANCELLED;lanes_drop(at,f);
      if(f->posted!=f->sent+f->landed)plan_outstanding(port,f,f->posted-f->sent-f->landed,served);
    }
    traced|=f->trace[MESH_SEND] || f->trace[MESH_RECEIVE];
  }
  if(at->ended)return 0;
  int busy=0;
  if(!link_receive_progress(port,served,traced,&busy) || !link_send_progress(port,served,traced,&busy))return 0;
  for(uint32_t j=0;j<at->listed;j++){
    const struct flow *f=at->loaded[at->adjacent[j]];
    busy|=f && f->posted!=f->sent+f->landed;
  }
  for(int d=0;d<2;d++)for(uint32_t q=0;q<MESH_LINK_LANES;q++)busy|=served>>q&1 && at->current[d][q];
  return busy;
}

/* tag pre-return-20261103 rdma/mesh-link.c:367-371 port_tell (60b00774), verbatim */
/* the progress thread's tell (the epoch it has seen, the guard it holds at), its raiser woken (port_wait) */
static void port_tell(_Atomic uint64_t *word,uint64_t value){
  atomic_store_explicit(word,value,memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
/* tag pre-return-20261103 rdma/mesh-link.c:372-386 port_pass (8b757084), verbatim but its work: the plan's turn
   (link_run) in place of each stream's holding, SENDs, receives and completions, and whether it has work left told */
/* design/transport.md section 14: one pass of a port's progress thread: the lanes the link serves (`served` read once a
   pass: none posts on a lane its link no longer serves), the plan's turn on them (its receives first: a peer's SEND is
   held until its receive is posted, TN3205, so ours let its SENDs go).  It tells the port's epoch it began at (a
   conclusion around a stuck call moves it: ports_stalled).  1 where the plan has work left */
static int port_pass(struct port *port,uint64_t *told){
  const uint64_t epoch=atomic_load(&port->epoch);
  struct link *l=&port->link;
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  const int busy=served?link_run(port,served):0;
  if(epoch!=*told){port_tell(&port->seen,epoch);*told=epoch;}
  return busy;
}
/* tag pre-return-20261103 rdma/mesh-link.c:387-424 port_progress (3a82ede9), verbatim but the plan: while the link
   serves and its plan has no work left (none written, or done), the thread waits on the plan's word (hdr `plan`, its
   generation, region-shared) as it waits on the guard, looking again every MESH_PAUSE_NS (so a client's retirement,
   which wakes `control` and not `plan`, is seen within it); a link that serves no lane ends the plan it served
   (link_close: a halt); and the thread's end ends it too.  2026-11-04, owner B1 (design/compilation-coordination.md,
   Q5): the word waited on is hdr `plans`, which every publication, close of a claim and start raises (a retirement
   among them), where it was hdr `plan`; a halt and the thread's end end every plan the port holds (link_close).
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL3): the pass is
   idle (link_run's 0) where no plan has requests posted and not completed, no lane has taken an execution with
   records left, and no lane could take a published, submitted execution; then the thread waits on hdr `plans`, which
   the parent wakes after each turn that published (mesh-records.c orders_deliver), each start raises and every port
   wakes where its `taken[p]` rose, bounded by MESH_PAUSE_NS.  What must stay: the value read before the pass is the
   one waited on.  2026-11-05, owner B4 (DL9): the parent and the ports raise `plans` there, where they woke it, so a
   hand-off that lands between this read and the wait ends the wait at once.  What must stay: as above */
/* design/transport.md section 14: a port's progress thread, its SENDs' and its receives': it passes without pause while
   its link serves a lane (rubric M8: the queues read as fast as a hardware thread allows), and waits on the port's
   guard word while it serves none, looking again every MESH_PAUSE_NS (the stop's `closing`) */
static void *port_progress(void *argument){
  struct port *port=argument;
  pthread_setname_np("mesh.rdma");
  /* fault_heard: this thread its port's, and its handler on a stack of its own */
  stack_t alternate={.ss_sp=malloc(SIGSTKSZ),.ss_size=SIGSTKSZ};
  if(alternate.ss_sp)sigaltstack(&alternate,NULL);
  atomic_store_explicit(&port->thread[0],pthread_self(),memory_order_release);
  uint64_t told=atomic_load_explicit(&port->seen,memory_order_relaxed);
  while(!atomic_load_explicit(&port->closing,memory_order_acquire)){
    /* while the port's guard stands, or its link serves no lane, no pass (no post on any queue pair of its device) and
       no lane read: the thread waits on the guard word (design/transport.md B13: a guarded call in the driver costs no
       core), woken by its raising and lowering, by a settle and by the link's coming up (guard_wake), looking again
       every MESH_PAUSE_NS (the stop's `closing`), the guard told and the epoch at each waking, each tell waking its
       raiser (port_tell) */
    const uint64_t guard=atomic_load_explicit(&port->guard,memory_order_acquire);
    const uint32_t served=atomic_load_explicit(&port->link.served,memory_order_acquire);
    if(guard&1 || !served){
      /* a fault absorbed on the port (fault_heard: the handler, which may make no call but mmap, munmap and sigaction,
         halted the port) wakes its child's loop, which tells its parent at once, not at its next tick */
      if(atomic_load_explicit(&faulted,memory_order_relaxed))wake();
      if(guard&1)port_tell(&port->guarded,guard);
      /* a link that serves no lane ends the plan it served (no verbs call: link_close) */
      if(!served)link_close(port,0);
      const uint64_t epoch=atomic_load_explicit(&port->epoch,memory_order_acquire);
      if(epoch!=told){port_tell(&port->seen,epoch);told=epoch;}
      /* the wait on the word that holds the thread: the guard while it stands, else the lanes served while they are
         none, so a change of that word between the read above and the wait ends the wait at once (a wait on the guard
         for a link's coming up slept MESH_PAUSE_NS where the lanes were served between the two) */
      if(guard&1)os_sync_wait_on_address_with_timeout(&port->guard,guard,sizeof port->guard,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                      OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
      else os_sync_wait_on_address_with_timeout(&port->link.served,0,sizeof port->link.served,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
      continue;
    }
    /* hdr `plans` read once before the pass, the value the pass walks by (link_run) and the wait below waits on, so a
       publication, a close or a start during the pass ends the wait at once (each raises and wakes it, region-shared:
       mesh.h struct hdr `plans`).  2026-11-04, owner B1 (design/compilation-coordination.md, Q5; e6bd7db2's section
       6, "Discovery": "Each port child's progress thread reads `plans` once a pass, and when idle waits on it (SHARED,
       bounded by MESH_PAUSE_NS)"): `plans` in place of hdr `plan`, the one plan's generation, vacant from 110; what
       must stay: the value read before the pass is the one waited on */
    const uint64_t plans=atomic_load_explicit(&M->plans,memory_order_acquire);
    port_plans[port->index].plans=plans;
    if(port_pass(port,&told))continue;
    os_sync_wait_on_address_with_timeout(&M->plans,plans,sizeof M->plans,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                         OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
  }
  link_close(port,0);
  return NULL;
}
/* tag pre-return-20261103 rdma/mesh-link.c:425-429 guard_wake (60b00774), verbatim but not static, and the plans' word
   woken too (region-shared: the progress thread's third wait, port_progress), so a raise holds at once.  2026-11-04,
   owner B1 (Q5): hdr `plans`, where it was hdr `plan` */
/* design/transport.md B13: the progress thread parked on the port's guard looks again */
void guard_wake(struct port *port){
  os_sync_wake_by_address_all(&port->guard,sizeof port->guard,OS_SYNC_WAKE_BY_ADDRESS_NONE);
  os_sync_wake_by_address_all(&port->link.served,sizeof port->link.served,OS_SYNC_WAKE_BY_ADDRESS_NONE);
  if(M)os_sync_wake_by_address_all(&M->plans,sizeof M->plans,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* tag pre-return-20261103 rdma/mesh-link.c:430-442 port_threads (3a82ede9), verbatim but not static */
/* design/transport.md: the port's progress thread started, once; 0, or pthread_create's error */
int port_threads(struct port *port){
  if(port->threads)return 0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  atomic_store(&port->seen,atomic_load(&port->epoch));
  pthread_t created;
  const int error=pthread_create(&created,&attributes,port_progress,port);
  if(!error)port->threads=1;
  pthread_attr_destroy(&attributes);
  return error;
}
/* tag pre-return-20261103 rdma/mesh-link.c:443-456 port_wait (e3c4c986), verbatim but its deadline: main's mesh.h has
   no struct mesh_deadline, so the tag's mesh_deadline, mesh_deadline_left and mesh_expired are written out (the same
   MESH_PLATFORM_NS from the wait's start, the same end: the wait named on the log and the child's exit, status 1) */
/* design/transport.md: the raiser of a settle or a guard parked, never spinning, until the port's progress thread,
   where started, has its word (`told`: the epoch it has seen, or the guard it holds at) at `value`, woken by the
   thread's tell (port_tell), its guard's waiters woken as it begins and at each waking, within its deadline,
   MESH_PLATFORM_NS from its start: a thread that has not told by then is in a call that has not returned, and the
   port's child, the waiting process, ends at once naming the wait (design/eager-execution.md, assumption A4), its exit the
   kernel's release of its port's objects and its parent's loss of the port (child_reaped) */
static void port_wait(struct port *port,_Atomic uint64_t *told,uint64_t value){
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  for(uint64_t seen;port->threads && (seen=atomic_load_explicit(told,memory_order_acquire))<value;){
    const uint64_t at=clock_ns();
    if(at>=until){
      mesh_say("bridge","mesh-flow child: its port's progress thread holding a settle or the guard (port_wait): past its deadline "
               "of %.3f s; the program ends (design/eager-execution.md, assumption A4)",(double)MESH_PLATFORM_NS/1e9);
      _exit(1);
    }
    guard_wake(port);
    os_sync_wait_on_address_with_timeout(told,seen,sizeof *told,OS_SYNC_WAIT_ON_ADDRESS_NONE,OS_CLOCK_MACH_ABSOLUTE_TIME,until-at);
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:457-473 guard_lower and guard_raise (306b22be), verbatim but not static */
/* design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1": the guard of a port's device, on the port's queue,
   around a change of its verbs objects (queue pairs and completion queues made, connected or destroyed): raised, it
   returns once each started progress thread holds (it told the guard's value before its next pass, and passes no more
   while the guard stands), so no queue pair of the device is posted on while another is made, changed or destroyed
   (Apple's provider: a post on a long-lived queue pair across another's teardown stores to the device's shared doorbell
   mapping after it moved, tbt_post_recv); lowered, the threads parked on it woken and going on
   (design/transport.md B13).  It stands for every change once the port's threads run, the link lanes'
   long-lived queue pairs being posted on while the link is up.  Raised: 1; 0 where no progress thread runs (a thread
   that does not hold within its deadline ends the child: port_wait) */
void guard_lower(struct port *port,int raised){
  if(raised>0){atomic_fetch_add_explicit(&port->guard,1,memory_order_release);guard_wake(port);}
}
int guard_raise(struct port *port){
  if(!port->threads)return 0;
  port_wait(port,&port->guarded,atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1);
  return 1;
}

/* one standing plan here cancelled whole in the bridge parent (its state section `words`, which its caller found
   standing: plan_from, mesh_algo_standing), every receive of it and every synchronization edge into it (streams_cancel,
   plan_stale), as plan_end cancels its child's: for each receive entry of the list (copy c's at [c n, c n + n)), its
   range's count of copy c raised to its final value, the entry's `pages` x the executions t < `executions` (the plan's
   word, read once) with t mod u == c, never lowered, and woken (count_raise; design/disjoint-pages-change-request.md,
   C3.1; design/disjoint-pages-revision-coordination.md, "The bridge"); no phase is written.  The parent keeps no record
   of which receives landed (the children's `landings`), so it raises every such count: one already at its final value
   (every arrival in, or an earlier cancellation) is left as it is, and the entries of one range raise its word once.  A
   plan whose copies are 0 or do not divide its entries, or whose entries lie outside the region (plan_inside), which
   every port refuses (link_prepare, EINVAL), names no count to raise; nor does a receive entry whose `count` is 0 (a
   relay's own receive, which nobody waits on: 2026-11-04, the coordinator's ruling on review 15's finding 1).
   2026-11-04, sub-assignment 6: reversing 2026-11-03's cancellation, UINT64_MAX stored into the cancel word of each
   not-landed entry's execution (C4: the cancel word); what must stay: a count only raised, and no phase written.
   2026-11-04, owner B1 (design/compilation-coordination.md, Q5; e6bd7db2's section 6, "Cancellations, over every
   standing plan, on the parent's thread": "plan_cancel(state, from): one plan's receive entries from map index from
   (UINT32_MAX for every one), with executions read once"): reversing this function's region-wide form, the plan written
   named by its generation (hdr `plan`, review 8's finding 7) and read whole by hdr `hops` and `plan` read again around
   each entry (b3cc9990's re-read), and its caller streams_let_go (gone with it: a gone client's claims are closed by
   its retirement, mesh-dataflow.h mesh_retire, and the ports end its plans, plans_walk).  Why: any number of plans
   stand, each named by its state section, and a plan's entries are never rewritten while it stands (e6bd7db2's
   section 6: "the old re-checks for torn reads go"), so no entry read here can be torn; a plan that closes during the
   walk keeps its pages until the release, which runs on this thread after it (below).  What must stay: `executions`
   read once; no store from an entry of a plan the caller did not find standing; this walk and the release on one
   thread.
   Where its stores land (design/disjoint-pages-change-request.md, C3.6): in the plan's pages while its claim still
   holds them.  Every caller of this walk and of the release of a retired plan's claim (mesh-dataflow.c
   mesh_retired_release) runs on the bridge parent's one event-loop thread, mesh-link.c main's: streams_cancel from
   mesh-ports.c port_lost (called from the loop's turns and events alone: link_turn, ports_stalled, control_heard,
   hello_heard, port_flush, port_say, port_greet, port_watch, the message handlers, mesh-events.c network_heard and
   terminations_turn, mesh-children.c child_spawn and child_reaped), from mesh-children.c child_heard and child_reaped
   (the loop's events), from mesh-link.c region_adopt (main, before the loop) and from the stop (main, after it);
   plan_stale from mesh-records.c
   map_put and map_adopt alone (2026-11-04, sub-assignment 1, R2): map_put from record_changed and record_heard;
   record_changed from mesh-ports.c link_set (port_heard, port_lost, port_opened, ports_found) and link_turn,
   mesh-records.c health_turn (the loop's turn) and mesh-link.c main (before the loop); record_heard from mesh-ports.c
   message_heard (port_heard, from control_heard, the loop's events); port_lost from the callers above, and from
   port_say and port_flush at the stop (main, after the loop); map_adopt from mesh-link.c region_adopt (main, before
   the loop, after its release); and the release from clients_turn and region_adopt (after its streams_cancel) alone.
   The parent's other threads (control_watch, the finder's queue's block in ports_find, the terminations' queue's
   peer_terminated and peer_arrived) call none of them: they wake the loop, list the devices and store atomics
   (port_halt).  So a release never runs inside this walk: a plan found standing at a head keeps its claim's pages until
   a release that runs later on this thread, closed or not, and every count word computed from its entries is stored
   while they are held.  What must stay: this walk and the release on one thread.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 3f106747", BS and VW): where the whole plan is
   cancelled (plan_stale, a map's move or a view's installation), its synchronization edge's count
   into this node (mesh.h struct mesh_state `wait_count`, this thread its one writer) raised to UINT64_MAX and its
   waiter woken, so the program's start that waits on it ends at once and reads the moved serial (mesh-call.h
   mesh_transfers_start: ESTALE).  Why: design/aot-compilation-change-request.md, appendix item 2, "The only waits are
   at the sinks of the added synchronization edges, on a count that only rises.  Each ends when its source moves on, or
   when the view ends and the plan with it (`ESTALE`)"; C3.1, a cancellation arrives on a count.  What must stay: the
   count only raised.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 4, and "The
   revisions of 4e78f6f4"): every synchronization edge into the plan here (mesh.h struct mesh_sync, the `waits` entries
   at `syncs`) has its `count` raised to UINT64_MAX (release) and its waiter woken (os_sync_wake_by_address_all,
   region-shared), where the one edge's `wait_count` was.  Why: point 4, the program's whole IPC graph, a plan an actor
   a call, so a plan has as many edges into it here as Convert-to-SC-graph added (BSL §9, Fig. 6); appendix item 2:
   "Each ends when its source moves on, or when the view ends and the plan with it (`ESTALE`)".  What must stay: each
   count only raised, this thread its one writer.
   design/problems-of-14b9a757-change-request.md item 11 (E. A. Lee and S. Ha's self-timed strategy, as S. S.
   Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §2, state it: "retain the
   processor assignment and actor ordering specified by the fully static schedule"): every caller cancels the plan
   whole, a link loss as a view change does, so `from` goes */
static void plan_cancel(struct mesh_state *words){
  struct mesh_sync *edge=mesh_syncs(M,words);
  for(uint32_t e=0;e<words->waits;e++){
    atomic_store_explicit(&edge[e].count,UINT64_MAX,memory_order_release);
    os_sync_wake_by_address_all(&edge[e].count,sizeof edge[e].count,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  }
  /* 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K2:
     "`plan_inside` ... (mesh-flow.c ... 2053)"; "The revisions of 55b9cbad"): the return where `copies` was 0 or did
     not divide `entries` or the entries lay outside the region goes, and each receive's count word is raised as given,
     where a 0 named none: facts mesh-collective.c mesh_plan fixes (plan_inside's note; link_prepare's).  What must
     stay: each count only raised */
  const uint32_t copies=words->copies;
  const uint64_t executions=atomic_load_explicit(&words->executions,memory_order_acquire),n=words->entries/copies;
  const struct mesh_hop *list=mesh_hops(M,words);
  for(uint32_t c=0;c<copies;c++){
    const uint64_t serves=copy_executions(executions,copies,c);
    if(!serves)continue;
    for(uint64_t j=c*n;j<c*n+n;j++)
      if(list[j].direction==MESH_RECEIVE)count_raise(region_word(list[j].count),(uint64_t)list[j].pages*serves);
  }
}
/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): plan_after (the parent's walk of the claims' heads, mesh_head_after and mesh_head_state,
   for its cancellations) is removed; streams_cancel and plan_stale read the program's plans at their places
   (mesh-bridge.h plan_from).  Why: point 5, "Nothing walks heads to find it".  What must stay: every standing plan
   read, on the parent's thread */
/* the `held` bits `ports` (bit p port p) cleared (atomic_fetch_and, release) in every plan not yet released: each claim
   head whose owner word is not 0 (mesh-dataflow.h mesh_head_after) and whose publication `state` (acquire) names a
   state section, closed or standing, on the parent's thread, the releaser's.  A dead child's bit (e6bd7db2's section 6,
   "Retirement and the deferred release": "A child's death: streams_cancel(port) with !port->child clears bit p in
   every plan not yet released, on the parent's thread, the releaser's"; design/compilation-coordination.md, Q5): its
   requests are released by its exit, so nothing of it names the plans' pages any more, and the release
   (mesh_retired_release) frees a closed claim no live port holds.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): every plan of the program's root not yet released (its `owner`, acquire, not 0), closed
   or standing, each read at its place, the root's `table` word k for k below its `plans` (mesh.h struct
   mesh_program), where it was each claim head's publication `state`.  Why: point 5: the claim's head record goes, and
   "Release at retirement stays deferred reclamation (RCU)", whose release reads every plan's `held`
   (mesh-dataflow.c mesh_holding_retained).  What must stay: on the parent's thread, the releaser's.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4; "The
   revisions of 55b9cbad"): every plan of the operand (mesh.h struct mesh_algo) not yet released, its `published`
   (acquire) not 0 and not its `released`, closed or standing, each at the operand's `table` word k for k below its
   `plans` (mesh_algo_state), where the root's `owner` was read.  What must stay: on the parent's thread, the
   releaser's.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-10:
   "The duplicates R-1 leaves (W13), each merged: "published and not yet released""; W13: "written three times
   (mesh-collective.c 2091; mesh-dataflow.c 354; mesh-flow.c 2157), to be beside `mesh_algo_open`"; R-3: "every field
   of the published operand is read through the acquired publication, never before it"; "The revisions of 14b9a757",
   R-10 and R-3): the test is mesh.h mesh_algo_unreleased, its one form, where it was written here (its `released` read
   with acquire there, where it was relaxed here); then `plans` read once, after that acquire.  What must stay: on the
   parent's thread, the releaser's; every plan of the operand not yet released, closed or standing */
static void held_clear(uint64_t ports){
  if(!mesh_algo_unreleased(M))return;
  const struct mesh_algo *algo=mesh_algo(M);
  for(uint32_t k=0,plans=algo->plans;k<plans;k++)
    atomic_fetch_and_explicit(&mesh_algo_state(M,algo,k)->held,~ports,memory_order_release);
}
/* tag pre-return-20261103 rdma/mesh-link.c:619-654 streams_open (11807bec), reduced to what it is under the page lists:
   the tag's streams made new and live for a new pairing; here the port's serving opened, as its lanes connect and
   before any is served (mesh-child.c child_jobs): no plan loaded on the new pairing (`prepared` 0: its progress thread
   loads the plan written as the lanes are served, link_run), and the link's bandwidth told in the port's table, as
   d6dceac link_run told it at its pairing.  `node` is the port's peer, which the child already holds (`port->peer`,
   CHILD_CONNECT): the tag stored it in each stream.  2026-11-04, owner B1 (design/compilation-coordination.md, Q5):
   `prepared` is a load's transient announcement (plan_load), so this store finds it 0 already (no load runs while the
   link serves no lane: link_run) and is kept as before; the progress thread loads each standing plan as the lanes are
   served (plans_walk), where it loaded the plan written */
void streams_open(struct port *port,uint32_t node,uint64_t bandwidth){
  (void)node;
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  __atomic_store_n(&link->bandwidth,bandwidth,__ATOMIC_RELAXED);
  atomic_store_explicit(&link->port.prepared,0,memory_order_release);
}
/* tag pre-return-20261103 rdma/mesh-link.c:699-711 streams_cancel (11807bec), reduced to what it is under the page
   lists: the tag's streams of a port live no more and their words ~0; here, as its link is lost or the bridge stops
   (mesh-ports.c port_lost, mesh-children.c child_reaped, the stop), each plan whose lists name the port's peer (none
   where the view holds no record of it), every plan where `port` is NULL, cancelled whole (plan_cancel of the plan
   written, its generation hdr `plan` read at the call, review 8's finding 7: each one's
   count raised to its final value, no phase written; design/disjoint-pages-change-request.md, C3.1: "A link loss is
   such a cancellation"), and the port's (every port's) plan no longer loaded (`prepared` 0) where
   no child of it lives: a live child's requests may still name the plan's pages, so its `prepared` is its own to clear
   (link_close) or its reap's (child_reaped, after `port->child` is 0).  A receive the port's child lands after the
   cancellation flips its phase and adds its arrival beyond the final value, which a reader's test (the count at least
   its target) reads as the raise; the tag's cancel again (child_heard's connect of a lost pairing, child_reaped)
   raises the counts of the executions the plan's `executions` asks by then (2026-11-04, sub-assignment 6: the cancel
   word, UINT64_MAX and never cleared, gone: C4).  A cancel again comes after the loss has zeroed `port->peer` (mesh-ports.c
   port_lost, which cancels first), so each port's peer at its last cancel with one is kept (`cancelled`, the parent's,
   its node, its map index then and the view that index was read in, or every plan where the view was not read) and a
   cancel with no peer cancels by it,
   as at the tag (review R2, C1).  The parent never sees its child load a plan, so the peer is kept at that first cancel,
   where the loss's own cancel reads it.
   2026-11-04, owner B1 (design/compilation-coordination.md, Q5; e6bd7db2's section 6, "Cancellations, over every
   standing plan, on the parent's thread": "streams_cancel(port): every standing plan, from the lost peer", and
   "Retirement and the deferred release": "A child's death: streams_cancel(port) with !port->child clears bit p in
   every plan not yet released"): every standing plan cancelled (plan_after, plan_cancel), where it was the plan written
   (hdr `plan`); and a port with no live child has its bit of every plan's `held` cleared (held_clear) beside its
   `prepared` 0, where `prepared` alone held its plan.  With `port` NULL (region_adopt, the stop), every standing plan's
   receives cancelled and every port index no live child of this bridge holds (a reaped child's, a port not found yet,
   or a dead bridge's children's at the adoption) cleared from every plan's `held`, as `prepared` was cleared for every
   port with no child.  What must stay: the signature (mesh-children.c, a lifecycle file, calls it); a live child's
   holding its own to clear (plan_end), or its reap's.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5): every
   standing plan read at its place (mesh-bridge.h plan_from), where plan_after walked the heads.  What must stay: as
   above.
   2026-11-06 (design/aot-compilation-change-request.md A7, corrected 2026-11-06: "Every member of a view plans on the
   view's map"; "The revisions of 14b9a757", A7): the lost peer's map index its place in ascending node-id order among
   the installed view's records (view_read: mesh-map.h mesh_view_records, hdr `view_nodes`), the index the plans'
   entries name, where it was its place in this node's own map; every plan where the view is not read, as where the
   map was not.  A plan of another view was cancelled whole as the view was installed (plan_stale), and a raise of a
   count at its final value leaves it there (count_raise).  What must stay: as above.
   design/problems-of-14b9a757-change-request.md item 11 (E. A. Lee and S. Ha's self-timed strategy, as S. S.
   Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §2, state it: "retain the
   processor assignment and actor ordering specified by the fully static schedule ... Each processor executes the
   actors assigned to it in the order specified at compile time"; the plan's processors are its cables): each plan
   whose lists name the lost cable's peer, the places toward its index (mesh.h mesh_algo_adjacent), cancelled whole
   (plan_cancel), as a view change cancels it (plan_stale), where its receives from the peer alone were.  The index
   names the operand's entries only in the view it was read in (plans_load), so the operand's plans are cancelled by it
   only where the operand's view is that one.  Every child of the port is spawned after this call (mesh-ports.c
   port_lost and mesh-children.c child_reaped call it before mesh-ports.c port_open's child_spawn) */
static struct cancelled { uint32_t node,index,every; struct mesh_view_id view; } cancelled[MESH_PORTS];
void streams_cancel(const struct port *port){
  struct mesh_state *words;
  if(!port){
    for(uint32_t k=0;(words=plan_from(&k));k++)plan_cancel(words);
    uint64_t live=0;
    for(uint32_t p=0;p<port_count;p++){
      if(ports[p].child)live|=UINT64_C(1)<<ports[p].index;
      else atomic_store_explicit(&mesh_links(M)[ports[p].index].port.prepared,0,memory_order_release);
    }
    held_clear(~live);
    return;
  }
  struct cancelled *last=cancelled+port->index;
  if(port->peer){
    struct mesh_record *records=NULL;uint64_t count=0;
    struct mesh_view_id view={0};
    const int error=view_read(&records,&count,&view);
    *last=(struct cancelled){.node=port->peer,.index=error?UINT32_MAX:map_index(records,count,port->peer),
                             .every=error!=0,.view=view};
    free(records);
  }
  if(last->node && last->every)for(uint32_t k=0;(words=plan_from(&k));k++)plan_cancel(words);
  else if(last->node && last->index!=UINT32_MAX && mesh_algo_open(M) &&
          !mesh_view_compare(mesh_plan_view(mesh_algo_state(M,mesh_algo(M),0)),last->view)){
    uint32_t listed;
    const uint32_t *adjacent=mesh_algo_adjacent(M,last->index,&listed);
    for(uint32_t j=0;j<listed;j++)if((words=mesh_algo_standing(M,adjacent[j])))plan_cancel(words);
  }
  if(!port->child){
    atomic_store_explicit(&mesh_links(M)[port->index].port.prepared,0,memory_order_release);
    held_clear(UINT64_C(1)<<port->index);
  }
}
/* 2026-11-04, owner B1 (design/compilation-coordination.md, "The sequencer's messages": "`void streams_let_go(void)`
   goes (retirement closes a gone client's claims, and the ports end its plans)"; e6bd7db2's section 6): streams_let_go
   (tag pre-return-20261103 rdma/mesh-link.c:712-722, reduced: where no client held the region, hdr `client` 0, the plan
   written had its receives cancelled) is removed, with its declaration (mesh-bridge.h) and its call in mesh-records.c
   clients_turn (their owner's).  Why: hdr `client` is never written from 110 (L3); a gone client is retired
   (mesh-dataflow.h mesh_retire closes its claims and raises `plans`), and each port ends each plan whose claim closed,
   raising its receives' counts not landed (plans_walk, plan_end).  What must stay: no consumer reads an execution of a
   closed plan as whole */
/* A plan stands only while its map does (design/disjoint-pages-change-request.md, D.2 and D.3.6; RFC 2328 section 16:
   routes come from one version of the database, and a change recomputes them; design/disjoint-pages-rulings-
   coordination.md, R2), in the bridge parent, after a write of the map that moved hdr `map_serial` (mesh-records.c
   map_put and map_adopt, after the write's closing notification): a seq_cst fence; then, where a plan stands (`hops`
   above 0, acquire; `plan` its generation) and its entries' `serial` (the first entry's, read after `hops`: every
   entry carries the plan's one serial, mesh.h struct mesh_hop) is not `map_serial` as this thread reads it now (the
   parent is the map's one writer), every receive of it not landed cancelled: plan_cancel of every entry, that
   generation alone, each an arrival on its count raised to its final value, no phase written.  Every node's bridge
   hears the map's move within one flooding (mesh-records.c record_heard, map_put; RFC 2328 section 13), so every node
   of the plan reads its cancellation then, not its deadline, a relay's downstream nodes included, with no case for a
   relay.  The cancellation of the lost peer's receives at the cable's two ends (streams_cancel) stays.
   The fence pairs with the client's start (mesh-dataflow.c mesh_transfers_start; the contract's R3): the start raises
   the plan's `executions`, fences (seq_cst) and reads `map_serial`; this side has moved `map_serial`, fences, then
   reads `hops` and, in plan_cancel, `executions`.  Of the two fences one comes first, and a load after the later one
   sees the store before the earlier one, so at least one side sees the other: this cancellation covers the raised
   executions, or the start reads the moved serial, returns ESTALE and its program runs none of them.  It is D10's
   announcement, fence and re-read (design/disjoint-pages-revision-coordination.md; K. Fraser, Practical lock-freedom,
   2004), and nobody waits.  The fence comes before `hops` is read, not only before `executions`, so a plan whose `hops`
   its client stored after its last map read and before this move is covered alike: its `hops` stored before its
   start's fence is seen here, or its start sees the move.  The plan read whole, as plan_cancel reads it: `hops` and
   `plan`, the serial, an acquire fence, then both again; where either moved, the plan read is no longer the one
   written (its ports end it, link_close cancelling what has not landed) and a plan written after it is seen here or
   reads the move at its start, by the same pairing.  plan_cancel is given the generation judged here, so a plan
   written between this check and its walk, whose map may stand, is left as it is.
   Its thread: map_put and map_adopt run on the bridge parent's one event-loop thread, mesh-link.c main's, alone
   (plan_cancel's note lists their callers), so no release of a retired claim runs inside the walk, and a count word
   computed from an entry the walk found current lands in pages its plan's claim still holds.
   2026-11-04, sub-assignment 1: new (design/disjoint-pages-change-request.md, D.3.6), in place of nothing: before it
   only the lost cable's two ends cancelled, and a relay's downstream node read ETIMEDOUT at its deadline.  What must
   stay: the fence after the serial's move and before `executions` is read; plan_cancel's whole-plan re-read; this call
   on the parent's one thread.
   2026-11-04, owner B1 (design/disjoint-pages-rulings-coordination.md, R2; design/compilation-coordination.md, Q5;
   e6bd7db2's section 6: "plan_stale (R2): a fence, a walk, then every plan whose serial !=
   map_serial is cancelled"): a seq_cst fence, `map_serial` read, then a walk of the standing plans (plan_after), each
   whose state section's `serial` is not it cancelled whole (plan_cancel, every receive), where it was the one plan
   written (hdr `hops`, `plan`, its first entry's `serial`) read whole around an acquire fence.  The fence still comes
   before every read of a plan, its head's publication among them: a plan published before its start's fence is seen
   standing here, or its start reads the moved serial (R3), by the pairing above.  The generation passed to plan_cancel
   and the re-read go: each plan is named by its state section, never rewritten while it stands.  What must stay: the
   fence after the serial's move and before any plan or its `executions` is read; this call on the parent's one
   thread.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5): the standing
   plans read at their places (mesh-bridge.h plan_from), where plan_after walked the heads.  What must stay: as
   above.
   2026-11-06 (design/aot-compilation-change-request.md A7, corrected 2026-11-06: "Every member of a view plans on the
   view's map ... A plan then stands while its view does, and its view while the sequencer's map does (D.3.6)";
   mesh-bridge.h plan_stale; "The revisions of 14b9a757", A7), reversing the map serial: a seq_cst fence, then the
   installed view's identifier read (mesh.h mesh_view_installed; the parent its one writer, under `view_map`), then
   every standing plan whose view (mesh_plan_view) is not it cancelled whole (plan_cancel, every receive and every edge
   into it), called by mesh-records.c view_install, after it stores the view, and by map_adopt, after it stores no
   view installed (the identifier 0, so every plan is cancelled: mesh-bridge.h plan_stale).  The fence pairs with the
   start as before, with the identifier as the serial (mesh-call.h mesh_transfers_start: the raise of `executions`, a
   seq_cst fence, then the identifier read, ESTALE where it is not the plan's): of the two fences one comes
   first, so this cancellation covers the raised executions or the start reads the new view.  What must stay: the
   fence after the view's store and before any plan or its `executions` is read; this call on the parent's one
   thread */
void plan_stale(void){
  atomic_thread_fence(memory_order_seq_cst);
  const struct mesh_view_id view=mesh_view_installed(M,memory_order_relaxed);
  struct mesh_state *words;
  for(uint32_t k=0;(words=plan_from(&k));k++)
    if(mesh_view_compare(mesh_plan_view(words),view))plan_cancel(words);
}
