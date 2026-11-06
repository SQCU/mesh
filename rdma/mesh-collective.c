#include "mesh-collective.h"
#include "mesh-graph.h"
#include <float.h>
#include <os/os_sync_wait_on_address.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* tag pre-return-20261103 rdma/mesh-collective.c, verbatim: every node computes the same trees, routes and programs
   from the same map, so no product is fused into a sum where one node's compiler might fuse it and another's not */
#pragma STDC FP_CONTRACT OFF

_Static_assert(MESH_NODES_MOST<=64,"a demand's destinations are one 64-bit set of map indices");

/* -- a host program's side of the plan's words ------------------------------------------------------------------- */

/* mesh-collective.h mesh_host_publish (design/disjoint-pages-change-request.md, C3.7; design/disjoint-pages-revision-
   coordination.md, "The region", `supplied`, and D9): execution + 1 stored (release) into the supplied word a source
   range's first row names (its publication's first target, which mesh_plan writes), after the caller's writes to the
   range in that execution.  One store releases every send of the range's pages, and the word only rises while the
   caller's executions run in order.  A row that is no source range's first names no target (`sends` 0), and nothing is
   stored.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's form (sub-assignment 5), MESH_SUPPLIED stored into execution t's
   own word at the target's word + t x MESH_WORD_STRIDE.  Why: one word serves every execution, its value saying which,
   and no word is reset while its plan stands (C3.7).  What must stay: one store, release, after the range's last write.
   2026-11-05 (design/aot-compilation-change-request.md, appendix item 7, "A publish waits for nothing": "the publish
   is done in that event's notification"; design/compilation-coordination.md, "The revisions of e4a7398e", TP-3): the
   word raised to execution + 1 where it is below it (a compare-and-swap loop, release, that never lowers it), where it
   was stored.  Why: a torch program's publishes run in torch MPSEvents' notifications (rdma/mesh-torch.mm), and
   "Notifications may come in any order, so the count only rises" (mesh fba41159), so execution t's may run after t + 1's;
   C3.7: "supplied words ... only rise".  The loop retries only where another publish of the range raised the word
   meanwhile, and ends once the word is at least execution + 1.  What must stay: one release store raising the word,
   after the range's last write.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
   "The revisions of 14b9a757", R-1), reversing "the supplied word a source range's first row names" and the test of its
   `sends`: the word entry `transfer` of plan `plan`'s `transfers` names (mesh.h struct mesh_state, mesh_transfer_at:
   `targets[0]`, which mesh_plan writes where this node is the transfer's source), read by index, the entry tested for
   nothing.  Why: mesh-collective.h, "A range's words at this node are named by its declared transfer's index, not by a
   row: ranges sharing a block begin in one row"; and the operator's "Compile-time, never runtime checks ... Trace a bad
   value one hop up to its producer and make it impossible at compile time": the transfer is the caller's, which names
   only the transfers its compilation sources.  What must stay: one release raise of the word, never lowering it, after
   the range's last write. */
void mesh_host_publish(struct mesh_ctx *context,uint64_t plan,uint32_t transfer,uint64_t execution){
  struct hdr *m=context->M;
  const struct mesh_publication *entry=mesh_transfer_at(m,(const struct mesh_state *)(const void *)((char *)m+plan),
                                                        transfer);
  _Atomic uint64_t *word=(_Atomic uint64_t *)(void *)((char *)m+entry->targets[0].word);
  uint64_t seen=atomic_load_explicit(word,memory_order_relaxed);
  while(seen<execution+1 &&
        !atomic_compare_exchange_weak_explicit(word,&seen,execution+1,memory_order_release,memory_order_relaxed));
}

/* mesh-collective.h mesh_host_arrived (C3.1, C3.7; the contract's "The region", the phases and `counts`, and D1):
   execution t of a received range, from its first row's publication (`pages` P, `phases` its copy-0 group, `count` its
   copy-0 count word, `state` its plan's struct mesh_state, whose `copies` is u).  Copy c = t mod u: its count word
   (count + 8 c) is read first (acquire), then its group's words (phases + c x mesh_group_bytes(P), 32 fields a word,
   acquire).  A landing XORs its field before it adds to the count (both release, and every write of a count is a
   read-modify-write), so a count read at (t / u + 1) x P makes every landing it counts visible to the reads after it:
   a field still one behind then is a cancelled receive, never a landing in flight.  1 where every field reads
   mesh_phase(t / u + 1); UINT64_MAX where one reads neither that nor mesh_phase(t / u) (a write ran ahead of the
   allocator's order), or where the count has reached (t / u + 1) x P with a field behind (cancelled); else 0.  0 where
   the row is no received range's first (`pages` 0).
   2026-11-04, sub-assignment 3: reversing 2026-11-03's form (sub-assignment 5), the range's bit run in execution t's
   landed record and its cancel word.  Why: the records and the cancel word are gone (C4); a landing is a phase, a
   cancellation an arrival on the count (C3.1, C3.7).  What must stay: 0 until the range has landed, then 1; UINT64_MAX
   once a receive of it was cancelled; a range read by its words, never a word a page.
   2026-11-06 (R-1; "The revisions of 14b9a757", R-1), reversing "from its first row's publication" and its tests of
   `pages` and `state`: the words entry `transfer` of plan `plan`'s `transfers` names (`pages`, `phases`, `count`, which
   mesh_plan writes where this node is the transfer's destination), and u from the plan's own section, `plan`, read by
   index, the entry tested for nothing.  Why: as mesh_host_publish's note above (R-1's interface; "Compile-time, never
   runtime checks"): the caller reads only the transfers its compilation receives.  What must stay: as above. */
uint64_t mesh_host_arrived(struct mesh_ctx *context,uint64_t plan,uint32_t transfer,uint64_t execution){
  struct hdr *m=context->M;
  const struct mesh_state *state=(const struct mesh_state *)(const void *)((char *)m+plan);
  const struct mesh_publication *entry=mesh_transfer_at(m,state,transfer);
  const uint64_t copies=state->copies,copy=execution%copies,round=execution/copies,pages=entry->pages;
  const uint32_t expected=mesh_phase(round+1),behind=mesh_phase(round);
  const uint64_t count=atomic_load_explicit((_Atomic uint64_t *)((char *)m+entry->count+8*copy),memory_order_acquire);
  _Atomic uint64_t *group=(_Atomic uint64_t *)((char *)m+entry->phases+copy*mesh_group_bytes(entry->pages));
  int landed=1;
  for(uint64_t w=0;w<(pages+31)/32;w++){
    const uint64_t word=atomic_load_explicit(group+w,memory_order_acquire);
    for(uint64_t k=w*32;k<pages && k<w*32+32;k++){
      const uint32_t field=(uint32_t)(word>>(2*(k%32))&3);
      if(field==expected)continue;
      if(field!=behind)return UINT64_MAX;
      landed=0;
    }
  }
  if(landed)return 1;
  return count>=(round+1)*pages?UINT64_MAX:0;
}

/* -- the plan call (mesh-collective.h mesh_plan) ------------------------------------------------------------------ */

/* tag pre-return-20261103 rdma/nccl-mesh.c MESH_SOLVE_EPS: the concurrent-flow solve's epsilon, its rounds and room
   the tag's mesh_solve and mesh_solve_room (rdma/mesh-collective.c) */
#define MESH_PLAN_EPS 0.1

/* a demand: the declared transfers of one identity from one source, one range to each of their destinations */
struct plan_demand {
  uint32_t source,identity,first;         /* first: its first declared transfer, which numbers it (GO5) */
  uint32_t mine;                          /* the index in `values` of this node's value of it (this node its source or
                                             a destination), else MESH_ABSENT */
  /* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7; "The
     revisions of 4e78f6f4"): `offset`, its first block in this node's value `mine` (struct mesh_declared `first`), so
     its pages are blocks [offset, offset + pages) of each copy of that value.  Why: point 7, "name, in each declared
     transfer, the pages of its destination value that it lands in, as NCCL's receives land at offsets of one result";
     design/plan-to-bridge-change-request.md R2, corrected: "A collective's receive lands in the pages of the value it
     belongs to, at its place in that value".  What must stay: `mine` as above
     2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
     "The revisions of 14b9a757", R-1), reversing `offset`: `view`, its view at this node of value `mine` (mesh-collective.h
     struct mesh_strided: its declared transfers' `from` at its source, its declared transfer's `to` at a destination;
     NULL elsewhere).  Why: R-1, "A transfer's source and destination are each a view (value, offset, sizes, strides)";
     mesh-collective.h struct mesh_declared, "`first` and `bytes` go: a range's place is its views".  What must stay:
     `mine` as above */
  const struct mesh_strided *view;
  uint32_t tree,trees;                    /* its trees, plan `tree` [tree, tree + trees) */
  uint32_t release;                       /* this node's release of it, its supplied word's index (struct mesh_state
                                             `supplied`), where this node is its source */
  /* its destinations (a set of map indices), its bytes B (its views', struct mesh_declared) and its instructions P.
     2026-11-06 (R-1; "The revisions of 14b9a757", R-1): `instructions` where `pages` was (ceil(B / Q) pages), the range's
     instructions (plan_instructions), the unit its trees, communications, tasks and fields count.  Why: R-1, "The
     bridge's instructions are those blocks (aot A9: MSCCL-IR's steps with offset and count)".  `cut`: its instructions'
     bounds, p->cut[cut .. cut + P], instruction k the range's bytes [cut[k], cut[k + 1]); `at`: where this node is its
     source or a destination, p->at[at + k] the byte of value `mine` at which instruction k begins; `relay`: where this
     node is a relay of it, its routed value's set here (plan_sets).  What must stay: its destinations and bytes */
  uint64_t leaves,bytes,instructions,cut,at;
  uint32_t relay;
  /* its part of the dependence template (plan_template) and its value here: */
  uint64_t on;                            /* the nodes its trees reach past its source (a set of map indices), the
                                             trees the fill gave pages alone: each has its W and L */
  uint32_t forwards;                      /* its distinct tree edges y -> c over those trees, each its F */
  uint32_t own;                           /* 1 where a dependence names its source's own read: it has S(d) */
  uint32_t set;                           /* 1 where its range is a set: every declared transfer of it MESH_PAGES
                                             (design/compilation-coordination.md, S1), each tree's pages one set */
  uint32_t vertex;                        /* its first template vertex, P(d); the rest follow (plan_landed) */
  uint32_t value;                         /* its routed value at this node, p->here[value]; MESH_ABSENT where this node
                                             holds none of its pages */
  /* this node's received range of it (plan_walk's count): g, in the plan's demand order (MESH_ABSENT where none of its
     pages lands here), and its pages that land here in one copy, P_g (its pages P at a destination; a relay's those of
     the trees it is on) */
  uint32_t range,received;
  /* its consumed range here: h, among the received ranges this node is a destination of, in the plan's demand order,
     its count words' row (MESH_ABSENT elsewhere: a relay's receives count nothing, review 15's finding 1) */
  uint32_t consumed;
  /* its count words' row here, among every received range's (plan_layout): `consumed` where this node consumes it,
     else the consumed ranges' number plus its place among the relayed ones, in the plan's demand order (MESH_ABSENT
     where none of its pages lands here; design/compilation-coordination.md, "The revisions of abecd389", RC) */
  uint32_t counted;
  uint64_t group;                         /* range g's copy-0 phase group, its offset in the phase list (plan_layout) */
};
/* A routed value (design/disjoint-pages-change-request.md, C3.3; design/disjoint-pages-revision-coordination.md, "The
   allocator"): demand `demand`'s pages at node `node`, its source or a node its trees reach (a destination or a relay),
   at every node, since u is the collective's.  Its write is P at the source, W(node) elsewhere; its reads
   p->read[read .. read + reads), F(node, c) for each c it forwards to, ascending, then C(node) at a destination, then
   S at the source where its demand has one (the source's own read): at least one (every leaf of a tree is a
   destination, so a relay forwards).  `source`: its first row among the least lags' sources (plan_unroll), its write,
   then its reads in turn. */
struct plan_value { uint32_t demand,node,write,read,reads,source; };
/* One of this node's hops of one copy (plan_walk; mesh.h struct mesh_hop), every copy's the same: routed value
   p->here[value]'s page `page`, sent to or received from `node`, of demand `demand`.  `field`: a receive's field in its
   range's group (its demand's receives here in walk order, 0 .. P_g - 1); a forwarding send's, its receive's of the
   same page; MESH_ABSENT for a send from the range's source, which its release's supplied word releases.
   `communication`: the page's crossing of the tree edge the hop is (plan_communication), one instruction of the
   plan's instruction DAG (plan_schedule; design/compilation-coordination.md, "The documents' revisions of edd4b583",
   GO1); `place`: that instruction's place in the global topological order (GO3), the entry's place in its list.
   `set` (S1, S2): j + 1 where its demand's range is a set and its page is on tree j (the solve's trees grouped by
   demand, p->tree's index), else 0.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): `at` (its group's least `place`,
   the place of the set's first page, GO4) is gone.  Why: design/plan-to-bridge-change-request.md R5, "A set's pages
   ... keep their places in that order.  The set's random permutation is drawn over those places.  (Corrected
   2026-11-05: the first text moved a set's pages to consecutive places, which nothing requires.)"  What must stay:
   every other field as plan_walk assigns it.
   2026-11-04, owner C3 (GO4): `walk` (the hop's place in plan_walk's order, T6's tie), `start` (its link's scheduled
   start of its page, T6), `earliest` (its group's least start) and `least` (its group's least walk, S3) are gone,
   `place` and `at` in their place.  Why: each cable's list is a subsequence of one global topological order built as
   MSCCLang §5.2 states in its step 3, and "Scheduled start times do not order the lists"
   (design/plan-to-bridge-change-request.md, R5), which overturns T6 and S3.  What must stay: every other field as
   plan_walk assigns it.
   2026-11-05, owner C3 (design/compilation-coordination.md, "The documents' revisions of edd4b583", "Amendments
   during the work (2026-11-05)", GO4′: "`walk` goes with T6's ties"): the removal of `walk` cited to the ruling.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
   revisions of 14b9a757", R-1): `instruction` where `page` was, its demand's instruction k (plan_instructions), and
   `value` gone: the hop's frame is the block holding its instruction at this node, a relay's set item or an end's
   value's row (mesh_plan), found from its demand.  Why: R-1, "The bridge's instructions are those blocks"; mesh.h struct
   mesh_hop, "Its `page` is the frame of the block holding the instruction".  What must stay: every other field as
   plan_walk assigns it */
struct plan_entry { uint32_t node,direction,demand,field,communication,set,place; uint64_t instruction; };
/* 2026-11-06 (R-1; design/plan-to-bridge-change-request.md R2 at mesh 56ede97e: "The share record of the
   implementation (`struct mesh_share`), and its rules, is a restricted form of the sharing stated above ... The record
   goes (R1)"; "Ranges may share a block where their bytes are disjoint"; "The revisions of 14b9a757", R-1): struct
   plan_offset (a value's covers, its ranges' block runs and the shares' runs, sorted to find two sharing a block) and
   plan_offset_order are removed.  Why: a range is its instructions' bytes in its value's blocks (plan_instructions), a
   block's item the set of ranges touching it (plan_sets), and neither a share nor "no two ranges at this node share a
   block of it" remains.  What must stay: nothing of it */
/* One synchronization edge of the program's IPC graph at one of the plan's actors here, as struct mesh_sync holds it
   (2026-11-05, point 4; plan_graph): `node`, `function`, `position`, `every`, `at` and `delay` as mesh.h states them.
   2026-11-05 (118; design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K1; "The
   revisions of 55b9cbad"): `node`, `function`, `entry`, `every` and `base` as mesh.h struct mesh_sync states them at
   118, `position`, `at` and `delay` folded into `base` and the waits' rows.  What must stay: struct mesh_sync's words
   but `count` */
struct plan_sync { uint32_t node,function,entry,every; int64_t base; };
struct plan {
  struct mesh_ctx *context;
  const struct mesh_declared *declared;
  uint32_t transfers;                     /* the declared transfers, declared[0 .. transfers) */
  struct mesh_value *given;               /* the caller's values, `givens` of them */
  uint32_t givens;
  /* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7; "The revisions
     of 4e78f6f4"): named[j] the place in `offset` of values[j]'s first range here, its others after it (`offsets` of
     them in all, sorted by value and first block), MESH_ABSENT where none lies in it (a local value), where it was the
     first declared transfer of the one demand whose range values[j] was.  Why: point 7, "no concatenation and no copy
     of a part remains"; R2, corrected: "A range is not a value of its own".  What must stay: MESH_ABSENT for a local
     value.
     2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V2 and G2;
     "The revisions of 55b9cbad"), reversing named[j]'s place: named[j] 1 where a range lies in values[j] here, 0 for
     a local value; cover[j] the place in `offset` of values[j]'s first cover (a range or a share, struct plan_offset),
     MESH_ABSENT where none; uncovered[j] its blocks a copy that no cover holds, its own item's (plan_items), counted
     where the covers are sorted (plan_function), where plan_items walked its ranges again; shared[j] 1 where values[j]
     is a struct mesh_share's `of`; `share`, `shares` the function's struct mesh_share.  Why: V2, "bind `value`'s
     shared blocks of copy c to the pages of `of`'s blocks of copy c ... so they are no part of `value`'s uncovered
     item"; G2, "the two walks of a value's ranges".  What must stay: 0 for a local value
     2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
     "The revisions of 14b9a757", R-1), reversing `cover`, `shared`, `share`, `shares`, `offset` and `offsets` (the
     covers and struct mesh_share, which go): `named` as above; `uncovered[j]` values[j]'s blocks a copy in its own item
     (every block of a local or a MESH_INVARIANT value; a named MESH_EXECUTION value's blocks some byte of which no
     instruction here covers, plan_sets).  The instructions (plan_instructions): `cut` every demand's bounds, `at` its
     first bytes in its value at this node's end (struct plan_demand `cut`, `at`).  The frames' items here (plan_sets):
     set s's members, the routed values here (p->here's indices) member[set_first[s] .. set_first[s + 1]), its
     `set_blocks[s]` blocks a copy, `set_far[s]` E_s, the largest farthest[a] over its members a, its copy c item
     set_item + s x u + c (plan_items); a named MESH_EXECUTION value j's block k at index block_first[j] + k of
     `block_set` (its set, MESH_ABSENT where it stays in the value's own item) and `block_element` (its element in that
     item), block_first[j] MESH_ABSENT for every other value.  `toward`: the map indices this node's entries name (a set;
     plan_walk), mesh.h struct mesh_algo `adjacent`'s.  Why: R-1, "A frame's item lives from the first write to the last
     read over every value naming it, with C3.3's lag test"; R-6, "Each cable's processor executes its list ... It scans
     no table of plans".  What must stay: `named` 0 for a local value */
  unsigned char *named;
  uint32_t *uncovered;
  uint64_t *cut,*at,toward;
  uint32_t sets,set_item,*set_first,*member,*set_blocks,*set_far,*block_first,*block_set,*block_element;
  /* the function's operands (struct mesh_function): its weights and dependences, A10's bound */
  const struct mesh_weight *weights;
  const struct mesh_dependence *dependence;
  uint32_t dependences;
  struct plan_demand *demand;             /* the demands, by their first declared transfer (GO5) */
  uint32_t demands,n,self,copies;         /* copies: u (plan_unroll) */
  uint64_t serial;                        /* mesh_nodes_read's: the installed view's identifier (2026-11-06, aot A7,
                                             corrected; "The revisions of 14b9a757", A7), where it was the map serial */
  uint32_t *of;                           /* of[i]: declared[i]'s demand, p->demand[of[i]] */
  /* the node graph (design/compilation-coordination.md, T2): pair e joins from[e] < to[e], of capacity[e], the lesser
     of the two directions' map capacities; `pairs` of them */
  uint32_t pairs,from[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1],to[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1];
  double capacity[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1];
  size_t *tree;                           /* tree[j] the solve's tree, demand by demand (mesh_graph_rows) */
  uint32_t *parent;                       /* the solve's parents, tree by tree */
  uint64_t *run;                          /* run[j]: tree[j]'s pages of its demand */
  uint32_t made;                          /* the solve's trees, j < made */
  /* each tree's communications (plan_trees): members[j] its vertices but its root (a set of map indices), lead[j] its
     demand's first page on it, comm[j] its first communication; `comms` of them in all */
  uint64_t *members,*lead,*comm,comms;
  uint64_t releases;                      /* this node's releases, the demands it is the source of */
  /* this node's hops of one copy, every copy's in this order (plan_walk): */
  struct plan_entry *entry;               /* entry[0 .. entries), NULL while plan_walk counts */
  uint64_t entries,receives;              /* their count (n of mesh.h's u x n) and their receives */
  uint32_t ranges,consumed;               /* this node's received ranges, and those it consumes (a destination's) */
  /* The dependence template (plan_template), one execution's, every rank's the same from the same declared transfers,
     dependences and map; each edge carries the execution distance `lag` from its tail's execution to its head's. */
  uint64_t *child;                        /* child[i x n + y]: the nodes y sends demand i's pages to over its trees (a
                                             set of map indices; 0 for a node off them) */
  struct plan_value *value;               /* every node's routed values, demand by demand, a demand's by node */
  uint32_t values;
  uint32_t *read;                         /* their read vertices (struct plan_value `read`, `reads`) */
  uint32_t *here,local;                   /* this node's routed values, here[0 .. local), p->value's, in demand order */
  uint32_t vertices;                      /* the template's vertices, demand by demand from each one's `vertex` */
  size_t edges;                           /* the template's edges */
  size_t *first;                          /* vertex v's edges, in compressed rows: [first[v], first[v + 1]),
                                             vertices + 1 entries (mesh_graph_rows by tail) */
  uint32_t *head,*lag;                    /* edge k's head vertex, and its execution distance */
  uint32_t *between;                      /* between[a x local + b]: D_ab, the largest least lag from a read of this
                                             node's routed value a to b's write (plan_unroll) */
  uint32_t *farthest;                     /* farthest[w]: E_w, the largest D_wz over this node's routed values z
                                             (UINT32_MAX where any is; plan_unroll, plan_before) */
  /* the state section (plan_layout), offsets from its start: `hops` its entries', `words` its end (2026-11-05, "The
     revisions of 46ae8fdf", LP9: `sequences`, the sequence words', gone).  2026-11-06 (R-1; "The revisions of
     14b9a757"): `publications`, the declared transfers' entries (mesh.h struct mesh_state `transfers`).  What must stay:
     every other offset */
  uint64_t phases,counts,supplied,publications,status,hops,words;
  /* 2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS and PQ): the plan's two queues in
     the section, `deliveries` (room `room` struct mesh_delivery) and `submits` (room `outstanding` struct
     mesh_submission), between the status and the entries.
     2026-11-05 ("The revisions of 3f106747", BS): the submissions' queue's room `submit_room`, and `room` and it the
     queues' self-timed buffer bounds, `outstanding` A10's bound alone.
     2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 4; "The revisions
     of 4e78f6f4"): the plan's synchronization edges at this node, `waits` into its actors here then `tells` out of
     them (`sync`, struct mesh_sync at `syncs` in the section), where it held one edge's source and sink map indices and
     `wait_delay`; `calls` its function's calls in the program's traced block; `time` t(x_v) of its actor at each node v
     and `nodes` its nodes in the program's IPC graph (plan_graph).  Why: point 4, "build the program's whole IPC graph
     at compile time, every function together, with each queue's room (5) in it"; mesh.h struct mesh_state's 117 note.
     What must stay: `room` and `submit_room` the queues' self-timed buffer bounds, `outstanding` A10's bound.
     2026-11-05 (118; design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K1;
     "The revisions of 55b9cbad"): `rows`, the waits' rows' part of the section (plan_layout), and `wait_row`, its
     `calls` + 1 words, row a's first wait the waits' a-th (mesh.h struct mesh_state `rows`).  What must stay: as
     above */
  uint64_t deliveries,submits,syncs,rows,room,outstanding,submit_room,spanned,nodes,forwarded;
  uint32_t waits,tells,calls,*wait_row;
  struct plan_sync *sync;
  double time[MESH_NODES_MOST];
  /* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 5 and 6; "The
     revisions of 4e78f6f4"): the plan's items among the program's (plan_place): routed copy (a, c) item routed + a x u
     + c; values[j]'s first item[j] (a local value's copies, or a named value's uncovered blocks, one item a copy;
     MESH_ABSENT where a named value has none); the state section item `state`; `base` the section's region offset,
     where the items were numbered among this plan's alone, `colour` their first blocks, `window` and `outside` each
     colouring's blocks and `claim` the claim of them.  Why: point 5, "place the whole program's values in one offline
     allocation at compile time.  These go: `mesh_claim_take`'s first fit and the claim's head record"; point 6, "Place
     every value's pages anywhere, by the fewest chains of the interference order".  What must stay: every value's,
     every routed copy's and the state section's items
     2026-11-06 (R-1; "The revisions of 14b9a757"): `routed` goes for `set_item` (above): a routed copy is a relay's set,
     and a range at its ends lies in its value's blocks.  What must stay: every value's and the state section's items */
  uint32_t *item,state;
  uint64_t base;
};

struct plan_key { uint32_t source,identity,index; };
static int plan_key_order(const void *a,const void *b){
  const struct plan_key *x=a,*y=b;
  if(x->source!=y->source)return x->source<y->source?-1:1;
  if(x->identity!=y->identity)return x->identity<y->identity?-1:1;
  return (x->index>y->index)-(x->index<y->index);
}
/* 2026-11-04, owner C (design/compilation-coordination.md, T6): plan_demand_order (lower `order` first, a transfer
   that blocks a pipeline stage, then the declared order) is removed with struct mesh_declared's `order`.  Why: the
   order of every node pair's sends is the static schedule's (design/plan-to-bridge-change-request.md, R5;
   plan_schedule), not a hand priority.  What must stay: the demands numbered alike at every rank, now by their first
   declared transfer (mesh_plan). */
/* mesh_graph_reach's edges: the node graph's pair e, its ends from[e] and to[e] */
struct plan_edges { const uint32_t *from,*to; };
static uint32_t plan_end(const void *context,uint32_t e,uint32_t side){
  const struct plan_edges *graph=context;
  return side?graph->to[e]:graph->from[e];
}
/* mesh_graph_flow's leaves: demand d's destinations */
static int plan_leaf(const void *context,uint32_t d,uint32_t v){
  return (int)(((const struct plan_demand *)context)[d].leaves>>v&1);
}
/* the caller's array's word k: mesh_graph_rows' keys (a solved tree's demand, its trees grouped by demand; a call's
   function, the program's calls grouped by function; an IPC edge's tail) and mesh_graph_order's heads (plan_acyclic's
   lag-0 rows).  2026-11-05 ("The revisions of 4e78f6f4"): one function, where plan_tree_demand, plan_head_of and
   plan_room_tail were three copies of it (the operator's "Refactor by pattern analysis").  What must stay: word k */
static uint32_t plan_at(const void *context,size_t k){ return ((const uint32_t *)context)[k]; }
/* mesh_graph_rows' keys: a vertex's parent on a tree, so a vertex's row its children; the root and a vertex off the
   tree key `n`, past every vertex */
struct plan_tree { const uint32_t *parent; uint32_t n,root; };
static uint32_t plan_tree_parent(const void *context,size_t v){
  const struct plan_tree *tree=context;
  return v==tree->root || tree->parent[v]>=tree->n?tree->n:tree->parent[v];
}
/* mesh_graph_climb's visit: the vertex one step up, a vertex's parent */
static void plan_tree_up(void *consumer,uint32_t at,uint32_t step){ if(step==1)*(uint32_t *)consumer=at; }
/* a block's bytes, Q = block x pgsz, and a value's blocks: `bytes` in whole blocks of Q bytes */
static uint64_t plan_quantum(const struct hdr *m){ return (uint64_t)m->block*m->pgsz; }
static uint64_t plan_blocks(const struct hdr *m,uint64_t bytes){
  const uint64_t quantum=plan_quantum(m);
  return (bytes+quantum-1)/quantum;
}
/* values[j]'s copies: u where a range lies in it, k for a value of lag k >= 1, 1 for every other (2026-11-05,
   design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G2: "the copies expression
   (1947, 2026)", one function where mesh_plan wrote it twice and plan_items once more; "The revisions of 55b9cbad").
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
   design/plan-to-bridge-change-request.md R2, corrected 2026-11-06: "Any value a transfer names is sent from its own
   pages, whatever its copies"; "The revisions of 14b9a757", R-1): a value a range names here 1 where it is
   MESH_INVARIANT (C3.2: "placed once, for the plan's whole life"; mesh-collective.h: "one copy for the plan's life,
   which every copy's send reads") and u otherwise, a value of `lag` k >= 1 among them ("whose copies are then the plan's
   u").  What must stay: one function */
static uint32_t plan_copies(const struct plan *p,uint32_t j){
  if(p->named[j])return p->given[j].kind==MESH_INVARIANT?1:p->copies;
  return p->given[j].lag?p->given[j].lag:1;
}

/* -- the dependence template (design/disjoint-pages-revision-coordination.md, "The allocator") -------------------- */

/* v's place among a set's members (map indices below 64) */
static uint32_t plan_rank(uint64_t set,uint32_t v){ return (uint32_t)__builtin_popcountll(set&((UINT64_C(1)<<v)-1)); }
/* Demand i's template vertices, from its `vertex`: P(i), its production at its source (an edge into it is before its
   first write, an edge out of it after its last); for each node y of `on` in ascending order, W(i, y) its first
   landing at y and L(i, y) its last; for each tree edge y -> c in (y, c) order, F(i, y, c), y's last forward to c; for
   each destination z in ascending order, C(i, z), z's consumption (an edge into it is before its first read, an edge
   out of it after its last); last, where its `own` is 1, S(i), its source's own read of it (the same).
   2026-11-04, sub-assignment 3 (the coordinator's ruling on 13's and 14's finding; design/disjoint-pages-revision-
   coordination.md, "The allocator"): S(i) added.  Why: a source's local read of the range it sends was in no event,
   its value's reads its forwards alone, so the allocator could let a later landing reuse its pages while the source
   still read them (in e2b, a peer's exchange k + 1 landing in the pages of this node's exchange-k partial as its own
   combine reads it).  S(i) exists only where a dependence names it: a read with no edge out has no unroll bound.
   What must stay: every other vertex's place, so the offsets below hold. */
static uint32_t plan_produced(const struct plan *p,uint32_t i){ return p->demand[i].vertex; }
static uint32_t plan_landed(const struct plan *p,uint32_t i,uint32_t y,int last){
  return p->demand[i].vertex+1+2*plan_rank(p->demand[i].on,y)+(uint32_t)!!last;
}
static uint32_t plan_forwarded(const struct plan *p,uint32_t i,uint32_t y,uint32_t c){
  const uint64_t *child=p->child+(size_t)i*p->n;
  uint32_t at=p->demand[i].vertex+1+2*(uint32_t)__builtin_popcountll(p->demand[i].on);
  for(uint32_t x=0;x<y;x++)at+=(uint32_t)__builtin_popcountll(child[x]);
  return at+plan_rank(child[y],c);
}
static uint32_t plan_consumed(const struct plan *p,uint32_t i,uint32_t z){
  const struct plan_demand *d=p->demand+i;
  return d->vertex+1+2*(uint32_t)__builtin_popcountll(d->on)+d->forwards+plan_rank(d->leaves,z);
}
static uint32_t plan_own(const struct plan *p,uint32_t i){
  const struct plan_demand *d=p->demand+i;
  return d->vertex+1+2*(uint32_t)__builtin_popcountll(d->on)+d->forwards+(uint32_t)__builtin_popcountll(d->leaves);
}
/* a program event (mesh-collective.h struct mesh_dependence), from its end's bits (`produced` MESH_FROM_PRODUCED or
   MESH_TO_PRODUCED, `own` MESH_FROM_SOURCE_READ or MESH_TO_SOURCE_READ): declared[i]'s production at its source, P of
   its demand; its source's own read, S of its demand; or its consumption at its destination, C of its demand there */
static uint32_t plan_event(const struct plan *p,uint32_t i,int produced,int own){
  if(produced)return plan_produced(p,p->of[i]);
  if(own)return plan_own(p,p->of[i]);
  return plan_consumed(p,p->of[i],p->declared[i].destination);
}
/* the node an event is at: the source for a production or the source's own read, else the destination */
static uint32_t plan_event_node(const struct mesh_declared *declared,uint32_t i,int produced,int own){
  return produced || own?declared[i].source:declared[i].destination;
}
/* a template edge before its row is known (mesh_graph_rows' records, keyed by tail) */
struct plan_edge { uint32_t tail,head,lag; };
static uint32_t plan_edge_tail(const void *context,size_t k){ return ((const struct plan_edge *)context)[k].tail; }

/* The dependence template and the routed values (design/disjoint-pages-change-request.md, C3.3;
   design/disjoint-pages-revision-coordination.md, "The allocator"), after the routing.
   - Each demand's trees, the ones the fill gave pages (a tree of none carries nothing): each vertex's children on
     them, from its row of the tree's vertices keyed by parent (mesh_graph_rows, as plan_walk reads them; no walk over
     a parent table here, C9), make `child` and `on`.
   - The template's edges, all of lag 0 but the program's: P -> W(y) for every y of `on`; P -> F(source, c); W(y) ->
     L(y); W(y) -> F(y, c); F(y, c) -> L(c); L(z) -> C(z); P -> S where the demand has S; and each dependence e one
     edge from its `from` event's vertex to its `to` event's (plan_event: P of the demand where its produced bit is
     set, S where its source-read bit is, else C at the declared transfer's destination), of lag e.lag.  A demand has S
     where a dependence's end names its source's own read.  Every edge is sound: its tail happens before its head
     however the pages travel (a range is produced before it is sent and before its source reads it; a page lands before
     it is forwarded, is sent before it lands, and lands before it is consumed; the program's edges are its own
     claims).  Grouped into rows by tail (mesh_graph_rows), so every rank
     numbers them alike.
   - The routed values, at every node, demand by demand and a demand's by ascending node: its source's, and each node
     of `on`'s (a destination's or a relay's), each with its write and reads (struct plan_value); this node's among them
     (`here`, and each demand's `value`).  Every node's, since u is the largest L over the collective's values
     (design/disjoint-pages-change-request.md, C3.3: "The unroll factor u is the largest L"), which every rank computes
     alike.
   - Every walk of this template into a write passes the write's range's production P: a landing's write W(y) has one
     edge in, P -> W(y), and the edges into P are the program's dependences alone.  That is the half of
     design/compilation-coordination.md, U1, which this template holds (plan_unroll).
   2026-11-04, owner C2 (design/compilation-coordination.md, U1): u's citation, the contract's D5, is C3.3's.  Why: D5
     appears in neither Lam, Rau nor C3.3's text (design/aot-compilation-change-request.md, appendix item 1).  What
     must stay: u every node's routed values', so every rank computes the same u.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's (sub-assignment 3's) program edges, one kind, C(consumed) ->
   P(produced), and its values at this node alone.  Why: the contract's struct mesh_dependence names either event of
   either declared transfer (four kinds), and u needs every node's values.  What must stay: the transport's edges as
   they were, every edge sound, every rank's template the same.
   0, or ENOMEM. */
static int plan_template(struct plan *p,const struct mesh_dependence *dependence,uint32_t dependences){
  const uint32_t n=p->n,self=p->self;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  p->child=calloc((size_t)p->demands*n+1,sizeof *p->child);
  if(!p->child)return ENOMEM;
  uint64_t vertices=0,edges=dependences,values=0,reads=0;
  for(uint32_t k=0;k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    if(x->kinds&MESH_FROM_SOURCE_READ)p->demand[p->of[x->from]].own=1;
    if(x->kinds&MESH_TO_SOURCE_READ)p->demand[p->of[x->to]].own=1;
  }
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    uint64_t *below=p->child+(size_t)i*n;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      if(!p->run[j])continue;
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      for(uint32_t y=0;y<n;y++)for(size_t c=first[y];c<first[y+1];c++)below[y]|=UINT64_C(1)<<child[c];
    }
    d->on=0;d->forwards=0;
    for(uint32_t y=0;y<n;y++){d->on|=below[y];d->forwards+=(uint32_t)__builtin_popcountll(below[y]);}
    const uint64_t on=(uint64_t)__builtin_popcountll(d->on),leaves=(uint64_t)__builtin_popcountll(d->leaves);
    d->vertex=(uint32_t)vertices;
    vertices+=1+2*on+d->forwards+leaves+d->own;
    edges+=2*on+2*(uint64_t)d->forwards+leaves+d->own;
    values+=1+on;
    reads+=d->forwards+leaves+d->own;
    if(vertices>=UINT32_MAX || reads+values>=UINT32_MAX)return ENOMEM;
  }
  p->vertices=(uint32_t)vertices;
  p->edges=(size_t)edges;
  p->value=malloc(((size_t)values+1)*sizeof *p->value);
  p->read=malloc(((size_t)reads+1)*sizeof *p->read);
  p->here=malloc(((size_t)p->demands+1)*sizeof *p->here);
  struct plan_edge *edge=malloc((p->edges+1)*sizeof *edge);
  size_t *order=malloc((p->edges+1)*sizeof *order);
  p->first=malloc(((size_t)p->vertices+1)*sizeof *p->first);
  p->head=malloc((p->edges+1)*sizeof *p->head);
  p->lag=malloc((p->edges+1)*sizeof *p->lag);
  if(!p->value || !p->read || !p->here || !edge || !order || !p->first || !p->head || !p->lag){
    free(edge);free(order);return ENOMEM;
  }
  size_t e=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    const uint64_t *below=p->child+(size_t)i*n;
    const uint32_t produced=plan_produced(p,i);
    for(uint64_t s=d->on;s;s&=s-1){
      const uint32_t y=(uint32_t)__builtin_ctzll(s);
      edge[e++]=(struct plan_edge){produced,plan_landed(p,i,y,0),0};
      edge[e++]=(struct plan_edge){plan_landed(p,i,y,0),plan_landed(p,i,y,1),0};
    }
    for(uint32_t y=0;y<n;y++)for(uint64_t s=below[y];s;s&=s-1){
      const uint32_t c=(uint32_t)__builtin_ctzll(s),forward=plan_forwarded(p,i,y,c);
      edge[e++]=(struct plan_edge){y==d->source?produced:plan_landed(p,i,y,0),forward,0};
      edge[e++]=(struct plan_edge){forward,plan_landed(p,i,c,1),0};
    }
    for(uint64_t s=d->leaves;s;s&=s-1){
      const uint32_t z=(uint32_t)__builtin_ctzll(s);
      edge[e++]=(struct plan_edge){plan_landed(p,i,z,1),plan_consumed(p,i,z),0};
    }
    if(d->own)edge[e++]=(struct plan_edge){produced,plan_own(p,i),0};
  }
  for(uint32_t k=0;k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    edge[e++]=(struct plan_edge){
      plan_event(p,x->from,!!(x->kinds&MESH_FROM_PRODUCED),!!(x->kinds&MESH_FROM_SOURCE_READ)),
      plan_event(p,x->to,!!(x->kinds&MESH_TO_PRODUCED),!!(x->kinds&MESH_TO_SOURCE_READ)),x->lag};
  }
  mesh_graph_rows(p->vertices,p->edges,plan_edge_tail,edge,p->first,order);
  for(size_t k=0;k<p->edges;k++){p->head[k]=edge[order[k]].head;p->lag[k]=edge[order[k]].lag;}
  free(edge);free(order);
  /* the routed values, every node's */
  uint32_t at=0,source=0;
  p->values=0;p->local=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    d->value=MESH_ABSENT;
    for(uint64_t s=d->on|UINT64_C(1)<<d->source;s;s&=s-1){
      const uint32_t x=(uint32_t)__builtin_ctzll(s);
      struct plan_value *v=p->value+p->values;
      *v=(struct plan_value){.demand=i,.node=x,.write=x==d->source?plan_produced(p,i):plan_landed(p,i,x,0),.read=at,
                             .source=source};
      for(uint64_t c=p->child[(size_t)i*n+x];c;c&=c-1)p->read[at++]=plan_forwarded(p,i,x,(uint32_t)__builtin_ctzll(c));
      if(d->leaves>>x&1)p->read[at++]=plan_consumed(p,i,x);
      if(x==d->source && d->own)p->read[at++]=plan_own(p,i);
      v->reads=at-v->read;
      source+=1+v->reads;
      if(x==self){d->value=p->local;p->here[p->local++]=p->values;}
      p->values++;
    }
  }
  return 0;
}

/* -- the one allocator (design/disjoint-pages-change-request.md, C3.2-C3.4; the contract's "The allocator") -------- */

/* mesh_graph_lags' and mesh_graph_order's edges: the template's, edge k's head and execution distance; and the lag-0
   edges' heads alone (plan_acyclic's rows, plan_at) */
static uint32_t plan_head(const void *context,size_t k){ return ((const struct plan *)context)->head[k]; }
static uint32_t plan_lag(const void *context,size_t k){ return ((const struct plan *)context)->lag[k]; }

/* No lag-0 cycle (the contract's "The plan call": EINVAL for a lag-0 cycle): the template's lag-0 edges, in rows by
   tail as the template's, have a topological order (mesh-graph.h mesh_graph_order; its EDEADLK is EINVAL), as
   2026-11-03's mesh_graph_happens_before refused one.  The template's own edges are acyclic (each demand's run down
   from P), so a cycle is the program's: one execution's events each before the next and the last before the first.
   The contract asks for D(x -> x) == 0 on any routed vertex; this tests every vertex at once, since a least lag shows
   a cycle only at a vertex both a source and a target of mesh_graph_lags (whose targets are the writes), and a cycle
   of consumptions alone passes no write (owner 2's remark, routed by the coordinator).  With it, every D(w -> w) and
   every L(v) below is at least 1.  0; EINVAL; ENOMEM. */
static int plan_acyclic(const struct plan *p){
  size_t *first=malloc(((size_t)p->vertices+1)*sizeof *first);
  uint32_t *head=malloc((p->edges+1)*sizeof *head),*indegree=malloc(((size_t)p->vertices+1)*sizeof *indegree),
           *ordinal=malloc(((size_t)p->vertices+1)*sizeof *ordinal);
  int status=!first || !head || !indegree || !ordinal?ENOMEM:0;
  if(!status){
    size_t e=0;
    for(uint32_t v=0;v<p->vertices;v++){
      first[v]=e;
      for(size_t k=p->first[v];k<p->first[v+1];k++)if(!p->lag[k])head[e++]=p->head[k];
    }
    first[p->vertices]=e;
    status=mesh_graph_order(p->vertices,first,plan_at,head,indegree,ordinal)?EINVAL:0;
  }
  free(first);free(head);free(indegree);free(ordinal);
  return status;
}

/* a routed value named on stderr (the contract's "The plan call", ENOMEM): its range's identity and source, its node,
   and here its index in `values` (or that it is this node's relay); the least lag found */
static void plan_name(const struct plan *p,uint32_t v,const char *what,uint32_t lag){
  const struct plan_value *x=p->value+v;
  const struct plan_demand *d=p->demand+x->demand;
  char here[48]="";
  if(x->node==p->self){
    if(d->mine!=MESH_ABSENT)snprintf(here,sizeof here," (values[%u] here)",d->mine);
    else snprintf(here,sizeof here," (this node's relay)");
  }
  if(lag==UINT32_MAX)
    fprintf(stderr,"mesh_plan: the value of range %u from rank %u at rank %u%s: %s (no walk)\n",d->identity,d->source,
            x->node,here,what);
  else fprintf(stderr,"mesh_plan: the value of range %u from rank %u at rank %u%s: %s (least lag %u)\n",d->identity,
               d->source,x->node,here,what,lag);
}

/* Unrolling by modulo variable expansion (design/disjoint-pages-change-request.md, C3.3: M. Lam, Software pipelining,
   PLDI 1988; B. R. Rau, Iterative modulo scheduling, MICRO 1994; design/compilation-coordination.md, U1, and
   design/disjoint-pages-revision-coordination.md, "The allocator").  The least lags D(x -> y) over the template
   (mesh-graph.h mesh_graph_lags, Dijkstra with the lags as weights), from every routed value's write and reads, at
   every node, to every routed value's write:
   - L(v) = the largest D(r -> write v) over v's reads r, the least lag at which every read of v in execution t happens
     before its write in t + L (C3.3: "For each value, L is the smallest lag at which every read of it in execution t
     happens before its write in execution t + L"); UINT32_MAX (a read that reaches no later write of it) is ENOMEM
     naming the value (C3.3: "A value with no such L fails the plan with ENOMEM, naming the value"), and nothing else
     is;
   - u = the largest L(v) over every routed value at every node, 1 where there is none (C3.3: "The unroll factor u is
     the largest L").  Every rank computes the same template, lags and u, so both nodes of every adjacent pair hold the
     same copies;
   - between[a x local + b] = D_ab, the largest D(r -> write b) over this node's routed value a's reads r, for every
     pair of this node's routed values, so the order's tests (plan_before) are table reads; and farthest[a] = E_a, the
     largest D_ab over b (UINT32_MAX where any is).
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 6; "The revisions
   of 4e78f6f4"): farthest added, the order's last clause's operand (plan_before), where between was the colouring's
   (plan_interferes, gone).  What must stay: between as it is.
   The least lags are C3.3's happens-before at every larger distance too (U1): every walk of the template into a write
   passes its range's production P (plan_template, the plan's own); and, the premise the least lags rest on, a property
   of the program's dependences and not of the plan (mesh-collective.h struct mesh_dependence), every event of a node
   in execution t precedes each of its events in t + 1 by the program's own edges, as both programs give them
   (tools/e2b_replay.m e2b_ir's rank chains, "r's last to its first at lag 1"; rdma/mesh_compile.py _dependences,
   "every event no lag-0 edge leaves to every event none enters (lag 1)"), so a production reaches its next instance
   at lag 1 and a walk at lag L through P continues through P's next instances at every lag past L.  This function
   does not test the premise (design/aot-compilation-change-request.md, appendix item 1: ENOMEM refuses only a value
   with no such L).  D(w -> w) == 0, a lag-0 cycle, plan_acyclic has refused.
   2026-11-04, owner C2 (design/compilation-coordination.md, U1; design/aot-compilation-change-request.md, appendix
   item 1): the refusal of a write whose least cycle D(w -> w) is not 1 (design/disjoint-pages-revision-coordination.md,
   D5), ENOMEM naming its value, is gone.  Why: C3.3 fails only a value with no such L, and D5 appears in neither Lam,
   Rau nor C3.3's text; it refused a lone AllGather or AllToAll, whose landing reaches its next instance at lag 2 (u 2
   now).  What must stay: L(v), u, between and plan_interferes as they were; ENOMEM, named, for a read that reaches no
   later write.
   0; ENOMEM. */
static int plan_unroll(struct plan *p){
  const uint32_t values=p->values;
  p->copies=1;
  if(!values)return 0;
  const struct plan_value *last=p->value+values-1;
  const uint32_t sources=last->source+1+last->reads;
  size_t cells,pairs;
  if(__builtin_mul_overflow((size_t)sources,(size_t)values,&cells) || cells>=SIZE_MAX/sizeof(uint32_t) ||
     __builtin_mul_overflow((size_t)p->local,(size_t)p->local,&pairs) || pairs>=SIZE_MAX/sizeof(uint32_t))return ENOMEM;
  uint32_t *source=malloc((size_t)sources*sizeof *source),*target=malloc((size_t)values*sizeof *target),
           *lags=malloc((cells+1)*sizeof *lags);
  p->between=malloc((pairs+1)*sizeof *p->between);
  p->farthest=calloc((size_t)p->local+1,sizeof *p->farthest);
  int status=!source || !target || !lags || !p->between || !p->farthest?ENOMEM:0;
  for(uint32_t v=0;!status && v<values;v++){
    const struct plan_value *x=p->value+v;
    source[x->source]=x->write;
    for(uint32_t j=0;j<x->reads;j++)source[x->source+1+j]=p->read[x->read+j];
    target[v]=x->write;
  }
  if(!status)status=mesh_graph_lags(p->vertices,p->first,plan_head,plan_lag,p,sources,source,values,target,lags);
  for(uint32_t v=0;!status && v<values;v++){
    const struct plan_value *x=p->value+v;
    uint32_t most=0;
    for(uint32_t j=0;j<x->reads;j++){
      const uint32_t lag=lags[((size_t)x->source+1+j)*values+v];
      if(lag>most)most=lag;
    }
    if(most==UINT32_MAX){plan_name(p,v,"a read of it happens before no later write of it",most);status=ENOMEM;break;}
    if(most>p->copies)p->copies=most;
  }
  for(uint32_t a=0;!status && a<p->local;a++)for(uint32_t b=0;b<p->local;b++){
    const struct plan_value *x=p->value+p->here[a];
    uint32_t most=0;
    for(uint32_t j=0;j<x->reads;j++){
      const uint32_t lag=lags[((size_t)x->source+1+j)*values+p->here[b]];
      if(lag>most)most=lag;
    }
    p->between[(size_t)a*p->local+b]=most;
    if(most>p->farthest[a])p->farthest[a]=most;
  }
  free(source);free(target);free(lags);
  return status;
}

/* The state section's layout (design/disjoint-pages-change-request.md, C3.2 and C3.7; the contract's "The region"),
   offsets from its start, each part 64-byte aligned, its total whole blocks (plan_place sizes it):
   - 0: struct mesh_state (64 bytes);
   - `phases` (64): the phase list, received range g's copy c group at phases + group_g + c x mesh_group_bytes(P_g),
     group_g the sum of u x mesh_group_bytes(P_g') over g' < g (groups in (g, c) order, g major; each whole lanes, so a
     lane's MESH_LANE_BYTES never spans two groups); page k of the range (k < P_g) at bits [2 (k % 32), 2 (k % 32) + 2)
     of uint64 word k / 32 of its group;
   - `counts`: consumed x u uint64 words, consumed range h's copy c at index h x u + c: a count word for each range
     this node is a destination of, none for a relay's receives, which no consumer here waits on and whose forwards
     their phases release (review 15's finding 1, the coordinator's ruling); struct mesh_state `ranges` is `consumed`,
     the counts' rows, while the phase list holds a group for every received range, a relay's among them;
     2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", RC): ranges x u words, every
     received range's row its `counted` (the consumed ranges' first, at their `consumed`, then the relayed ones), where
     a relay's receives had none.  Why: design/aot-compilation-change-request.md, appendix item 5: "The code must name
     a page for round j by that count, in place of PM2's test of the phase field", the count "each range's landed
     count of each copy, which every landing raises"; a relay's receive is named for its rounds as a destination's is
     (mesh-flow.c link_receive_drain).  What must stay: a consumed range's row its `consumed`, `ranges` its number;
   - `supplied`: releases uint64 words, release r's at index r;
   - `status`: one struct mesh_status;
   - `deliveries`: the plan's delivered calls' queue here, `room` struct mesh_delivery, and `submits`, the program's
     submissions' queue, `outstanding` struct mesh_submission (mesh.h struct mesh_state's notes; 2026-11-05, "The
     revisions of abecd389", PQ and PS: design/aot-compilation-change-request.md, appendix item 2, "It is laid out with
     the plan's pages when the plan is placed, and claimed and released with them (C3.2, C3.4)");
   - `syncs`: the plan's synchronization edges at this node, `waits` then `tells` struct mesh_sync (mesh.h; plan_graph);
     2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 4; "The
     revisions of 4e78f6f4"): added between the submissions' queue and the entries.  Why: mesh.h struct mesh_state's
     117 note, "`syncs` the region offset of their struct mesh_sync entries, the `waits` first, in the state section";
     a plan has an actor for each call of its function, and Convert-to-SC-graph adds at most one edge into each actor
     and one out of it.  What must stay: every offset before the submissions' queue's end as it was;
   - `rows` (2026-11-05, 118; design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines",
     K1; "The revisions of 55b9cbad"): `calls` + 1 uint32_t words, the waits' rows by the call place they enter (mesh.h
     struct mesh_state `rows`), after the synchronization edges and before the entries.  Why: K1, "the waits are
     stored in rows keyed by call place".  What must stay: every offset before it where it was;
   - `hops`: this node's entries, u x n struct mesh_hop, copy c's n at [c n, c n + n) (design/compilation-
     coordination.md, "The region", struct mesh_state: "the entries; each 64-aligned"); [0, hops) the bytes mesh_plan
     starts (D3), the entries written whole after them;
   - `words`: the end of the entries, the section's bytes.
   Every offset in struct mesh_state is a region offset: the section's own plus these.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP9): reversing DL1's sequence words
   (u uint64 words after the status, L. Lamport's "Message Buffer" B[0 .. u - 1]): the entries follow the status
   again, and struct mesh_state `sequences` is 0.  Why: design/aot-compilation-change-request.md, appendix item 2 at
   mesh 0e56d8da: "publish them to each port through the channel above, in place of the sequence words (`sequences`),
   `taken[p]` and the parent's wait while they are full (Lamport's buffer; the coordination's DL1)"; "How the bridge
   holds the channel is the bridge's own".  What must stay: every offset before the status's end as it was; every part
   64-aligned.
   2026-11-04, owner C3 (DL1): the sequence words placed after the status and before the entries, where the entries
   followed the status.  Why: the parent's delivery publishes each released execution's sequence number in its plan's
   state, one word per copy, each word one writer (appendix item 2, "Delivery at a node").  What must stay: every
   offset before the sequence words as it was; every part 64-aligned.
   2026-11-04, owner C (design/compilation-coordination.md, "The region" and Q5): the entries in the section, where they
   were hdr's hop area (`hop_off`, room `hop_capacity`).  Why: any number of plans stand at once, each in its own claim
   (design/plan-to-bridge-change-request.md, R8).  What must stay: the words' offsets before the entries as they
   were.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; mesh.h
   struct mesh_state `transfers`; "The revisions of 14b9a757", R-1): `publications`, one struct mesh_publication (64
   bytes) for each of the function's declared transfers in their order, after the supplied words and before the status,
   so inside the bytes the GPU's transport wraps (mesh-metal.m mesh_metal_transport_create: the struct to the greatest
   end of its phases, counts, supplied words and status, which every section has; mesh-metal.h: "entry `transfer` of
   the plan's `transfers` ... which lies in the plan's state section, the transport's buffer").  Why: mesh.h, "the
   region offset of one struct mesh_publication (64 bytes) for each of the function's declared transfers, in their
   order (mesh_transfer_at), each written once by mesh_plan, read by index".  What must stay: every part 64-aligned, the
   struct, the phases, the counts and the supplied words where they were */
static uint64_t plan_align(uint64_t at){ return (at+63)/64*64; }
static void plan_layout(struct plan *p){
  uint64_t at=sizeof(struct mesh_state);
  p->phases=at;
  at=0;
  uint32_t relayed=p->consumed;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    d->counted=d->range==MESH_ABSENT?MESH_ABSENT:d->consumed!=MESH_ABSENT?d->consumed:relayed++;
    if(d->range==MESH_ABSENT)continue;
    d->group=at;
    at+=(uint64_t)p->copies*mesh_group_bytes(d->received);
  }
  p->counts=plan_align(p->phases+at);
  p->supplied=plan_align(p->counts+(uint64_t)p->ranges*p->copies*sizeof(uint64_t));
  p->publications=plan_align(p->supplied+p->releases*sizeof(uint64_t));
  p->status=plan_align(p->publications+(uint64_t)p->transfers*sizeof(struct mesh_publication));
  p->deliveries=plan_align(p->status+sizeof(struct mesh_status));
  p->submits=plan_align(p->deliveries+p->room*sizeof(struct mesh_delivery));
  p->syncs=plan_align(p->submits+p->submit_room*sizeof(struct mesh_submission));
  p->rows=plan_align(p->syncs+((uint64_t)p->waits+p->tells)*sizeof(struct mesh_sync));
  p->hops=plan_align(p->rows+((uint64_t)p->calls+1)*sizeof(uint32_t));
  p->words=p->hops+p->entries*p->copies*sizeof(struct mesh_hop);
}
/* demand d's words here, region offsets in its plan's placed section (`base`; the layout above): its release's
   supplied word, its received range's copy-c phase group, and its copy-c count word (2026-11-06, "The revisions of
   14b9a757", item 6 of the brief: "Look for any other duplicate the change leaves in mesh-collective.c and merge it";
   the operator's "merge duplicated patterns found by analysis"): one form each, where the entries and the ranges' words
   wrote each expression twice.  What must stay: the layout's offsets */
static uint64_t plan_supplied(const struct plan *p,const struct plan_demand *d){
  return p->base+p->supplied+(uint64_t)d->release*sizeof(uint64_t);
}
static uint64_t plan_group(const struct plan *p,const struct plan_demand *d,uint32_t c){
  return p->base+p->phases+d->group+(uint64_t)c*mesh_group_bytes(d->received);
}
static uint64_t plan_count(const struct plan *p,const struct plan_demand *d,uint32_t c){
  return p->base+p->counts+((uint64_t)d->counted*p->copies+c)*sizeof(uint64_t);
}

/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 6: "Place every
   value's pages anywhere, by the fewest chains of the interference order ... Greedy by size goes, with
   `mesh_graph_left_edge`'s colouring"; "The revisions of 4e78f6f4"): plan_sized_order (greedy by size's order, an
   item's blocks largest first), plan_interferes (mesh_graph_left_edge's interference), plan_item_page and the
   plan's own plan_place (two greedy-by-size colourings of one plan's items, their ENOSPC as ENOMEM, its pages the
   claim's runs) are removed.  Why: design/plan-to-bridge-change-request.md R2, corrected: "No value is placed as one
   run.  So greedy by size (Pisarchyk and Lee), and the contiguous-block problem it approximates (M. R. Garey and D. S.
   Johnson, Computers and Intractability, 1979, problem SR2), have no use here.  The chains replace C3.3's colouring";
   point 5: "place the whole program's values in one offline allocation at compile time".  What must stay: the items
   (each routed value's u copies, a relay's among them, each other value, a value of lag k >= 1 its k copies, the state
   section), the least lags' tests at the distances the copies' executions meet, ENOMEM before anything is written. */

/* An item of the program's one allocation (plan_place): `blocks` elements, each a block (point 6: "Every item is
   counted in blocks"; mesh-graph.h mesh_graph_chains' elements), the first `element` among the chains' elements, in
   space `space` (1 the registered window, 0 outside it: mesh-dataflow.h mesh_arena_range); `kind` the order it takes
   part in (plan_before), of plan `function` (UINT32_MAX: the program's table), `index` its routed value here
   (PLAN_ROUTED) or its value in `values` (PLAN_LOCAL, PLAN_ALONE), `copy` its copy */
enum { PLAN_ALONE, PLAN_LOCAL, PLAN_ROUTED };
struct plan_item { uint64_t element,blocks; uint32_t kind,function,index,copy,space; };
/* the program's items (`items`, the window's first, [0, split), then the rest), their blocks (mesh_graph_chains'
   `count`), each element's chain, each space's arena range (`range[space]`) and the program's table's item */
struct plan_placement { struct plan_item *item; uint64_t *count,*chain,items,elements; uint32_t split,table;
                        struct mesh_range range[2]; };

/* The interference order, mesh_graph_chains' operand: before(x, y) where x's every access happens before y's first
   write at every distance their executions meet (design/plan-to-bridge-change-request.md R2, corrected: "Two items
   may share a page exactly when one's last access happens before the other's first write.  That is C3.3's own
   interference test, over Lamport's happens-before (L. Lamport, Time, clocks, and the ordering of events in a
   distributed system, CACM 21(7), 1978).  That relation is a partial order"; design/disjoint-pages-change-request.md
   C3.3, corrected: "Two items may share a page exactly when this interference test orders them").  The tests are
   plan_interferes' as they were, each pair that shared now ordered one way (2026-11-05, design/compilation-
   coordination.md, "The documents' corrections of 2026-11-05", point 6; "The revisions of 4e78f6f4"):
   - items of different plans are unordered: no edge of either template orders them;
   - PLAN_ALONE, unordered with every item, as plan_interferes had them interfering with every item: an invariant (the
     state sections and the table among them); a copy of a value of lag k >= 1, whose write an event of the program's
     orders, a host producer's, not an edge of the template (design/disjoint-pages-change-request.md, D.3.3;
     design/disjoint-pages-rulings-coordination.md, R4); and a named value's blocks no range covers (point 7; mesh-
     collective.h struct mesh_declared `first`: "no edge of the template orders their write"); and (2026-11-05, V2,
     "The revisions of 55b9cbad") every item of a value that a struct mesh_share's `of` names, its routed copies and
     its own item (plan_items): the shared blocks' item then lives the program's whole life, as an invariant's does,
     which covers both values' accesses at the cost of pages (PB R2, corrected: "Those pages are one item of the
     allocation, living until the later of the two values' last reads"; mesh-collective.h struct mesh_share: "whose
     lifetime covers both values' accesses (mesh-collective.c's minimal form is the implementer's, recorded at the
     code)").  The implementer's decision, recorded: every item of `of` (a superset of those its shared blocks lie in),
     so no second walk of `of`'s covers finds them; a tighter form, adding `value`'s readers to the item's reads, would
     need reads the template does not hold for a share.  Its premise is U1's: `value`'s reads of the shared blocks in
     execution t precede `of`'s write in t + u where that write is this node's own (a production or a local value's);
     a share whose `of` blocks land here is not ordered so by any edge the plan holds;
   - two local execution values of lag 0 of one plan (PLAN_LOCAL): x before y where x's `end` is below y's `begin`,
     their positions on this node's program order in one execution, this node's work running in that order (and, by
     U1's premise, each node's events of execution t before those of t + 1, so the next execution's write of x
     follows y's reads); a local execution value and a routed copy are unordered;
   - two routed copies x = (v, c) and y = (w, c') of one plan (PLAN_ROUTED), D_vw = between[v x local + w], E_w =
     farthest[w], u the copies: x before y where c <= c', c' - c >= D_vw (and, where c == c', D_vw == 0 and v != w:
     y's write in x's execution follows x's every read), and c' + E_w <= u.  The last clause is the document's
     (design/plan-to-bridge-change-request.md R2, "(Corrected 2026-11-05: design/compilation-coordination.md, "The
     revisions of 4e78f6f4", PO-1.)  Over C3.3's u copies the order needs one more clause": "Copies (v, c) and (w, c′)
     are ordered where c ≤ c′, c′ − c is at least the least lag from v to w, and c′ plus w's largest least lag to a
     write at the node is at most u.  That keeps the relation transitive", modulo variable expansion's premise (M. Lam,
     Software pipelining, PLDI 1988) and a lifetime wrapping the period a circular arc (L. J. Hendren, G. R. Gao, E. R.
     Altman and C. Mukerji, A register allocation framework based on hierarchical cyclic interval graphs, CC 1992)),
     where this note called it the coordinating implementer's (2026-11-05, design/compilation-coordination.md,
     "Answered in the documents", PO-1's text; "The revisions of 55b9cbad").  It is the cyclic case of R2's relation
     over one execution: a page holding a chain x_1 < ... < x_k is written again by x_1's
     next instance, in execution c_1 + u, which must follow x_k's reads in execution c_k, and c_k + E_{x_k} <= u <=
     c_1 + u gives it.  With it the relation is transitive: for x < y < z, every read of v reaches z's write within
     D_vy + D_yz (through y's write, which precedes y's reads at lag 0), so D_vz <= D_vy + D_yz <= c_z - c_x, and c_z
     + E_z <= u holds from y < z; and irreflexive (c == c' asks v != w, and x < y < x would be a lag-0 cycle, which
     plan_acyclic refuses).  Dilworth's theorem needs both (mesh-graph.h: "a strict partial order the caller gives").
     Within a chain each pair of neighbours is so ordered, and happens-before is transitive, so every earlier item's
     accesses on a page precede each later write of it.
   The least lags decide each test at every larger distance of its class (U1; plan_unroll).
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1: "A
   frame's item lives from the first write to the last read over every value naming it, with C3.3's lag test"; "The
   revisions of 14b9a757", R-1), generalizing the routed copies' clause to the frames' set items (plan_sets), where a
   routed copy is a set of one member, a relay's: copies x = (X, c) and y = (Y, c') of one plan (PLAN_ROUTED, `index`
   the set), X and Y sets of this node's routed values (a range's at its source, a destination or a relay), x before y
   where all three hold:
   - c <= c' and c' - c >= D_XY, the largest between[a x local + b] over a in X and b in Y (every read of every member of
     X in copy c's execution precedes every write of every member of Y in c''s: the item's accesses are its members');
   - c' + E_Y <= u, E_Y the largest farthest[b] over b in Y (`set_far`), the document's cyclic clause for the item's
     last members;
   - c < c' or X != Y.
   PO-1's argument, restated for sets.  Transitive: for x < y < z, every read of a in X reaches the write of each b in
   Y within D_ab <= D_XY, b's write precedes b's reads at lag 0 (the template's P -> F and S, W -> F and L -> C), and
   b's read reaches the write of each z' in Z within D_bz' <= D_YZ, so D_az' <= D_ab + D_bz' and D_XZ <= D_XY + D_YZ <=
   c_z - c_x (Y has a member); c_z + E_Z <= u holds from y < z.  And c_x < c_z or X != Z: were c_x = c_z and X = Z,
   then c_x = c_y = c_z, D_XY = D_YX = 0, and some a in X would have D_aa <= D_ab + D_ba = 0, a lag-0 cycle, which
   plan_acyclic refuses; that also makes it irreflexive (c == c' asks X != Y) and so a strict partial order, as
   Dilworth's theorem and mesh-graph.h mesh_graph_chains need ("a strict partial order the caller gives").  A set of
   one member whose blocks are its range's whole blocks is the routed copy above.  PLAN_ALONE's items of a struct
   mesh_share's `of` go with the record (PB R2 at mesh 56ede97e: "The record goes"); a block holding bytes no
   instruction here covers is its value's own item, PLAN_ALONE as an uncovered block was.  What must stay: PLAN_ALONE
   and PLAN_LOCAL otherwise as above, and items of different plans or kinds unordered */
struct plan_order { const struct plan *plans; const struct plan_item *item; };
static int plan_before(const void *context,uint32_t a,uint32_t b){
  const struct plan_order *order=context;
  const struct plan_item *x=order->item+a,*y=order->item+b;
  if(x->kind!=y->kind || x->function!=y->function || x->kind==PLAN_ALONE)return 0;
  const struct plan *p=order->plans+x->function;
  if(x->kind==PLAN_LOCAL)return p->given[x->index].end<p->given[y->index].begin;
  const uint64_t u=p->copies,c=x->copy,later=y->copy;
  if(c>later || later+p->set_far[y->index]>u)return 0;
  uint64_t most=0;
  for(uint32_t i=p->set_first[x->index];i<p->set_first[x->index+1];i++)
    for(uint32_t k=p->set_first[y->index];k<p->set_first[y->index+1];k++){
      const uint32_t lag=p->between[(size_t)p->member[i]*p->local+p->member[k]];
      if(lag>most)most=lag;
    }
  if(later-c<most)return 0;
  return c<later || x->index!=y->index;
}

/* an item appended (`item` NULL: counted alone) */
static void plan_item_put(struct plan_placement *x,uint32_t kind,uint32_t function,uint32_t index,uint32_t copy,
                          uint64_t blocks,uint32_t space){
  if(x->item){
    x->item[x->items]=(struct plan_item){x->elements,blocks,kind,function,index,copy,space};
    x->count[x->items]=blocks;
  }
  x->items++;x->elements+=blocks;
}
/* The program's items of space `space`, plan by plan (point 6: "The items, per function, at this node, as now"): its
   routed copies (a, c), a major, each its demand's P blocks, a relay's among them (this node on a demand's trees but
   neither its source nor a destination: no `values` entry names it and no row holds it), in the window; then its
   values in `values`' order: a named value's blocks no range covers, one item a copy, in the window (the value is
   wire); a local value's one item, or a value of lag k >= 1's k, its copies, in the window where `wire` is 1; then its
   state section (plan_layout's `words` in blocks), outside; last the program's table, `count` words in blocks,
   outside, once (mesh.h struct mesh_algo `table`).  Every item has a block at least.  The plan's `routed`, `item`
   and `state` and the table's index are each item's place in this order, rewritten alike by every pass.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V2 and G2; "The
   revisions of 55b9cbad"): a value's own item, a copy each (plan_copies), holds its blocks no cover holds, no range
   and no struct mesh_share (`uncovered`, counted where plan_function sorts the covers, where this walked the value's
   ranges again), in its `wire` space (a named value's 1, mesh-collective.h struct mesh_declared), none where every
   block is covered; one form for a named and a local value, the kind as before; and every item of a value a share's
   `of` names (`shared`), its routed copies and its own item, PLAN_ALONE (plan_before, V2).  Why: G2, "the two walks of
   a value's ranges ... merge each into one form"; V2, "bind `value`'s shared blocks of copy c to the pages of `of`'s
   blocks of copy c ... so they are no part of `value`'s uncovered item".  What must stay: the items' order above
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1 and
   R-6; "The revisions of 14b9a757", R-1 and R-6), reversing the routed copies and `shared`:
   - first, the plan's sets' items (plan_sets), s major, each copy c of set s one item of its `set_blocks[s]` blocks,
     PLAN_ROUTED, in the window (a relay's routed copy, or the blocks of its values a set of ranges here covers whole);
   - a value's own item as above, in the window where a range here names it whatever its `wire` (mesh-collective.h:
     "Every value a transfer names here is placed in the registered window, whatever its `wire`"), else in its `wire`
     space; a named value's PLAN_ALONE;
   - the program's table grows by mesh.h struct mesh_algo `adjacent`'s words (R-6: "Place it in the table's item, which
     grows by those words"): n + 1 uint32_t offsets and the plans' places toward each map index, `toward`'s, after the
     `count` words (plan_table).
   Why: R-1, "Two values' rows may name one frame ... A frame's item lives from the first write to the last read over
   every value naming it"; mesh-collective.h, "a block holding bytes no range here covers is an item that lives the
   program's whole life, as before".  What must stay: the items' order above, every pass alike */
/* the program's table's bytes (mesh.h struct mesh_algo `table`, `adjacent`): `count` words, then `nodes` + 1 uint32_t
   offsets and `places` places */
static uint64_t plan_table(uint32_t count,uint32_t nodes,uint64_t places){
  return (uint64_t)count*sizeof(uint64_t)+((uint64_t)nodes+1+places)*sizeof(uint32_t);
}
static void plan_items(struct plan *plans,uint32_t count,const struct hdr *m,struct plan_placement *x,uint32_t space){
  uint64_t places=0;
  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    const uint32_t u=p->copies;
    places+=(uint64_t)__builtin_popcountll(p->toward);
    if(space){
      p->set_item=(uint32_t)x->items;
      for(uint32_t s=0;s<p->sets;s++)for(uint32_t c=0;c<u;c++)plan_item_put(x,PLAN_ROUTED,f,s,c,p->set_blocks[s],1);
    }
    for(uint32_t j=0;j<p->givens;j++){
      const struct mesh_value *v=p->given+j;
      if((p->named[j]?1u:v->wire)!=space)continue;
      const uint32_t kind=!p->named[j] && v->kind==MESH_EXECUTION && !v->lag?PLAN_LOCAL:PLAN_ALONE;
      p->item[j]=p->uncovered[j]?(uint32_t)x->items:MESH_ABSENT;
      for(uint32_t c=0;p->uncovered[j] && c<plan_copies(p,j);c++)plan_item_put(x,kind,f,j,c,p->uncovered[j],space);
    }
    if(!space){p->state=(uint32_t)x->items;plan_item_put(x,PLAN_ALONE,f,0,0,plan_blocks(m,p->words),0);}
  }
  if(!space){
    x->table=(uint32_t)x->items;
    plan_item_put(x,PLAN_ALONE,UINT32_MAX,0,0,plan_blocks(m,plan_table(count,plans[0].n,places)),0);
  }
}

/* The program's one allocation (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points
   5 and 6; design/plan-to-bridge-change-request.md R2 and R8, corrected: "One offline allocation over the whole
   exported program ..., at compile time ... Nothing takes pages at run time, and no handle is looked up at run time";
   design/disjoint-pages-change-request.md A3, C3.3 and C3.4, corrected; "The revisions of 4e78f6f4"): every item of
   every plan at this node and the program's table (plan_items), decided once, after every plan's least lags and
   layout and before anything is written.
   - One call of mesh_graph_chains a space (mesh-graph.h: Dilworth's fewest chains by Fulkerson's bipartite matching),
     the window's items then the others', each over the interference order (plan_before), every item counted in
     blocks.  Where the two spaces are one range (an undeclared window is the whole arena, and mesh-dataflow.h
     mesh_arena_range then gives "one range over it"), one call over every item.
   - Chain j of a space is its block j: element k of item i at arena page range.first + chain x block (plan_page).  An
     item ordered with no other has chains of its own, consecutive (mesh_graph_chains' numbering), so a state section's
     and the table's blocks are consecutive and each is at its first block's page.
   - ENOMEM where a space's chains pass its blocks (mesh_arena_range(m, space).count / block), or the items pass their
     words, before anything is written.  No first fit, no claim and no ENOMEM from fragmentation (point 5).
   0; ENOMEM (mesh_graph_chains' too). */
static int plan_place(struct plan *plans,uint32_t count,struct hdr *m,struct plan_placement *x){
  *x=(struct plan_placement){0};
  plan_items(plans,count,m,x,1);
  plan_items(plans,count,m,x,0);
  if(x->items>=UINT32_MAX || x->elements>=SIZE_MAX/sizeof(uint64_t))return ENOMEM;
  x->item=malloc(((size_t)x->items+1)*sizeof *x->item);
  x->count=malloc(((size_t)x->items+1)*sizeof *x->count);
  x->chain=malloc(((size_t)x->elements+1)*sizeof *x->chain);
  if(!x->item || !x->count || !x->chain)return ENOMEM;
  x->items=x->elements=0;
  plan_items(plans,count,m,x,1);
  x->split=(uint32_t)x->items;
  plan_items(plans,count,m,x,0);
  x->range[1]=mesh_arena_range(m,1);
  x->range[0]=mesh_arena_range(m,0);
  const int one=x->range[0].first==x->range[1].first && x->range[0].count==x->range[1].count;
  const uint32_t first[3]={0,one?(uint32_t)x->items:x->split,(uint32_t)x->items};
  for(uint32_t s=0;s<2;s++){
    if(first[s]==first[s+1])continue;
    const struct plan_order order={plans,x->item+first[s]};
    uint64_t chains=0;
    const int status=mesh_graph_chains(first[s+1]-first[s],x->count+first[s],plan_before,&order,
                                       x->chain+x->item[first[s]].element,&chains);
    if(status)return status;
    if(chains>x->range[x->item[first[s]].space].count/m->block)return ENOMEM;
  }
  return 0;
}
/* item i's element k's first arena page: its space's first page and its chain's block (point 6: "Chain j of a space is
   that space's block j"); a routed copy's a block inside the registered window */
static uint32_t plan_page(const struct plan_placement *x,uint32_t block,uint32_t i,uint64_t k){
  return (uint32_t)(x->range[x->item[i].space].first+x->chain[x->item[i].element+k]*block);
}

/* Every row's page and every value's `shares`, decided from the placement before anything is written (2026-11-05,
   design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G2, V2 and K4; "The
   revisions of 55b9cbad"); mesh_plan then binds each row to its page (mesh_backing_bind), as a decider's pages handed
   to an effector ("Deciders see pages, effectors see pointers").  page[r] is row r's arena page, `rows` of them, the
   program's values' rows as mesh_plan numbered them (struct mesh_section `first`, `count`, `stride`).
   - The walk (G2: "the two walks of a value's ranges (mesh-collective.c 770, 2039) ... merge each into one form"):
     copy c of values[j], its covers in block order (struct plan_offset, from cover[j]); a range's blocks the routed
     copy c's of its demand here, block k - first (point 7); a share's run none yet (MESH_ABSENT); every other block
     its own item's next (item[j] + c, plan_items), where mesh_plan walked the ranges to bind them and plan_items to
     count them.
   - The shares (V2, PB R2 corrected: "the result's rows for the rank's own part name the operand's own pages"; struct
     mesh_share): each copy c's run of `value` bound to the pages of `of`'s blocks [at, at + blocks) of copy c, so no
     copy of the part is made.  The implementer's decision, recorded (the header is silent): EINVAL where a share's
     `of` blocks are themselves a share's run (no page named for them by the walk), tested for every share before any
     is bound, so the answer is the same in any order of the shares; a chain of shares would need its order, and a
     cycle of them names no page.
   - K4 ("the plan knows which blocks two values share and where one hands over to the next"): `shares` 1 where a
     page bound in one of a value's rows is bound in another row at this node (mesh-collective.h struct mesh_value
     `shares`), from one count of each page's rows over every row and one pass over each value's rows, no search per
     row.  A page's count is kept by its block, page / block: every page placed is a block's first (plan_page, a
     space's first page plus a chain's block), the window's blocks [0, window) and the rest's from the window's end,
     so no two blocks placed share page / block, below mesh_blocks(m).
   0; EINVAL; ENOMEM
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
   revisions of 14b9a757", R-1), reversing the covers' walk and the shares' passes with their EINVAL: block k of copy c
   of values[j] is its block's set item's element (copy c of set block_set[block_first[j] + k], element
   block_element[...]: plan_sets) where a set of ranges here covers it whole, else its own item's (item[j] + c), its
   element block_element[...] for a named MESH_EXECUTION value and k for every other (plan_frame).  Why: R-1, "A value
   is its page-table rows, and two values' rows may name one frame"; PB R2 at mesh 56ede97e, "A value's rows naming
   another value's frames is the page table's own sharing, with no record and no rules of its own".  K4's `shares` is
   computed as above, over these bindings.  What must stay: every row's page decided before anything is written;
   `shares` as above.  0; ENOMEM */
static uint32_t plan_frame(const struct plan *p,const struct plan_placement *x,uint32_t block,uint32_t j,uint32_t c,
                           uint32_t k){
  if(p->block_first[j]==MESH_ABSENT)return plan_page(x,block,p->item[j]+c,k);
  const uint32_t at=p->block_first[j]+k,s=p->block_set[at];
  return plan_page(x,block,s==MESH_ABSENT?p->item[j]+c:p->set_item+s*p->copies+c,p->block_element[at]);
}
static int plan_rows(struct plan *plans,uint32_t count,const struct hdr *m,const struct plan_placement *x,
                     uint32_t *page,uint64_t rows){
  const uint32_t block=m->block;
  for(uint32_t f=0;f<count;f++){
    const struct plan *p=plans+f;
    for(uint32_t j=0;j<p->givens;j++){
      const struct mesh_section v=p->given[j].section;
      for(uint32_t c=0;c<v.count;c++)for(uint32_t k=0;k<v.stride;k++)
        page[v.first+(size_t)c*v.stride+k]=plan_frame(p,x,block,j,c,k);
    }
  }
  unsigned char *seen=calloc((size_t)mesh_blocks(m)+1,1);
  if(!seen)return ENOMEM;
  for(uint64_t r=0;r<rows;r++)if(seen[page[r]/block]<2)seen[page[r]/block]++;
  for(uint32_t f=0;f<count;f++)for(uint32_t j=0;j<plans[f].givens;j++){
    struct mesh_value *v=plans[f].given+j;
    const uint64_t end=v->section.first+(uint64_t)v->section.count*v->section.stride;
    v->shares=0;
    for(uint64_t r=v->section.first;r<end && !v->shares;r++)v->shares=seen[page[r]/block]>1;
  }
  free(seen);
  return 0;
}

/* -- this node's hops (design/disjoint-pages-change-request.md, C3.3 and C3.7) ------------------------------------ */

/* Each tree's communications (design/compilation-coordination.md, T4: "a (page, link) is one communication"; the
   instructions after the tasks, "The documents' revisions of edd4b583", GO1 and GO3, numbered here for the order's
   ties).  A tree carries a page over each of its edges once (mesh.h, "A tree carries a page over a node pair once at
   most"), so one communication is a (page, tree edge): the tree's page `page` (its place among the demand's pages on
   that tree, from 0) over the edge into vertex `head`.  They are numbered tree by tree in the solve's order grouped by
   demand, then page by page, then by the head's place among the tree's vertices but its root (`members`, from its row
   of the tree's vertices keyed by parent, mesh_graph_rows, as plan_walk reads a tree; no walk over a parent table
   here, C9), the same at every rank.  Every one lies on some destination's route (plan_schedule): mesh_graph_flow
   makes a tree of its leaves' paths to its root, and its leaves are the demand's destinations.  0; ENOMEM where they
   pass 32 bits (each is instruction tasks + c, below 32 bits: plan_schedule; 2026-11-05, "The revisions of 46ae8fdf",
   LP: where it cited mesh_graph_contention_schedule's `communication`, which is removed) or the call's own memory. */
static int plan_trees(struct plan *p){
  const uint32_t n=p->n;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  p->members=malloc(((size_t)p->made+1)*sizeof *p->members);
  p->lead=malloc(((size_t)p->made+1)*sizeof *p->lead);
  p->comm=malloc(((size_t)p->made+1)*sizeof *p->comm);
  if(!p->members || !p->lead || !p->comm)return ENOMEM;
  p->comms=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    uint64_t k=0;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      p->members[j]=0;
      for(size_t c=first[0];c<first[n];c++)p->members[j]|=UINT64_C(1)<<child[c];
      p->lead[j]=k;
      p->comm[j]=p->comms;
      k+=p->run[j];
      p->comms+=p->run[j]*(uint64_t)__builtin_popcountll(p->members[j]);
      if(p->comms>=UINT32_MAX)return ENOMEM;
    }
  }
  return 0;
}
/* tree j's page `page` (from 0 on that tree) over its edge into `head`, a member of the tree (plan_trees) */
static uint32_t plan_communication(const struct plan *p,uint32_t j,uint64_t page,uint32_t head){
  return (uint32_t)(p->comm[j]+page*(uint64_t)__builtin_popcountll(p->members[j])+plan_rank(p->members[j],head));
}

/* Every page of the plan in one order, the same at every rank: the demands by their first declared transfer, a
   demand's pages tree by tree.  A tree carries a page over a node pair once at most, so visiting this node's hops page
   by page in that order, for a pair a -> b, a's sends to b and b's receives from a are the same hops in the same order
   (mesh.h "Both ends").  A page's hops here: its receive from this node's parent on the tree (mesh-graph.h
   mesh_graph_climb's first step), then its sends to this node's children (its row of the tree's vertices keyed by
   parent, mesh_graph_rows), in their order.  The tree's edges are the graph module's: no walk over a parent table
   here (the C9 reviews of c5bffd0c and c9ffedcf).  A page's children are its own tree's, which p->child (every tree's
   union, plan_template) does not tell, so the rows are made here again.
   This is one copy's list; every copy's is the same hops in the same order (mesh.h, "Copies"; mesh_plan writes copy
   c's at [c n, c n + n)).  Counting (p->entry NULL) gives p->entries (n), p->receives, and each demand's received
   range (`range`, in the plan's demand order; p->ranges of them) and its pages landing here (`received`, P_g);
   filling writes p->entry[0 .. n).  Each receive takes the next field of its range's group, so a range's receives
   here are fields 0 .. P_g - 1 of its group; a send from the range's source names its release, any other send its
   receive's field (a relay's forward, or a destination's on the way on).  Filling also gives each entry its
   communication (plan_communication: a receive's the tree edge into this node, a send's the edge into its child), and
   its `set`: j + 1 for a page on tree j of a demand whose range is a set (its every declared transfer MESH_PAGES), j
   the index this walk iterates, d->tree .. d->tree + d->trees into p->tree and p->run, the same at every rank; 0 for
   every other entry (design/compilation-coordination.md, S1 and S2: the set's unit is one tree's pages, every edge of
   tree j carrying the same run of them).  plan_schedule then orders the entries by their places in the global
   topological order of the instructions (design/plan-to-bridge-change-request.md R5; design/compilation-
   coordination.md, "The documents' revisions of edd4b583", GO3), and every field stays as assigned here.
   2026-11-04, owner C3 (GO4): no entry records its place in this order (`walk`, T6's tie and S3's least walk, gone
   with them).  What must stay: the order above, in which tree j's entries of one node and direction are one run
   (plan_schedule's groups).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): plan_schedule forms no groups now
   (R5: a set's pages "keep their places in that order"), so that run is no group's.  What must stay: every field as
   assigned here; the order above, the same at every rank and in every copy.
   2026-11-04, sub-assignment 3: reversing 2026-11-03's (sub-assignment 5's) bit a receive in one landed record an
   execution, counted over all of the execution's receives.  Why: a landing is a phase field of its range's group and
   an arrival on its range's count (C3.7; the contract's D1, counts per received range and copy).  What must stay: the
   order above, the same at every rank and in every copy, so both ends' lists agree; one receive a page at a node, so
   each field has one writer.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1 and
   R-6; "The revisions of 14b9a757", R-1 and R-6): a demand's units its instructions (plan_instructions), each entry's
   `instruction` its tree's instruction k, where they were its pages; and `toward`, the map indices the entries name
   (a receive's parent, a send's child), each tree that carries an instruction's, the same in both passes.  Why: R-1,
   "The bridge's instructions are those blocks"; mesh.h struct mesh_hop, "A receive's `pages` is its range's
   instructions this node receives in one copy (its range's P), each of them one field of its range's group and one
   arrival on its count"; R-6, the plans "with an entry naming node i (struct mesh_hop `node`, either direction)"
   (mesh.h struct mesh_algo `adjacent`).  What must stay: the order above */
static void plan_walk(struct plan *p){
  const uint32_t n=p->n,self=p->self;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  uint64_t at=0,receives=0;
  if(!p->entry)p->ranges=p->consumed=0,p->toward=0;
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    uint32_t field=0;
    uint64_t k=0;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      const uint64_t end=k+p->run[j];
      const uint32_t set=d->set?j+1:0;
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      uint32_t in=MESH_ABSENT;
      mesh_graph_climb(tree.parent,n,d->source,self,plan_tree_up,&in);
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      if(in==MESH_ABSENT && first[self+1]==first[self]){k=end;continue;}
      if(k<end){
        if(in!=MESH_ABSENT)p->toward|=UINT64_C(1)<<in;
        for(size_t c=first[self];c<first[self+1];c++)p->toward|=UINT64_C(1)<<child[c];
      }
      for(;k<end;k++){
        /* the page's field: its receive's here; none at the source, whose sends its release's supplied word releases */
        uint32_t mine=MESH_ABSENT;
        if(in!=MESH_ABSENT){
          mine=field++;
          if(p->entry)p->entry[at]=(struct plan_entry){.node=in,.direction=MESH_RECEIVE,.demand=i,.field=mine,
                                                       .communication=plan_communication(p,j,k-p->lead[j],self),
                                                       .set=set,.instruction=k};
          at++;
        }
        for(size_t c=first[self];c<first[self+1];c++,at++)
          if(p->entry)p->entry[at]=(struct plan_entry){.node=(uint32_t)child[c],.direction=MESH_SEND,.demand=i,
                                                       .field=mine,
                                                       .communication=plan_communication(p,j,k-p->lead[j],
                                                                                         (uint32_t)child[c]),
                                                       .set=set,.instruction=k};
      }
    }
    if(!p->entry){
      d->received=field;
      d->range=field?p->ranges++:MESH_ABSENT;
      d->consumed=field && d->leaves>>self&1?p->consumed++:MESH_ABSENT;
      receives+=field;
    }
  }
  p->entries=at;
  if(!p->entry)p->receives=receives;
}

/* -- the task graph and its schedule (design/plan-to-bridge-change-request.md, R3 to R5) --------------------------- */

/* mesh_graph_climb's visit, the route of a page down its tree from the root to `below` (plan_schedule): the vertex
   met at step s >= 1 is the parent of the one met at step s - 1, and the tree edge between them is the route's link
   depth - s, in route order from the root; `head` that link's lower vertex.  A visit of mesh_graph_climb recording the
   edges it visits, as plan_tree_up does, so it stays here (design/compilation-coordination.md, C9 rulings); the route
   is the tree path the fill gave the page (T1) */
struct plan_path { const uint32_t *link_of; uint32_t n,depth,below,*link,*head; };
static void plan_path_step(void *consumer,uint32_t at,uint32_t step){
  struct plan_path *path=consumer;
  if(step){
    path->link[path->depth-step]=path->link_of[(size_t)at*path->n+path->below];
    path->head[path->depth-step]=path->below;
  }
  path->below=at;
}
/* an event's tasks (plan_schedule; design/compilation-coordination.md, T1: the tasks are the template's events, a
   consumption one task for MESH_JOIN and one a page for MESH_PAGES, design/plan-to-bridge-change-request.md, R4), its
   first and how many: P(d) and S(d) one each, a consumption one or its range's pages */
struct plan_span { uint32_t first,count; };
static struct plan_span plan_tasks(const struct plan *p,const struct plan_span *consumer,const uint32_t *reader,
                                   uint32_t i,int produced,int own){
  if(produced)return (struct plan_span){p->of[i],1};
  if(own)return (struct plan_span){reader[p->of[i]],1};
  return consumer[i];
}
/* design/plan-to-bridge-change-request.md R5, "The order of each cable's list, by the reference algorithm": the
   entries by their own place (`place`), each list a subsequence of the one global topological order, a set's pages
   keeping their places in it.  A total order, so every rank's sort ends alike: each communication has one place, and
   no two of this node's entries are one communication (a receive's is its tree edge into this node, each send's the
   edge into its own child).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): reversing GO4's first key, the
   entry's group's least place (`at`), by which a set's entries on one node pair and direction took consecutive places
   at the place of its first page.  Why: R5, "A set's pages (design/aot-compilation-change-request.md, appendix item
   5) keep their places in that order.  The set's random permutation is drawn over those places.  (Corrected
   2026-11-05: the first text moved a set's pages to consecutive places, which nothing requires.)"  What must stay: a
   total order of the entries both ends of a pair agree on.
   2026-11-04, owner C3 (GO4): reversing T6 as S3 amended it (the group's least scheduled start, then its least walk,
   then the entry's walk).  Why: each cable's list is a subsequence of MSCCLang §5.2 step 3's global topological order,
   "Scheduled start times do not order the lists", and a set's pages "take consecutive places at the place of the
   set's first page in that order" (design/plan-to-bridge-change-request.md, R5).  What must stay: a total order of the
   entries both ends of a pair agree on; a set's entries contiguous on their node pair and direction within each copy
   (the bridge's EINVAL where they are not: design/compilation-coordination.md, S7). */
static int plan_entry_order(const void *a,const void *b){
  const struct plan_entry *x=a,*y=b;
  return (x->place>y->place)-(x->place<y->place);
}
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): plan_entry_side (mesh_graph_rows'
   keys, an entry's node pair and direction, by which plan_schedule found each set's group and its least place) is gone
   with the groups.  Why: R5's corrected text, at plan_entry_order.  What must stay: nothing of it */

/* The task graph, its schedule and the order of the lists (design/plan-to-bridge-change-request.md, R3 to R5;
   design/aot-compilation-change-request.md, A6; design/compilation-coordination.md, T1, T2 and T4, and "The
   documents' revisions of edd4b583", GO1 to GO5; "The revisions of 46ae8fdf", LP), one execution's, every rank's the
   same from the same declared transfers, weights, dependences and map.
   - The task graph G = (V, E, w, c) (O. Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007; O. Sinnen and L. A.
     Sousa, IEEE TPDS 16(6) 2005, Definition 1), T1: its tasks the template's events (plan_template), each on the node
     its event is at (the placement operand: the declared transfers' sources and destinations).  Task d is demand d's
     production at its source, P(d), weighing the weighting operand's `produced` (design/aot-compilation-change-
     request.md, A3); then each demand's source read, S(d), where a dependence names it, weighing `read`; then each
     declared transfer's consumption at its destination, in declared order, one task weighing `consumed` for MESH_JOIN
     (function A, design/plan-to-bridge-change-request.md, R4) and one a page, each weighing `consumed`, for MESH_PAGES
     (function B).  Its edges, in this order: for each declared transfer in declared order, page by page, page k's
     edge from P to the consumption (to page k's task for MESH_PAGES), of cost `page`, its route the tree path the
     fill gave page k, from the source down to the destination (mesh_graph_climb, plan_path_step); each P(d) -> S(d);
     each dependence of lag 0 in the dependences' order, from each task of its `from` event to each of its `to`
     event's (T1: an edge out of a MESH_PAGES consumption leaves from each of its page tasks; design/compilation-
     coordination.md, T1': an edge into one enters each of them, each page's task the function's work on that page
     after its local predecessors).  A local edge (P -> S, a lag-0 dependence) has no route and costs 0 (T1': Sinnen
     and Sousa's classic model, local communication free, Definition 2).
   - The topology graph (Sinnen and Sousa, Definition 4), T2: the node graph's pairs, pair e two directed links, 2e
     from[e] -> to[e] and 2e + 1 back, each of relative data rate b(L) the pair's capacity over the greatest pair's.
   - T4: every route entry of one page over one tree edge is one communication (plan_communication), one instruction
     on its link.  Every tree edge lies on some destination's route (plan_trees).
   - The instructions (GO1): the task graph's tasks as numbered above, 0 .. tasks - 1, then the plan's communications
     (T4: a page over a tree edge; plan_trees and plan_communication number them), communication c at tasks + c.  A
     communication is one instruction: its send at the edge's upper node and its receive at its lower node take its
     one place, so a node pair's sends at one end and its receives at the other are the same hops in the same order
     (mesh.h, "Both ends"; R1: "the bridge's records name the same pages").  MSCCLang (M. Cowan et al., ASPLOS 2023)
     §4.2 expands a remote copy "into a send and a receive instruction" joined by "a communication edge"; joining them
     in one instruction is where this differs from it, so that both ends' lists agree (GO1).  The dependencies: each
     local edge of the task graph as it is (P(d) -> S(d), a lag-0 dependence); each remote edge a chain, its tail task
     -> its route's first communication -> each next -> its last -> its head task: MSCCLang §4.2's communication and
     processing edges, a page sent before it lands and landed before it is forwarded or consumed (plan_template's P ->
     W(y), W(y) -> F(y, c), L(z) -> C(z)).  Routes sharing a prefix (T4) give a dependency more than once, which
     adds no row (mesh-graph.h mesh_graph_completion_times: each edge once, LP4) and changes no order
     (mesh_graph_heap_order: an edge given twice counts twice and is removed twice).  Every rank builds the same
     instructions and dependencies, in the same order.
   - The priority (R5 at mesh 0e56d8da, "How: the program"): "one variable C_j, a completion time, for each instruction
     j of one execution's task graph (R3): each task at its node, and each hop of each page's route, on its cable
     (design/compilation-coordination.md GO2)"; "Instruction j's length d_j is its weight from the weighting operand
     (aot-compilation A3): w for a task, and for a hop c(e) over its cable's bandwidth".  A task's d_j is its weight
     (A3); communication c's on link L is c(e) / b(L) (b(L) T2's rate), the greatest over the route entries of the
     task graph's edges e that carry it (T4: one instruction for every edge that shares it), one correctly rounded /
     each and the max by comparisons, as GO2 took it.  Its resource (R5: "A resource is a cable in one direction, or
     a node's device"): a task's its node's device, numbered by the node's map index; a communication's its link L in
     one direction (T2), numbered n + L.  The rows are R5's, over these instructions and the dependencies above
     (mesh-graph.h mesh_graph_completion_times: Chen's (14), (14a), (1b), (1c), Hall et al.'s (5) on each resource by
     cutting planes, each linear program by HiGHS); the priority is each instruction's C̄_j in its solution.
   - The order (R5, "The order of each cable's list, by the reference algorithm"): MSCCLang §5.2, step 3: "Sort
     instructions into a global topological order with respect to their dependencies with a heap using the priority to
     order instructions" (mesh-graph.h mesh_graph_heap_order), "the least C̄_j first.  Where two are equal, the heap
     takes the lower instruction" (R5; given as -C̄, LP7), the same at every rank (F. B. Schneider, Implementing
     fault-tolerant services using the state machine approach, ACM Comput. Surv., 1990).  That order "is consistent
     with the precedence relation, as Hall et al.'s ties require" (R5): the heap places a vertex only after every
     vertex an edge into it leaves.
   - The lists (R5, "The order of each cable's list, by the reference algorithm"): each of this node's entries
     (plan_walk; every field as it assigns them) takes its communication's place (`place`), and the entries are sorted
     by it (plan_entry_order), a total order.  A set's entries (S1, S2: its nonzero `set`, tree j's pages over a node
     pair) keep their places like every other entry (R5: "A set's pages ... keep their places in that order.  The
     set's random permutation is drawn over those places"); the bridge draws it (mesh-flow.c link_prepare and
     position_record).  So each node pair's list at each end is a subsequence of the one global order, both ends'
     alike (a send at x to y and its receive at y are one communication, at one place), and every copy's the same
     (mesh_plan writes this one order for each copy).
   - Where it is a topological order.  The global order is a topological order of the instructions (GO3: mesh-graph.h
     mesh_graph_heap_order), each list a subsequence of it, and every lag points forward, so each cable's sequence of
     executions in that order is one (R5: "That order is a topological order of the unrolled executions, since every
     lag points forward").  Its dependencies are the bridge's store-and-forward waits (a forward released by its
     receive's landing, a source's send by its production: GO1), so in that order the cables' sequences cannot wait on
     each other in a cycle (MSCCLang §5.2: "By assigning instructions to thread blocks in a topological order that
     respects communication and processing edges, all implicit dependencies introduced by thread block sequential
     execution cannot produce cycles").  Each execution's one permutation of a set's every position on a pair, each
     cable carrying its own members in its order (design/aot-compilation-change-request.md, appendix item 5: "Every
     edge and every cable of the set's tree carries its share of the set's pages in that one permutation's order";
     mesh-flow.c), changes which page each of those places carries.  The item's argument for the permuted sequences,
     "each cable's sequence is the same renaming of them", holds where one cable carries every position of the set on
     each pair of its tree, and is not shown where the deal splits a set's positions over a pair's cables
     (design/compilation-coordination.md, "The revisions of 46ae8fdf", stop 3; 2026-11-05: this said the documents
     state no argument, "The revisions of 67ca86f8", stop 3).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP; design/plan-to-bridge-change-
   request.md R5 at mesh 0e56d8da): reversing GO2's and GO3's priority (each instruction's bottom level by Sinnen and
   Sousa's eq. (7), the greatest first) and GO5's call of the contention schedule (Sinnen and Sousa's Algorithm 1 with
   Algorithm 2, whose starts ordered nothing), with their `start`, `link_start`, `level` and `delay`.  Why: R5, "Each
   instruction's priority is C̄_j, its completion time in the program's optimal solution, the least first"; "It
   replaces MSCCLang's depth and reverse depth, and edge scheduling's start times, which are no longer computed for the
   order.  Edge scheduling's part, one communication at a time on each link, is now the resources' rows."  What must
   stay: GO1's instructions and dependencies; GO2's lengths; every entry's fields as plan_walk assigns them; both
   ends' lists of a node pair the same hops in the same order; every copy's the same order.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): reversing GO4's groups, a set's
   entries on one node pair and direction at consecutive places at the place of its first page (its least place,
   found in one pass over the entries' rows by node and direction, mesh_graph_rows), the entries sorted by their
   group's least place, then their own; and with them GO4's and GO4′'s argument that the groups in order of their
   least places were a topological order with each group whole, which made any permutation within a group one too.
   Why: R5, "keep their places in that order.  The set's random permutation is drawn over those places.  (Corrected
   2026-11-05: the first text moved a set's pages to consecutive places, which nothing requires.)"  What must stay:
   both ends' lists of a node pair the same hops in the same order; every copy's the same order.
   2026-11-04, owner C3 (GO1 to GO5; design/plan-to-bridge-change-request.md, R5, "The order of each cable's list, by
   the reference algorithm"): reversing T6 as S3 amended it, each entry at its link's scheduled start of its page
   (`start`, through comm_start, each communication's one start on its link), each group's least start (`earliest`)
   and least walk (`least`), the entries sorted by them and then by their walk.  Why: R5 (the operator: "what
   algorithm is cited here, if any? no citations? use the reference algorithm instead") orders each cable's list as a
   subsequence of MSCCLang §5.2 step 3's global topological order by bottom level, and "Scheduled start times do not
   order the lists"; it overturns T6 and S3 (design/compilation-coordination.md, "The documents' revisions of
   edd4b583").  What must stay: the schedule's call (GO5); every entry's fields as plan_walk assigns them; both ends'
   lists of a node pair the same hops in the same order; every copy's the same order; a set's entries contiguous on
   their node pair and direction within each copy (S7).
   2026-11-04, owner C3 (GO4; review W3 #6): the claim owner C2 withdrew here (the start order not shown to be a
   topological order of the bridge's store-and-forward waits, the contention schedule's (14) being nonaligned,
   cut-through) is answered above: the lists' order is now the instruction DAG's, whose dependencies are those waits,
   not the schedule's starts (GO4: "This answers the open item 'No cycle on a cable' (W3 #6)").
   0; EINVAL, EDOM or ENOMEM (mesh_graph_completion_times': a program with no solution, HiGHS's failure; and
   mesh_graph_heap_order's EINVAL, a cycle); ENOMEM.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
   revisions of 14b9a757", R-1): a range's units its instructions P (plan_instructions), where they were its pages: a
   MESH_PAGES consumption one task an instruction, each edge of declared[i] one an instruction, its cost `page` (A3's c
   of each page's edge), as plan_walk's entries and the fill's units are.  The implementer's decision, recorded (the
   documents are silent on the weight of an instruction shorter than a page): each instruction weighs as a page did, the
   weighting operand's unit kept; no weight is rescaled by an instruction's bytes here.  What must stay: as above */
static int plan_schedule(struct plan *p,const struct mesh_weight *weight,const struct mesh_dependence *dependence,
                         uint32_t dependences){
  const uint32_t n=p->n,demands=p->demands,count=p->transfers;
  const struct mesh_declared *declared=p->declared;
  uint32_t link_of[(size_t)n*n+1],path_link[(size_t)n+1],path_head[(size_t)n+1];
  uint32_t *reader=malloc(((size_t)demands+1)*sizeof *reader);
  struct plan_span *consumer=malloc(((size_t)count+1)*sizeof *consumer);
  if(!reader || !consumer){free(reader);free(consumer);return ENOMEM;}
  /* the tasks, the edges and their route entries counted (each below 32 bits, as the template's vertices) */
  uint64_t tasks=demands,edges=0,entries=0;
  int status=0;
  for(uint32_t d=0;d<demands;d++)
    if((reader[d]=p->demand[d].own?(uint32_t)tasks:MESH_ABSENT)!=MESH_ABSENT)edges++,tasks++;
  if(tasks>=UINT32_MAX)status=ENOMEM;
  for(uint32_t i=0;!status && i<count;i++){
    const struct plan_demand *d=p->demand+p->of[i];
    consumer[i]=(struct plan_span){(uint32_t)tasks,declared[i].kind==MESH_PAGES?(uint32_t)d->instructions:1};
    tasks+=consumer[i].count;
    edges+=d->instructions;
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++)
      if(p->run[j])entries+=p->run[j]*mesh_graph_climb(p->parent+p->tree[j]*n,n,d->source,declared[i].destination,
                                                       NULL,NULL);
    if(tasks>=UINT32_MAX || edges>=UINT32_MAX || entries>=UINT32_MAX)status=ENOMEM;
  }
  for(uint32_t k=0;!status && k<dependences;k++){
    const struct mesh_dependence *x=dependence+k;
    if(x->lag)continue;
    const struct plan_span a=plan_tasks(p,consumer,reader,x->from,!!(x->kinds&MESH_FROM_PRODUCED),
                                        !!(x->kinds&MESH_FROM_SOURCE_READ)),
                           b=plan_tasks(p,consumer,reader,x->to,!!(x->kinds&MESH_TO_PRODUCED),
                                        !!(x->kinds&MESH_TO_SOURCE_READ));
    if((edges+=(uint64_t)a.count*b.count)>=UINT32_MAX)status=ENOMEM;
  }
  /* the instructions (GO1): the tasks, then the communications; their dependencies: each edge's route entries and one
     more */
  const uint64_t instructions=tasks+p->comms,dependencies=edges+entries;
  if(instructions>=UINT32_MAX)status=ENOMEM;
  double *w=status?NULL:malloc(((size_t)instructions+1)*sizeof *w),
         *cost=status?NULL:malloc(((size_t)edges+1)*sizeof *cost),
         *rate=malloc((2*(size_t)p->pairs+1)*sizeof *rate),
         *completion=status?NULL:malloc(((size_t)instructions+1)*sizeof *completion);
  uint32_t *resource=status?NULL:malloc(((size_t)instructions+1)*sizeof *resource),
           *from=status?NULL:malloc(((size_t)edges+1)*sizeof *from),
           *to=status?NULL:malloc(((size_t)edges+1)*sizeof *to),
           *link=status?NULL:malloc(((size_t)entries+1)*sizeof *link),
           *communication=status?NULL:malloc(((size_t)entries+1)*sizeof *communication),
           *before=status?NULL:malloc(((size_t)dependencies+1)*sizeof *before),
           *after=status?NULL:malloc(((size_t)dependencies+1)*sizeof *after),
           *place=status?NULL:malloc(((size_t)instructions+1)*sizeof *place);
  size_t *route=status?NULL:malloc(((size_t)edges+1)*sizeof *route);
  if(!status && (!w || !cost || !rate || !completion || !resource || !from || !to || !link || !communication ||
                 !before || !after || !place || !route))status=ENOMEM;
  if(!status){
    /* the tasks: each on its event's node, weighing the weighting operand's weight of it (A3), its resource its node's
       device, resource index its node's map index (R5: "A resource is a cable in one direction, or a node's device") */
    for(uint32_t d=0;d<demands;d++){
      const struct plan_demand *x=p->demand+d;
      w[d]=weight[x->first].produced;
      resource[d]=x->source;
      if(reader[d]!=MESH_ABSENT){w[reader[d]]=weight[x->first].read;resource[reader[d]]=x->source;}
    }
    for(uint32_t i=0;i<count;i++)for(uint32_t q=0;q<consumer[i].count;q++){
      w[consumer[i].first+q]=weight[i].consumed;
      resource[consumer[i].first+q]=declared[i].destination;
    }
    /* the links (T2) */
    double most=0;
    for(size_t v=0;v<(size_t)n*n;v++)link_of[v]=MESH_ABSENT;
    for(uint32_t e=0;e<p->pairs;e++)if(p->capacity[e]>most)most=p->capacity[e];
    for(uint32_t e=0;e<p->pairs;e++){
      link_of[(size_t)p->from[e]*n+p->to[e]]=2*e;
      link_of[(size_t)p->to[e]*n+p->from[e]]=2*e+1;
      rate[2*e]=rate[2*e+1]=p->capacity[e]/most;
    }
    /* the edges, their costs and routes */
    size_t e=0,r=0;
    for(uint32_t i=0;i<count;i++){
      const struct plan_demand *d=p->demand+p->of[i];
      const uint32_t z=declared[i].destination;
      for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
        if(!p->run[j])continue;
        const uint32_t *parent=p->parent+p->tree[j]*n;
        struct plan_path path={link_of,n,mesh_graph_climb(parent,n,d->source,z,NULL,NULL),z,path_link,path_head};
        mesh_graph_climb(parent,n,d->source,z,plan_path_step,&path);
        for(uint64_t q=0;q<p->run[j];q++,e++){
          from[e]=p->of[i];
          to[e]=consumer[i].first+(declared[i].kind==MESH_PAGES?(uint32_t)(p->lead[j]+q):0);
          cost[e]=weight[i].page;
          route[e]=r;
          for(uint32_t s=0;s<path.depth;s++,r++){
            link[r]=path_link[s];
            communication[r]=plan_communication(p,j,q,path_head[s]);
          }
        }
      }
    }
    for(uint32_t d=0;d<demands;d++)
      if(reader[d]!=MESH_ABSENT){from[e]=d;to[e]=reader[d];cost[e]=0;route[e++]=r;}
    for(uint32_t k=0;k<dependences;k++){
      const struct mesh_dependence *x=dependence+k;
      if(x->lag)continue;
      const struct plan_span a=plan_tasks(p,consumer,reader,x->from,!!(x->kinds&MESH_FROM_PRODUCED),
                                          !!(x->kinds&MESH_FROM_SOURCE_READ)),
                             b=plan_tasks(p,consumer,reader,x->to,!!(x->kinds&MESH_TO_PRODUCED),
                                          !!(x->kinds&MESH_TO_SOURCE_READ));
      for(uint32_t s=0;s<a.count;s++)for(uint32_t t=0;t<b.count;t++){
        from[e]=a.first+s;to[e]=b.first+t;cost[e]=0;route[e++]=r;
      }
    }
    route[e]=r;
  }
  if(!status){
    /* GO1: each edge of the task graph a chain of dependencies from its tail task through its route's communications,
       in route order, to its head task (a local edge, of no route, one dependency from its tail to its head) */
    size_t k=0;
    for(size_t x=0;x<edges;x++){
      uint32_t tail=from[x];
      for(size_t r=route[x];r<route[x+1];r++,k++){before[k]=tail;tail=after[k]=(uint32_t)tasks+communication[r];}
      before[k]=tail;after[k++]=to[x];
    }
    /* GO2: communication c's length the greatest c(e) / b(L) over the route entries carrying it, from 0 (each such
       quotient is > 0, every `page` weight and rate being > 0, and every communication lies on a route, plan_trees),
       its resource its route entries' link L in one direction, n + L (R5: "for a hop c(e) over its cable's
       bandwidth"; every route entry of one communication is on one link, its tree edge's: plan_path_step) */
    for(uint64_t c=0;c<p->comms;c++){w[tasks+c]=0;resource[tasks+c]=n;}
    for(size_t x=0;x<edges;x++)for(size_t r=route[x];r<route[x+1];r++){
      const double occupy=cost[x]/rate[link[r]];
      if(occupy>w[tasks+communication[r]])w[tasks+communication[r]]=occupy;
      resource[tasks+communication[r]]=n+link[r];
    }
    /* R5's program over the instructions, its rows the dependencies (mesh-graph.h mesh_graph_completion_times), then
       the global topological order by the heap, the least C̄ first as -C̄ (LP7), a tie to the lower instruction */
    status=mesh_graph_completion_times((uint32_t)instructions,w,resource,n+2*p->pairs,(size_t)dependencies,before,after,
                                       completion);
    if(!status){
      for(uint64_t j=0;j<instructions;j++)completion[j]=-completion[j];
      status=mesh_graph_heap_order((uint32_t)instructions,completion,(size_t)dependencies,before,after,place);
    }
  }
  if(!status){
    /* R5: each entry at its communication's place, a set's among them, sorted by it (2026-11-05, "The revisions of
       67ca86f8", SP: the groups' least places gone) */
    for(uint64_t j=0;j<p->entries;j++)p->entry[j].place=place[tasks+p->entry[j].communication];
    qsort(p->entry,p->entries,sizeof *p->entry,plan_entry_order);
  }
  free(reader);free(consumer);free(w);free(cost);free(rate);free(completion);
  free(resource);free(from);free(to);free(link);free(communication);free(before);free(after);free(place);free(route);
  return status;
}

/* The queues' rooms and the program's synchronization edges, fixed at compile time by the self-timed buffer bound
   (design/aot-compilation-change-request.md, appendix item 2, "Its room is fixed at compile time, by the self-timed
   buffer bound"; S. S. Bhattacharyya, S. Sriram and E. A. Lee, Optimizing synchronization in multiprocessor DSP
   systems, IEEE Trans. Signal Processing 45(6), 1997, here BSL; design/compilation-coordination.md, "The revisions of
   3f106747", BS).
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 4: "build the
   program's whole IPC graph at compile time, every function together, with each queue's room (5) in it.
   `plan_rooms`' part graph goes"; appendix item 2, "Each plan builds its own part of that graph", withdrawn: "the
   compilation builds the program's whole IPC graph, as this item states above, and each queue's room is (5) in that
   graph"; "The revisions of 4e78f6f4"), reversing plan_rooms' graph of one plan (BS-3) and its one edge into and one
   out of this node (BS-7): the graph is the program's, built once in the plan call over every plan together, as BSL
   §4.1 builds one ("if actors v1, v2, …, vn are assigned to the same processor in that order, then Gipc would have a
   cycle ((v1, v2), (v2, v3), …, (vn − 1, vn), (vn, v1)), with delay((vn, v1)) = 1"), every delay counting executions
   of the program's traced block, BSL's iterations (mesh.h struct mesh_sync).  plan_graph builds its operands only:
   the components, Convert-to-SC-graph, the cycle mean, MinDelay, DetermineDelays and the least delays are the graph
   module's (item 2, "Where they live"; mesh-graph.h).  What must stay: each queue's room its edge's bound by (5),
   ENOMEM where none is found, before anything is written.
   - Its vertices, in this order, so ties of least time fall to σ (BS-2): for each call j (the program's calls in
     program order, calls[j] its function f), the program's submission σ_j and its bridge's TO-broadcast β_j at the
     sender s, the least of `program` (A7, sender(m)), vertices 2j and 2j + 1; then for each call j and each node v of
     f's plan (its `nodes`; BS-5: {s} for a plan with no declared transfer, at every rank) the actor x_{j,v}, node v's
     part of call j (item 2: "The actors are each node's parts of the program's calls, in program order").
   - Its edges, each with its delay:
     - the sender's two processors in program order, BSL §4.1's cycle: σ_j -> σ_{j+1} of delay 0, σ_last -> σ_0 of
       delay 1, and β_j -> β_{j+1}, β_last -> β_0 alike: one program thread submits in program order, and one
       sender's TO-broadcasts keep FIFO Order (A7);
     - each call's queues, its function's, each of delay 0: σ_j -> β_j, the submissions' queue; β_j -> x_{j,v} for
       each v, the delivered calls' queue at v (item 2: "Its source is the sender's TO-broadcast of the plan's calls,
       and its sink is the bridge's taking of them at the node"); and σ_j -> x_{j,s} where s is a node of f's plan;
     - A10's bound (item 2: "At each node, A10's bound is the delay on the edge from a call's completion to the
       program's later submission"): f's n calls in program order, b its `outstanding`, from the call at place i to
       the one at place m = (i + b) mod n, b of f's calls later, of delay ⌊(i + b) / n⌋ executions of the block: at s,
       x_{j_i,s} -> σ_{j_m}, or σ_{j_i} -> σ_{j_m} where s is no node of f's plan (BS: "σ -> σ of that delay"); at
       each other node v of f's plan, x_{j_i,v} -> x_{j_m,v}.  For n = 1 these are BS's x_s -> σ and x_v -> x_v of
       delay b;
     - each transfer, a tree edge y -> c carrying f's pages (`child`), both ways, x_{j_i,y} -> x_{j_m,c} and
       x_{j_i,c} -> x_{j_m,y}, m = (i + u) mod n, of delay ⌊(i + u) / n⌋: C3.3's u copies (item 2: "The IPC edges are
       the plans' transfers, each with C3.3's u copies as its delay"; BS-6), u of f's executions mapped to the
       block's as A10's edges are;
     - not added: an edge between two calls' actors at one node in program order (BSL's cycle over a node's actors).
       Stop IPC-1 (design/compilation-coordination.md, "The revisions of 4e78f6f4"): item 2 says "The processors are
       the program's nodes.  The actors are each node's parts of the program's calls, in program order", while it
       also says "Delivery does not wait for any earlier call to finish ... A call still running at a node delays no
       later call's delivery there", and A9 makes the processors "the sends of each adjacent node and direction, R5's
       cables"; a node's parts of successive calls overlap, so an edge forcing them in sequence would bound the
       queues by an order the code does not keep.
   - Its times: t(x_{j,v}) f's plan's t(x_v) (plan_times); t(σ) = t(β) = 0 (BS-2).
   - Then: ENOMEM where a cycle has zero total delay (mesh_graph_cycle_mean's INFINITY; BSL Lemma 1, "Every cycle C in
     the IPC graph has a path delay of at least one if and only if the static schedule it is constructed from is free
     of deadlock"); the edges Convert-to-SC-graph adds (§9, Fig. 6; mesh_graph_convert_sc), so that every queue
     "becomes encapsulated in an SCC" (§9), and their delays δ by DetermineDelays (Fig. 8;
     mesh_graph_determine_delays), its EDEADLK and ERANGE as ENOMEM.
   - The added edges.  σ_0 reaches every vertex (σ's cycle, σ_j -> β_j, β_j -> each x_{j,v}), so its component is the
     one source SCC and Convert-to-SC-graph chains the sink SCCs alone; no sink SCC holds a σ or a β but where the
     graph is strongly connected (σ's component reaches every vertex, and β's reaches a σ only through an actor, all
     three one component then).  So every added edge's tail is an actor x_{j,a}, and its head an actor x_{j',b} or the
     source SCC's vertex of least time, σ_0 (time 0, the lowest vertex; BS-2: "the sink-source edge's sink is the
     program's start at s").  An edge's count is f's starts at a (BS-4; item 2: "The count its source raises is
     therefore the program's starts of the plan at that node"), and the start of the call A10's edge out of x_{j,a}
     leads to (its place m among f's calls, its delay d) follows x_{j,a}'s end d executions earlier, so the edge's
     delay in the rooms' graph is δ + d.  It is a wait entry of plan f' = calls[j'] at b (at s for σ_{j'}: the
     program's start of call j' there), `node` a, `function` f + 1, `position` m, `every` n_f, `at` j''s place among
     f''s calls, `delay` δ; and a tell entry of plan f at a, `node` b (s for σ), `function` f' + 1 (mesh.h struct
     mesh_sync).  The bridge consumes them as A9's self-timed schedule's firing rule
     (design/aot-compilation-change-request.md A9, "What the bridge runs: the self-timed schedule": "Before firing an
     actor, a processor waits for the data needed by that actor to become available").
   - Rooms, (5) by Theorem 2's "Bfb(e) = ρ_Gs(snk(e), src(e)) + delay(e)", ρ the least total delay over walks in the
     converted graph (BSL §3; mesh_graph_lags, Dijkstra's, as the theorem's "Dijkstra's algorithm"): f's `room` here
     the sum over its calls j of ρ(x_{j,self}, β_j), its one queue here carrying every call's deliveries and so every
     call's tokens; its `submit_room` the sum of ρ(β_j, σ_j); `room` 1 at a node of the program that is no node of f's
     plan, whose queue no message reaches (Dest(m) is `nodes`; BS-5).  ENOMEM where a queue lies on no cycle after the
     conversion (ρ UINT32_MAX), where a bound is 0, or where a room passes its words.
   2026-11-05 (118; design/compilation-coordination.md, "Answered in the documents (2026-11-05, the coordinator, on
   this part's record)", IG-5, and "Review of 4e78f6f4 against the requirement lines", K1 and G1; design/aot-
   compilation-change-request.md appendix item 2, corrected: "The program-order cycle is each node's submissions, not
   its calls' parts ... the submissions σ_1 … σ_n of the block's calls are one processor's actors, in that order, each
   of time 0 ... Call k's part at the node, x_k, follows its submission (σ_k → x_k, delay 0).  A10's bound b is the edge
   from x_k to the submission b calls later ... no edge orders the parts themselves"; "The revisions of 55b9cbad"),
   reversing stop IPC-1 and the notes above where they differ:
   - The vertices: as above, and after call j's parts its submissions σ_{j,v} at each node v of f's plan but s, of
     time 0; at s call j's submission is σ_j, for every call, the sender submitting every call (A7).  The implementer's
     decision, recorded (the documents name "each node's submissions" and no set of them): node v's cycle holds the
     calls whose plan's `nodes` include v, those with a part x_{j,v} for σ_{j,v} to precede; a node's start of a call
     whose plan has no part there is an IPC-free actor of time 0 with no edge but its cycle's, and leaving it out
     changes no cycle's delay or mean.
   - The edges: each node's submissions in program order as BSL §4.1's cycle (delay 0 to the next, 1 from the last to
     the first), its row of the submissions keyed by node (mesh_graph_rows), s's σ cycle among them; β's cycle, σ_j ->
     β_j and β_j -> x_{j,v} as above; σ_{j,v} -> x_{j,v} at each node v of f's plan (at s, σ_j -> x_{j,s} as above);
     A10's x_{j_i,v} -> σ_{j_m,v} at each node v of f's plan, of the delay above, where it was x_{j_i,v} -> x_{j_m,v}
     at a node other than s; σ_{j_i} -> σ_{j_m} at s where s is no node of f's plan, as above, no part of the call
     being there to bound; the transfers as above.  No edge orders the parts themselves.
   - The edges' room: per call 4 (s's σ and β cycles, σ_j -> β_j, A10's σ -> σ), 3 a node of f's plan (β -> x, σ ->
     x, A10's x -> σ), 1 a submission at a node other than s (its cycle's), 2 a tree edge; Convert-to-SC-graph's 2 x
     vertices.
   - The added edges, the argument restated: σ_0 reaches every vertex (s's submissions' cycle, σ_j -> β_j, β_j ->
     x_{j,v}, x_{j,v} -> σ_{j_m,v} by A10, whose places (i + b) mod n are every place, and σ_{j,v} -> x_{j,v}), so its
     component is the one source SCC and Convert-to-SC-graph chains the sink SCCs alone.  A sink SCC holding s's σ_j
     holds what it reaches, every vertex (σ_j's cycle reaches σ_0); one holding β_j holds the parts β_j reaches, and
     so some σ_k at s, the one edge into β's cycle from outside it being σ_k -> β_k; either is then the whole graph,
     strongly connected, where nothing is added.  A sink SCC holding σ_{j,v} holds x_{j,v} (an edge from it).  So no
     sink SCC holds only a σ or a β, and each added edge's tail is a sink SCC's chosen vertex (mesh_graph_convert_sc:
     its least time, the lower vertex on a tie), a part x_{j,a} or a submission σ_{j,a} at a node a of f's plan other
     than s; its head the next sink SCC's chosen vertex, or the source SCC's, σ_0 (time 0, the lowest vertex; BS-2).  A head is a wait at its node before its call's start there (a part's at its
     submission, which it follows).  A tail's count is f's starts at a (BS-4): σ_{j,a}'s end is a's start of call j,
     place i, so the edge's delay in the rooms' graph is δ; x_{j,a}'s end is implied by its A10 successor's start
     (IG-3), place (i + b) mod n, and the delay in the rooms' graph is δ + ⌊(i + b) / n⌋ (`lead`, b for a part, 0 for a
     submission).
   - The entries (K1; mesh.h struct mesh_sync at 118): a wait of plan g = calls[j'] at b, `node` a, `function` f + 1,
     `every` n_f, `base` = position + 1 − δ x n_f (position the place above), `entry` 0, the waits of each plan at each
     node sorted by the place among g's calls of the call they enter, the added edges' order on a tie, so its k-th
     start, K = k / `calls`, waits on row k mod `calls` until `count` >= K x `every` + `base` (BSL (7)); a tell of plan
     f at a, `node` b, `function` g + 1, `entry` the edge's index among g's waits at b in that order (every rank
     computes the same graph, so the source computes the sink's index), `every` and `base` 0.  The groupings are
     mesh_graph_rows' (G1: "The added edges' grouping by plan is a hand-written count and fill ... where
     `mesh_graph_rows` does it"): the waits keyed by sink node and g's call place among the program's calls grouped by
     function, n x J keys (ENOMEM where they pass 32 bits), the tells here keyed by their plan.
   What must stay: the rooms as above; every rank's graph, delays and entries the same.
   Every rank computes the same graph, delays and edges from the same trees, u, weights, calls and operands, the graph
   module's arithmetic IEEE-exact.  0; ENOMEM, before anything is written. */
/* an edge of the IPC graph, a -> b of delay d, appended */
struct plan_room_edges { uint32_t *from,*to; uint64_t *delay; size_t edges; };
static void plan_room_edge(struct plan_room_edges *graph,uint32_t a,uint32_t b,uint64_t d){
  graph->from[graph->edges]=a;graph->to[graph->edges]=b;graph->delay[graph->edges]=d;graph->edges++;
}
struct plan_room_rows { const size_t *order; const uint32_t *to; const uint64_t *delay; };
static uint32_t plan_room_head(const void *context,size_t k){
  const struct plan_room_rows *rows=context;
  return rows->to[rows->order[k]];
}
static uint32_t plan_room_lag(const void *context,size_t k){
  const struct plan_room_rows *rows=context;
  const uint64_t delay=rows->delay[rows->order[k]];
  return delay>=UINT32_MAX?UINT32_MAX:(uint32_t)delay;
}
/* t(x_v) of the plan's actor at each node v (time[v]): the weighting operand's weights of v's tasks in one execution
   (each production at its source, each source read, each consumption at a destination, a page's task each for
   MESH_PAGES) and of its hops' pages, sent or received (BSL §4.1: "t(v) includes the time taken to execute all IPC
   operations (sends and receives) that the actor v performs"; item 2: "Execution times t(x) are A3's weighting
   operand"), a demand's page weight the greatest of its transfers' (GO2's); and `forwarded`, its demands' distinct
   tree edges carrying pages.  A tree's edges are its rows of vertices keyed by parent (mesh_graph_rows, as
   plan_template, plan_trees and plan_walk read them): no walk over a parent table here (C9).  2026-11-05 ("The
   revisions of 4e78f6f4"): plan_rooms' times for each plan, where plan_rooms summed them into its own graph's
   vertices; each tree edge's pages read from its parent's row, where plan_rooms read each vertex's parent from the
   table (the C9 hook's remark on c6c9c77f: "a walk over each tree's parent table ... it moves into the graph
   module").  What must stay: the sums, the same at every rank.  0; ENOMEM.
   2026-11-06 (R-1; "The revisions of 14b9a757", R-1): a MESH_PAGES consumption's tasks and the trees' hops counted in
   instructions (plan_instructions), where they were pages, as plan_schedule counts them (its note).  What must stay:
   the sums, the same at every rank */
static int plan_times(struct plan *p){
  const uint32_t n=p->n;
  size_t first[(size_t)n+2],child[(size_t)n+1];
  double *page=calloc((size_t)p->demands+1,sizeof *page);
  if(!page)return ENOMEM;
  for(uint32_t i=0;i<p->transfers;i++){
    const uint32_t d=p->of[i];
    if(p->weights[i].page>page[d])page[d]=p->weights[i].page;
    p->time[p->declared[i].destination]+=
      p->weights[i].consumed*(p->declared[i].kind==MESH_PAGES?(double)p->demand[d].instructions:1.0);
  }
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    p->time[d->source]+=p->weights[d->first].produced+(d->own?p->weights[d->first].read:0.0);
    for(uint32_t j=d->tree;j<d->tree+d->trees;j++){
      if(!p->run[j])continue;
      const struct plan_tree tree={p->parent+p->tree[j]*n,n,d->source};
      const double hops=(double)p->run[j]*page[i];
      mesh_graph_rows(n+1,n,plan_tree_parent,&tree,first,child);
      for(uint32_t y=0;y<n;y++)for(size_t c=first[y];c<first[y+1];c++){p->time[y]+=hops;p->time[child[c]]+=hops;}
    }
    p->forwarded+=d->forwards;
  }
  free(page);
  return 0;
}
/* The program's IPC graph and its rooms (the note above), over every plan of the call: `row` and `order` the program's
   calls grouped by function (mesh_graph_rows: function f's calls order[row[f]] .. order[row[f + 1] - 1], in program
   order), `self` this node's map index.  Out: each plan's `calls`, `room`, `submit_room`, `waits`, `tells`, `sync` and
   `wait_row` at this node */
/* call j's submission at node v of its plan's `nodes` (IG-5): σ_j, vertex 2j, at the sender s; else after the call's
   parts, at v's place among the plan's nodes but s */
static uint32_t plan_submission(const struct plan *p,const uint32_t *actor,uint32_t s,uint32_t j,uint32_t v){
  if(v==s)return 2*j;
  return actor[j]+(uint32_t)__builtin_popcountll(p->nodes)+plan_rank(p->nodes&~(UINT64_C(1)<<s),v);
}
static int plan_graph(struct plan *plans,uint32_t count,const uint32_t *calls,uint32_t ncalls,uint64_t program,
                      uint32_t self,const size_t *row,const size_t *order){
  const uint32_t s=(uint32_t)__builtin_ctzll(program),J=ncalls,n=plans[0].n;
  uint64_t vertices=2*(uint64_t)J,edges=0;
  uint32_t *place=malloc(((size_t)J+1)*sizeof *place),*actor=malloc(((size_t)J+1)*sizeof *actor);
  double *t=NULL;
  uint32_t *from=NULL,*to=NULL,*call=NULL,*node=NULL,*lead=NULL,*key=NULL,*added_from=NULL,*added_to=NULL,*entry=NULL;
  uint64_t *delay=NULL,*added_delay=NULL;
  size_t *first=NULL,*rows=NULL,*wait_first=NULL,*wait_order=NULL,*tell_first=NULL,*tell_order=NULL;
  int status=!place || !actor?ENOMEM:0;
  if(status)goto done;
  /* each call's place among its function's calls */
  for(uint32_t f=0;f<count;f++){
    plans[f].calls=(uint32_t)(row[f+1]-row[f]);
    for(size_t q=row[f];q<row[f+1];q++)place[order[q]]=(uint32_t)(q-row[f]);
  }
  /* each call's first part, its submissions at the nodes but s after its parts; the edges' room (the note above) */
  for(uint32_t j=0;j<J && vertices<UINT32_MAX;j++){
    const struct plan *p=plans+calls[j];
    const uint64_t parts=(uint64_t)__builtin_popcountll(p->nodes),
                   others=(uint64_t)__builtin_popcountll(p->nodes&~(UINT64_C(1)<<s));
    actor[j]=(uint32_t)vertices;
    vertices+=parts+others;
    edges+=4+3*parts+others+2*p->forwarded;
  }
  if(vertices>=UINT32_MAX/4 || edges>=SIZE_MAX/(8*sizeof(uint64_t)) || (uint64_t)n*J>=UINT32_MAX){
    status=ENOMEM;goto done;
  }
  const uint32_t V=(uint32_t)vertices;
  const size_t room=(size_t)edges+2*(size_t)V;
  t=calloc((size_t)V+1,sizeof *t);
  from=malloc((room+1)*sizeof *from);
  to=malloc((room+1)*sizeof *to);
  delay=malloc((room+1)*sizeof *delay);
  call=malloc(((size_t)V+1)*sizeof *call);
  node=malloc(((size_t)V+1)*sizeof *node);
  lead=malloc(((size_t)V+1)*sizeof *lead);
  key=malloc((2*(size_t)V+1)*sizeof *key);
  added_from=malloc((2*(size_t)V+1)*sizeof *added_from);
  added_to=malloc((2*(size_t)V+1)*sizeof *added_to);
  added_delay=malloc((2*(size_t)V+1)*sizeof *added_delay);
  entry=malloc((2*(size_t)V+1)*sizeof *entry);
  first=malloc(((size_t)V+n+2)*sizeof *first);
  rows=malloc((room+1)*sizeof *rows);
  wait_first=malloc(((size_t)n*J+1)*sizeof *wait_first);
  wait_order=malloc((2*(size_t)V+1)*sizeof *wait_order);
  tell_first=malloc(((size_t)count+2)*sizeof *tell_first);
  tell_order=malloc((2*(size_t)V+1)*sizeof *tell_order);
  if(!t || !from || !to || !delay || !call || !node || !lead || !key || !added_from || !added_to || !added_delay ||
     !entry || !first || !rows || !wait_first || !wait_order || !tell_first || !tell_order){status=ENOMEM;goto done;}
  /* the vertices: each one's call, node and time; `lead`, the calls of its function from its call to the call whose
     start at its node its end is implied by (a part's A10 bound b, a submission's 0); and its key in the submissions'
     rows, its node for a submission, n past every node for β and the parts */
  for(uint32_t j=0;j<J;j++){
    const struct plan *p=plans+calls[j];
    call[2*j]=call[2*j+1]=j;
    node[2*j]=node[2*j+1]=s;
    lead[2*j]=lead[2*j+1]=0;
    key[2*j]=s;key[2*j+1]=n;
    uint32_t k=actor[j];
    for(uint64_t set=p->nodes;set;set&=set-1,k++){
      call[k]=j;
      node[k]=(uint32_t)__builtin_ctzll(set);
      t[k]=p->time[node[k]];
      lead[k]=(uint32_t)p->outstanding;
      key[k]=n;
    }
    for(uint64_t set=p->nodes&~(UINT64_C(1)<<s);set;set&=set-1,k++){
      call[k]=j;
      node[k]=(uint32_t)__builtin_ctzll(set);
      lead[k]=0;
      key[k]=node[k];
    }
  }
  /* the edges: each node's submissions' cycle, in program order (its row keyed by node: s's σ_j at 2j, each other
     node's in its calls' parts' order, both the calls' order), then call by call β's cycle, the queues, the
     submissions' edges into the parts, A10's edges, the transfers */
  struct plan_room_edges graph={from,to,delay,0};
  mesh_graph_rows(n+1,V,plan_at,key,first,rows);
  for(uint32_t v=0;v<n;v++)for(size_t q=first[v];q<first[v+1];q++){
    const size_t next=q+1<first[v+1]?q+1:first[v];
    plan_room_edge(&graph,(uint32_t)rows[q],(uint32_t)rows[next],next==first[v]);
  }
  for(uint32_t j=0;j<J;j++){
    const uint32_t f=calls[j],i=place[j],next=j+1==J?0:j+1;
    const struct plan *p=plans+f;
    const uint64_t nf=p->calls,lag=(i+p->outstanding)/nf,span=(i+(uint64_t)p->copies)/nf;
    const uint32_t later=(uint32_t)order[row[f]+(i+p->outstanding)%nf],
                   ahead=(uint32_t)order[row[f]+(i+(uint64_t)p->copies)%nf];
    plan_room_edge(&graph,2*j+1,2*next+1,!next);
    plan_room_edge(&graph,2*j,2*j+1,0);
    for(uint64_t set=p->nodes;set;set&=set-1){
      const uint32_t v=(uint32_t)__builtin_ctzll(set),x=actor[j]+plan_rank(p->nodes,v);
      plan_room_edge(&graph,2*j+1,x,0);
      plan_room_edge(&graph,plan_submission(p,actor,s,j,v),x,0);
      plan_room_edge(&graph,x,plan_submission(p,actor,s,later,v),lag);
    }
    if(!(p->nodes>>s&1))plan_room_edge(&graph,2*j,2*later,lag);
    for(uint32_t d=0;d<p->demands;d++)for(uint32_t y=0;y<p->n;y++)
      for(uint64_t below=p->child[(size_t)d*p->n+y];below;below&=below-1){
        const uint32_t c=(uint32_t)__builtin_ctzll(below);
        plan_room_edge(&graph,actor[j]+plan_rank(p->nodes,y),actor[ahead]+plan_rank(p->nodes,c),span);
        plan_room_edge(&graph,actor[j]+plan_rank(p->nodes,c),actor[ahead]+plan_rank(p->nodes,y),span);
      }
  }
  const size_t built=graph.edges;
  double mean;
  uint32_t added;
  if((status=mesh_graph_cycle_mean(V,t,built,from,to,delay,&mean)))goto done;
  if(mean==INFINITY){status=ENOMEM;goto done;}
  if((status=mesh_graph_convert_sc(V,t,built,from,to,&added,added_from,added_to)) ||
     (status=mesh_graph_determine_delays(V,t,built,from,to,delay,added,added_from,added_to,added_delay))){
    if(status==EDEADLK || status==ERANGE)status=ENOMEM;
    goto done;
  }
  /* the added edges: each in the rooms' graph of delay δ + ⌊(i + lead) / n_f⌋ (BS-4's shift, a part's; a submission's
     0), keyed by its sink's node and the sink call's place among the program's calls grouped by function */
  for(uint32_t a=0;a<added;a++){
    const uint32_t tail=added_from[a],head=added_to[a],j=call[tail],h=call[head];
    if(__builtin_add_overflow(added_delay[a],(place[j]+(uint64_t)lead[tail])/plans[calls[j]].calls,&delay[built+a])){
      status=ENOMEM;goto done;
    }
    from[built+a]=tail;to[built+a]=head;
    key[a]=node[head]*J+(uint32_t)row[calls[h]]+place[h];
  }
  /* the waits in rows by sink node, plan and place (K1, G1), each edge's index there; the tells here by their plan */
  mesh_graph_rows(n*J,added,plan_at,key,wait_first,wait_order);
  for(uint32_t q=0;q<added;q++)entry[wait_order[q]]=q;
  for(uint32_t a=0;a<added;a++)key[a]=node[added_from[a]]==self?calls[call[added_from[a]]]:count;
  mesh_graph_rows(count+1,added,plan_at,key,tell_first,tell_order);
  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    const size_t *waits=wait_first+(size_t)self*J+row[f];
    p->waits=(uint32_t)(waits[p->calls]-waits[0]);
    p->tells=(uint32_t)(tell_first[f+1]-tell_first[f]);
    p->sync=malloc(((size_t)p->waits+p->tells+1)*sizeof *p->sync);
    p->wait_row=malloc(((size_t)p->calls+1)*sizeof *p->wait_row);
    if(!p->sync || !p->wait_row){status=ENOMEM;goto done;}
    for(uint32_t r=0;r<=p->calls;r++)p->wait_row[r]=(uint32_t)(waits[r]-waits[0]);
    for(uint32_t k=0;k<p->waits;k++){
      const uint32_t a=(uint32_t)wait_order[waits[0]+k],tail=added_from[a],j=call[tail];
      const uint32_t every=plans[calls[j]].calls;
      int64_t lagged;
      if(added_delay[a]>INT64_MAX || __builtin_mul_overflow((int64_t)added_delay[a],(int64_t)every,&lagged)){
        status=ENOMEM;goto done;
      }
      p->sync[k]=(struct plan_sync){.node=node[tail],.function=calls[j]+1,.every=every,
                                    .base=(int64_t)((place[j]+(uint64_t)lead[tail])%every)+1-lagged};
    }
    for(uint32_t k=0;k<p->tells;k++){
      const uint32_t a=(uint32_t)tell_order[tell_first[f]+k],head=added_to[a],g=calls[call[head]];
      p->sync[p->waits+k]=(struct plan_sync){.node=node[head],.function=g+1,
                                             .entry=(uint32_t)(entry[a]-wait_first[(size_t)node[head]*J+row[g]])};
    }
  }
  /* the rooms, (5) in the converted graph: per call, ρ(β_j, σ_j) and ρ(x_{j,self}, β_j) */
  mesh_graph_rows(V,built+added,plan_at,from,first,rows);
  const struct plan_room_rows rooms={rows,to,delay};
  for(uint32_t f=0;f<count;f++){plans[f].room=plans[f].nodes>>self&1?0:1;plans[f].submit_room=0;}
  for(uint32_t j=0;j<J;j++){
    struct plan *p=plans+calls[j];
    const int here=(int)(p->nodes>>self&1);
    const uint32_t source[2]={2*j+1,here?actor[j]+plan_rank(p->nodes,self):2*j+1},target[2]={2*j,2*j+1};
    uint32_t lags[4];
    if((status=mesh_graph_lags(V,first,plan_room_head,plan_room_lag,&rooms,2,source,2,target,lags)))goto done;
    if(lags[0]==UINT32_MAX || !lags[0] || (here && (lags[3]==UINT32_MAX || !lags[3]))){status=ENOMEM;goto done;}
    p->submit_room+=lags[0];
    if(here)p->room+=lags[3];
  }
  for(uint32_t f=0;f<count;f++)
    if(plans[f].room>=UINT64_MAX/(4*sizeof(struct mesh_delivery)) ||
       plans[f].submit_room>=UINT64_MAX/(4*sizeof(struct mesh_submission))){status=ENOMEM;goto done;}
done:
  free(place);free(actor);free(t);free(from);free(to);free(delay);free(call);free(node);free(lead);free(key);
  free(added_from);free(added_to);free(added_delay);free(entry);free(first);free(rows);free(wait_first);
  free(wait_order);free(tell_first);free(tell_order);
  return status;
}

/* -- the transfers' instructions and the frames' items (design/plan-to-bridge-change-request.md R1 at mesh 56ede97e,
   "What the code must do instead"; design/compilation-coordination.md, "The root cause of the copies, and the revised
   work list", R-1; "The revisions of 14b9a757", R-1) ------------------------------------------------------------- */

/* A view's bytes and its last extent's end (mesh-collective.h struct mesh_strided): *bytes its element x its sizes'
   product, *end its offset plus each dimension's last index x its stride plus its element, past which none of its
   extents ends (every stride nonnegative).  0; EINVAL where it is malformed (mesh-collective.h: "`element` >= 1, every
   size[d] >= 1 ... EINVAL otherwise, at the plan") or a product or sum passes 64 bits, before anything is written */
static int plan_view(const struct mesh_strided *v,uint64_t *bytes,uint64_t *end){
  uint64_t b=v->element,e;
  if(!b || __builtin_add_overflow(v->offset,v->element,&e))return EINVAL;
  for(uint64_t d=0;d<v->dims;d++){
    const uint64_t size=v->pairs[2*d],stride=v->pairs[2*d+1];
    uint64_t step;
    if(!size || __builtin_mul_overflow(b,size,&b) || __builtin_mul_overflow(size-1,stride,&step) ||
       __builtin_add_overflow(e,step,&e))return EINVAL;
  }
  *bytes=b;*end=e;
  return 0;
}
/* two views alike (mesh-collective.h struct mesh_declared: `from` "the same in every declared transfer of one range"):
   their offsets, elements and dims, and their `dims` (size, stride) pairs (struct mesh_strided `pairs`) */
static int plan_alike(const struct mesh_strided *a,const struct mesh_strided *b){
  if(a->offset!=b->offset || a->element!=b->element || a->dims!=b->dims)return 0;
  for(uint64_t k=0;k<2*(uint64_t)a->dims;k++)if(a->pairs[k]!=b->pairs[k])return 0;
  return 1;
}
/* A view flattened (R-1: "The compilation flattens each view into its list of (row, offset, bytes) blocks, adjacent
   blocks merged, as datatype processing flattens a type map"; R. Ross, N. Miller and W. D. Gropp, Implementing fast and
   reusable datatype processing, EuroPVM/MPI 2003, as PB R1 cites them): visit(context, offset, length) for each of its
   extents of the value's bytes, in its C index order (struct mesh_strided: "the last dimension fastest"), each two
   adjacent ones merged.  Its trailing dimensions that are contiguous (a size of 1, or a stride equal to the bytes the
   dimensions after it span) are one run, its element's bytes times their sizes, as datatype processing collapses a
   contiguous inner type into one block; the indices of the dimensions before them are counted in C order, each index
   a run at its offset, and a run that begins where the extent before it ends is merged into it.  The view is well
   formed (plan_view), so no sum passes its end */
static void plan_extents(const struct mesh_strided *v,void (*visit)(void *,uint64_t,uint64_t),void *context){
  const uint64_t *pair=v->pairs;
  int64_t d=(int64_t)v->dims-1;
  uint64_t run=v->element;
  for(;d>=0 && (pair[2*d]==1 || pair[2*d+1]==run);d--)run*=pair[2*d];
  uint64_t index[(size_t)v->dims+1],at=v->offset,start=at,length=0;
  memset(index,0,sizeof index);
  for(;;){
    if(length && start+length==at)length+=run;
    else{
      if(length)visit(context,start,length);
      start=at;length=run;
    }
    int64_t e=d;
    for(;e>=0 && ++index[e]==pair[2*e];e--){at-=(pair[2*e]-1)*pair[2*e+1];index[e]=0;}
    if(e<0)break;
    at+=pair[2*e+1];
  }
  visit(context,start,length);
}
/* plan_extents' visits in plan_instructions, over one view of a range, `range` the range's bytes its extents before
   this one hold: each extent cut at the multiples of `quantum` in its value's bytes (mesh-graph.h mesh_graph_grid_cut,
   into `start`, `count`, `index`), its pieces counted (`cuts` NULL: mesh_graph_grid_cut's runs, (offset + length - 1) /
   quantum - offset / quantum + 1) or each piece's first byte given as a byte of the range at `cuts` */
struct plan_cutting { uint64_t quantum,range,made,*cuts,*start,*count,*index; };
static void plan_cut(void *context,uint64_t offset,uint64_t length){
  struct plan_cutting *x=context;
  if(!x->cuts)x->made+=(offset+length-1)/x->quantum-offset/x->quantum+1;
  else{
    const uint32_t pieces=mesh_graph_grid_cut(offset,length,x->quantum,x->start,x->count,x->index);
    for(uint32_t k=0;k<pieces;k++)x->cuts[x->made++]=x->range+x->start[k]-offset;
  }
  x->range+=length;
}
/* and, a range's instructions' bounds known (`cut`, `instructions` of them), each instruction's first byte in the
   view's value (`at`, from instruction `next` on): the k-th lies in the extent holding range byte cut[k], at its
   offset there */
struct plan_starting { const uint64_t *cut; uint64_t *at,range,next,instructions; };
static void plan_start(void *context,uint64_t offset,uint64_t length){
  struct plan_starting *x=context;
  for(;x->next<x->instructions && x->cut[x->next]<x->range+length;x->next++)
    x->at[x->next]=offset+x->cut[x->next]-x->range;
  x->range+=length;
}
static int plan_cut_order(const void *a,const void *b){
  const uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b;
  return (x>y)-(x<y);
}
/* Each demand's instructions (mesh-collective.h, the note before struct mesh_value: "Its instructions are the common
   refinement of the source's and every destination's extents, each cut at its value's block boundaries, and of the
   range's own bytes cut at every multiple of a block's bytes (block x pgsz), so each instruction is one extent within
   one block at its source, at each destination and at a relay"):
   - the cuts, bytes of the range's B in its source view's order: each extent of its source's view (`from`) and of each
     of its declared transfers' destinations' views (`to`, byte i of `from` landing at byte i of `to`), cut at the
     multiples of Q = block x pgsz in its value's bytes (mesh_graph_grid_cut), each piece's first byte; the range's own
     bytes [0, B) cut at the multiples of Q likewise; and B;
   - the instructions, the common refinement of those partitions of [0, B): the cuts ascending, each once, instruction
     k the range's bytes [cut[k], cut[k + 1]), P of them.  Each lies within one extent and one block of every view, so
     within one block at its source, at each destination and at a relay, whose range byte s is at its block s / Q,
     offset s mod Q (the header: "A relay holds the range in its own bytes' order");
   - where this node is the range's source or a destination, each instruction's first byte in its value `mine`, from
     its view here (`view`, plan_start).
   Every rank computes the same instructions from the same views (every declared transfer's `from` and `to` is every
   rank's: struct mesh_declared, "both every rank's alike"; aot A7, corrected 2026-11-06, F. B. Schneider, ACM Comput.
   Surv. 22(4), 1990), so both ends' lists hold the same hops.  Each demand's cuts are bounded first (each extent's
   pieces), so one allocation holds every demand's.  0; ENOMEM (P past 32 bits, or the call's own memory) */
static int plan_instructions(struct plan *p,const struct hdr *m){
  const uint64_t quantum=plan_quantum(m);
  const uint32_t count=p->transfers;
  size_t *first=malloc(((size_t)p->demands+1)*sizeof *first),*order=malloc(((size_t)count+1)*sizeof *order);
  uint64_t *start=NULL,*pieces=NULL,*index=NULL,room=0,most=0,ends=0,at=0;
  int status=!first || !order?ENOMEM:0;
  if(!status){
    /* each demand's declared transfers (mesh-graph.h mesh_graph_rows, keyed by demand) and its cuts' bound */
    mesh_graph_rows(p->demands,count,plan_at,p->of,first,order);
    for(uint32_t i=0;i<p->demands;i++){
      const struct plan_demand *d=p->demand+i;
      const uint64_t blocks=plan_blocks(m,d->bytes);
      struct plan_cutting cutting={.quantum=quantum};
      plan_extents(&p->declared[d->first].from,plan_cut,&cutting);
      for(size_t q=first[i];q<first[i+1];q++)plan_extents(&p->declared[order[q]].to,plan_cut,&cutting);
      room+=cutting.made+blocks+1;
      if(blocks>most)most=blocks;
    }
    p->cut=malloc((room+1)*sizeof *p->cut);
    start=malloc((most+2)*sizeof *start);
    pieces=malloc((most+2)*sizeof *pieces);
    index=malloc((most+2)*sizeof *index);
    if(!p->cut || !start || !pieces || !index)status=ENOMEM;
  }
  for(uint32_t i=0;!status && i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    struct plan_cutting cutting={quantum,0,0,p->cut+at,start,pieces,index};
    plan_extents(&p->declared[d->first].from,plan_cut,&cutting);
    for(size_t q=first[i];q<first[i+1];q++){cutting.range=0;plan_extents(&p->declared[order[q]].to,plan_cut,&cutting);}
    const uint32_t own=mesh_graph_grid_cut(0,d->bytes,quantum,start,pieces,index);
    for(uint32_t k=0;k<own;k++)cutting.cuts[cutting.made++]=start[k];
    cutting.cuts[cutting.made++]=d->bytes;
    qsort(cutting.cuts,cutting.made,sizeof *cutting.cuts,plan_cut_order);
    uint64_t made=1;
    for(uint64_t k=1;k<cutting.made;k++)if(cutting.cuts[k]!=cutting.cuts[made-1])cutting.cuts[made++]=cutting.cuts[k];
    d->cut=at;d->instructions=made-1;at+=made;
    if(d->instructions>=UINT32_MAX)status=ENOMEM;
    if(d->mine!=MESH_ABSENT)ends+=d->instructions;
  }
  if(!status && !(p->at=malloc((ends+1)*sizeof *p->at)))status=ENOMEM;
  at=0;
  for(uint32_t i=0;!status && i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    if(d->mine==MESH_ABSENT)continue;
    struct plan_starting starting={p->cut+d->cut,p->at+at,0,0,d->instructions};
    plan_extents(d->view,plan_start,&starting);
    d->at=at;at+=d->instructions;
  }
  free(first);free(order);free(start);free(pieces);free(index);
  return status;
}
/* MPI 4.1 §6.1.11: "A datatype may specify overlapping entries.  The use of such a datatype in any communication in
   association with a buffer updated by the operation is erroneous."  This node's destination instructions, each the
   bytes [at, at + its bytes) of the value it lands in, by mesh-graph.h mesh_graph_intersect.  0; EINVAL; ENOMEM */
static int plan_landings(const struct plan *p){
  uint64_t count=0,k=0;
  for(uint32_t i=0;i<p->demands;i++)
    if(p->demand[i].mine!=MESH_ABSENT && p->demand[i].leaves>>p->self&1)count+=p->demand[i].instructions;
  struct mesh_graph_interval *interval=malloc((count+1)*sizeof *interval);
  if(!interval)return ENOMEM;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    if(d->mine==MESH_ABSENT || !(d->leaves>>p->self&1))continue;
    for(uint64_t j=0;j<d->instructions;j++){
      const uint64_t at=p->at[d->at+j];
      interval[k++]=(struct mesh_graph_interval){d->mine,at,at+p->cut[d->cut+j+1]-p->cut[d->cut+j]};
    }
  }
  const int status=mesh_graph_intersect(count,interval)?EINVAL:0;
  free(interval);
  return status;
}

/* an instruction of a range at one of its ends here (plan_sets): bytes [start, end) of values[value], of the range whose
   routed value here is `member` (p->here's index), by value and first byte */
struct plan_touch { uint64_t start,end; uint32_t value,member; };
static int plan_touch_order(const void *a,const void *b){
  const struct plan_touch *x=a,*y=b;
  if(x->value!=y->value)return x->value<y->value?-1:1;
  return (x->start>y->start)-(x->start<y->start);
}
static int plan_member_order(const void *a,const void *b){
  const uint32_t x=*(const uint32_t *)a,y=*(const uint32_t *)b;
  return (x>y)-(x<y);
}
/* a block a set of ranges here covers whole (plan_sets): block `block` of values[value], its set the `members` routed
   values here at `member`, ascending; by its set, then by value and block, so each set's blocks are consecutive */
struct plan_cover { const uint32_t *member; uint32_t members,value,block; };
static int plan_same_set(const struct plan_cover *x,const struct plan_cover *y){
  return x->members==y->members && !memcmp(x->member,y->member,(size_t)x->members*sizeof *x->member);
}
static int plan_cover_order(const void *a,const void *b){
  const struct plan_cover *x=a,*y=b;
  for(uint32_t k=0;k<x->members && k<y->members;k++)
    if(x->member[k]!=y->member[k])return x->member[k]<y->member[k]?-1:1;
  if(x->members!=y->members)return x->members<y->members?-1:1;
  if(x->value!=y->value)return x->value<y->value?-1:1;
  return (x->block>y->block)-(x->block<y->block);
}
/* The frames' items here (R-1: "Two values' rows may name one frame, the page table's sharing (fbufs; IO-Lite).  A
   frame's item lives from the first write to the last read over every value naming it, with C3.3's lag test";
   mesh-collective.h, the note before struct mesh_value), after the least lags (plan_unroll):
   - a relay of a range (this node on its trees, neither its source nor a destination) keeps its routed value's item, a
     set of one member, ceil(B / Q) blocks a copy, its instruction k's bytes at block cut[k] / Q, offset cut[k] mod Q
     (plan_instructions), as before;
   - at a source or a destination a range has no blocks of its own: its instructions lie in its value's blocks.  For each
     block of a MESH_EXECUTION value a range here names, the set of this node's ranges whose instructions touch it (their
     routed values here, ascending).  Where the union of those instructions' bytes is every byte of the block within the
     value's bytes, the block is an element of that set's item, one item a set and copy (u copies), its elements every
     such block of the set in (value, block) order; where any byte of it is no instruction's, it stays in the value's own
     item (`uncovered`, its element its place among them), which lives the program's whole life (PLAN_ALONE: the
     program's own writes and reads of its other bytes are ordered by no edge of the template);
   - a MESH_INVARIANT source has one copy for the plan's life, every block of it its own item, PLAN_ALONE (mesh-
     collective.h: "one copy for the plan's life, which every copy's send reads"), and every copy's sends name copy 0's
     frames (mesh_plan).
   E_s, `set_far`, is the largest farthest[a] over its members (plan_before).  The union of a block's instructions is
   counted over them by first byte (each lies within one block: plan_instructions), each adding its bytes past the
   greatest end before it.  The coordinating implementer's design ("The revisions of 14b9a757", brief C, R-1).  0;
   ENOMEM */
static int plan_sets(struct plan *p,const struct hdr *m){
  const uint64_t quantum=plan_quantum(m);
  uint64_t blocks=0,touches=0,relays=0;
  if(!(p->block_first=malloc(((size_t)p->givens+1)*sizeof *p->block_first)))return ENOMEM;
  for(uint32_t j=0;j<p->givens;j++){
    p->block_first[j]=MESH_ABSENT;
    if(!p->named[j] || p->given[j].kind!=MESH_EXECUTION)continue;
    p->block_first[j]=(uint32_t)blocks;
    if((blocks+=plan_blocks(m,p->given[j].section.bytes))>=UINT32_MAX)return ENOMEM;
  }
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    if(d->value==MESH_ABSENT)continue;
    if(d->mine==MESH_ABSENT)relays++;
    else if(p->block_first[d->mine]!=MESH_ABSENT)touches+=d->instructions;
  }
  struct plan_touch *touch=malloc((touches+1)*sizeof *touch);
  struct plan_cover *cover=malloc((blocks+1)*sizeof *cover);
  uint32_t *held=malloc((touches+1)*sizeof *held);
  p->block_set=malloc((blocks+1)*sizeof *p->block_set);
  p->block_element=malloc((blocks+1)*sizeof *p->block_element);
  int status=!touch || !cover || !held || !p->block_set || !p->block_element?ENOMEM:0;
  if(status)goto done;
  uint64_t t=0,covers=0,members=0;
  for(uint32_t i=0;i<p->demands;i++){
    const struct plan_demand *d=p->demand+i;
    if(d->mine==MESH_ABSENT || p->block_first[d->mine]==MESH_ABSENT)continue;
    for(uint64_t k=0;k<d->instructions;k++){
      const uint64_t at=p->at[d->at+k];
      touch[t++]=(struct plan_touch){at,at+p->cut[d->cut+k+1]-p->cut[d->cut+k],d->mine,d->value};
    }
  }
  qsort(touch,touches,sizeof *touch,plan_touch_order);
  for(uint64_t b=0;b<blocks;b++)p->block_set[b]=MESH_ABSENT;
  /* each touched block's bytes' union, and its set where that is every byte of it */
  for(uint64_t q=0;q<touches;){
    const uint32_t j=touch[q].value;
    const uint64_t b=touch[q].start/quantum,low=b*quantum,bytes=p->given[j].section.bytes,
                   whole=bytes-low<quantum?bytes-low:quantum,from=members;
    uint64_t reach=low,covered=0;
    for(;q<touches && touch[q].value==j && touch[q].start/quantum==b;q++){
      if(touch[q].end>reach){covered+=touch[q].end-(touch[q].start>reach?touch[q].start:reach);reach=touch[q].end;}
      held[members++]=touch[q].member;
    }
    if(covered!=whole){members=from;continue;}
    qsort(held+from,members-from,sizeof *held,plan_member_order);
    uint64_t kept=from+1;
    for(uint64_t k=from+1;k<members;k++)if(held[k]!=held[kept-1])held[kept++]=held[k];
    members=kept;
    cover[covers++]=(struct plan_cover){held+from,(uint32_t)(members-from),j,(uint32_t)b};
  }
  /* the sets: the covered blocks' by their members, then each relay's */
  qsort(cover,covers,sizeof *cover,plan_cover_order);
  uint64_t sets=relays,listed=relays;
  for(uint64_t c=0;c<covers;c++)if(!c || !plan_same_set(cover+c-1,cover+c)){sets++;listed+=cover[c].members;}
  if(sets>=UINT32_MAX || listed>=UINT32_MAX){status=ENOMEM;goto done;}
  p->set_first=malloc((sets+1)*sizeof *p->set_first);
  p->member=malloc((listed+1)*sizeof *p->member);
  p->set_blocks=malloc((sets+1)*sizeof *p->set_blocks);
  p->set_far=malloc((sets+1)*sizeof *p->set_far);
  if(!p->set_first || !p->member || !p->set_blocks || !p->set_far){status=ENOMEM;goto done;}
  uint32_t s=0,at=0;
  for(uint64_t c=0;c<covers;c++){
    if(!c || !plan_same_set(cover+c-1,cover+c)){
      p->set_first[s]=at;p->set_blocks[s]=0;
      for(uint32_t k=0;k<cover[c].members;k++)p->member[at++]=cover[c].member[k];
      s++;
    }
    const uint32_t b=p->block_first[cover[c].value]+cover[c].block;
    p->block_set[b]=s-1;p->block_element[b]=p->set_blocks[s-1]++;
  }
  for(uint32_t i=0;i<p->demands;i++){
    struct plan_demand *d=p->demand+i;
    d->relay=MESH_ABSENT;
    if(d->value==MESH_ABSENT || d->mine!=MESH_ABSENT)continue;
    d->relay=s;p->set_first[s]=at;p->member[at++]=d->value;p->set_blocks[s++]=(uint32_t)plan_blocks(m,d->bytes);
  }
  p->set_first[s]=at;p->sets=s;
  for(uint32_t x=0;x<s;x++){
    p->set_far[x]=0;
    for(uint32_t k=p->set_first[x];k<p->set_first[x+1];k++)
      if(p->farthest[p->member[k]]>p->set_far[x])p->set_far[x]=p->farthest[p->member[k]];
  }
  /* each named MESH_EXECUTION value's blocks no set holds, its own item's in block order */
  for(uint32_t j=0;j<p->givens;j++){
    if(p->block_first[j]==MESH_ABSENT)continue;
    const uint64_t own=plan_blocks(m,p->given[j].section.bytes);
    uint32_t alone=0;
    for(uint64_t k=0;k<own;k++)
      if(p->block_set[p->block_first[j]+k]==MESH_ABSENT)p->block_element[p->block_first[j]+k]=alone++;
    p->uncovered[j]=alone;
  }
done:
  free(touch);free(cover);free(held);
  return status;
}

/* One compiled function's plan (mesh-collective.h struct mesh_function), computed whole before anything is written:
   its operands' tests, its demands, routing, pages, the trees' communications, the template, no lag-0 cycle, the least
   lags and u, this node's entries of one copy counted, its nodes and its actors' times.  The demands: the declared
   transfers grouped by source and identity, a range to each of its destinations, numbered by their first declared
   transfer (design/compilation-coordination.md, "The documents' revisions of edd4b583", GO5: "The demands stay
   numbered by their first declared transfer (T6's other half, not an order of the lists)").  The routing: every
   demand at once over the map's node graph (a node pair an edge where the map gives it a capacity both ways, of the
   lesser of the two: mesh-dataflow.h struct mesh_nodes `capacity`, D.2's node pairs and their capacities from the
   map, and nothing else of how the bridge carries pages), a destination its source reaches by no path refused
   (mesh-graph.h mesh_graph_reach), by the concurrent-flow solve (mesh_graph_flow, Garg and Koenemann's multiplicative
   weights, the tag's mesh_solve rounds and room), demand d a tree rooted at its source reaching its destinations, its
   bytes down from the root: each demand's trees and their shares, a tree's vertices past its source and destinations
   the relays, members all.  The pages: each demand's pages cut over its trees by their shares by earliest finish
   (mesh_graph_fill), a tree's pages one run of them.  The sets: a demand whose every declared transfer is MESH_PAGES is
   a set on each of its trees (S1, S2; its `set`).  Every rank computes the same demands, trees, pages, template and u
   from the same declared transfers, weights, dependences and map.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5; "The revisions
   of 4e78f6f4"): mesh_plan's computation of its one function, for each function of the program's call, where
   mesh_plan took one function's operands as its own arguments.  What must stay: every test and computation as it was,
   every failure before anything is written.  0, or mesh_plan's errors before its writes. */
static int plan_function(struct plan *p,const struct mesh_function *function,const struct mesh_nodes *nodes,
                         uint64_t program){
  struct hdr *m=p->context->M;
  const struct mesh_declared *declared=function->declared;
  const struct mesh_weight *weights=function->weights;
  const struct mesh_dependence *dependence=function->dependence;
  struct mesh_value *values=function->value;
  const uint32_t count=function->count,given=function->values,dependences=function->dependences;
  if((count && (!declared || !weights)) || (given && !values) || (dependences && !dependence))return EINVAL;
  /* a value: its bytes, its kind, its wire and state flags, and its lag (k >= 1 only on an execution value outside the
     window that is no state: design/disjoint-pages-change-request.md, D.3.3; design/disjoint-pages-rulings-
     coordination.md, R4; that no declared transfer names it is the declared transfers' test below; 2026-11-06, R-1,
     "The revisions of 14b9a757": that test goes, "lag-k named by no declared transfer", and struct mesh_function's
     `share` with it).
     2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 6 and 8; "The
     revisions of 4e78f6f4"): and its `run`, 0 or 1 (mesh-collective.h struct mesh_value).  What must stay: the tests
     above */
  for(uint32_t j=0;j<given;j++){
    const struct mesh_value *v=values+j;
    if(!v->section.bytes || v->kind>MESH_EXECUTION || v->wire>1 || v->state>1 || v->run>1 ||
       (v->state && (v->wire || v->kind!=MESH_INVARIANT)) ||
       (v->lag && (v->kind!=MESH_EXECUTION || v->wire || v->state)))return EINVAL;
    if(plan_blocks(m,v->section.bytes)>=UINT32_MAX/m->block)return ENOMEM;
  }
  /* a declared transfer's kind (design/plan-to-bridge-change-request.md, R4) and its weights (design/aot-compilation-
     change-request.md, A3), each finite and > 0 (design/compilation-coordination.md, C-weights): every task's,
     `produced`, `consumed` and `read`, as Sinnen and Sousa's Definition 1 states ("The positive weight w(n)"), and
     every `page`, so every occupation of a link lasts > 0.
     2026-11-04, owner C2 (review W3 #6, the coordinator's ruling): withdrawn, the reason that with it each link's
     order of starts is a topological order (R5's no-cycle argument; MSCCLang, ASPLOS 2023, §5.2): the contention
     schedule starts a page on a route's links by Sinnen and Sousa's eq. (14), nonaligned (cut-through), so a page's
     start on a link can precede its finish on the link before, and the start order is not shown to be a topological
     order of the bridge's store-and-forward waits.  Open for the operator (design/compilation-coordination.md, the
     appendix's revisions, record).  The EINVAL stays.
     2026-11-04, owner C3 (design/compilation-coordination.md, "The documents' revisions of edd4b583", GO4): no list is
     ordered by the starts now; the no-cycle argument is GO4's (plan_schedule), over the instruction DAG's
     store-and-forward dependencies, and it rests on no weight.  So `page` > 0 keeps no reason from it; the EINVAL
     stays, as C-weights decided it (the open item "No cycle on a cable", where the refusal "stays").
     2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): GO4's argument went with its
     groups; the argument is R5's own for the places' order (plan_schedule), and it rests on no weight either.  The
     EINVAL stays, as above.
     2026-11-04, owner C: reversing >= 0 for every weight.  Why: C-weights (reviews V1 #8, V3 #5).  What must stay:
     EINVAL before anything is written.
     2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP): Sinnen and Sousa's Definition 1
     is no longer this file's reason, the contention schedule being removed (R5's linear program takes any length d_j
     >= 0); the EINVAL stays, as C-weights decided it, since R5 asks no change of the operand's test.  What must stay:
     as above */
  for(uint32_t i=0;i<count;i++){
    const struct mesh_weight *x=weights+i;
    if(declared[i].kind>MESH_PAGES || !(x->produced>0 && x->produced<=DBL_MAX) ||
       !(x->consumed>0 && x->consumed<=DBL_MAX) || !(x->read>0 && x->read<=DBL_MAX) ||
       !(x->page>0 && x->page<=DBL_MAX))return EINVAL;
  }
  /* a program's data edge: its two declared transfers, its kinds (no end both produced and its source's own read),
     both events at one node, a lag a least lag can hold (mesh_graph_lags: UINT32_MAX is no walk) */
  const uint32_t kinds=MESH_FROM_PRODUCED|MESH_TO_PRODUCED|MESH_FROM_SOURCE_READ|MESH_TO_SOURCE_READ;
  for(uint32_t e=0;e<dependences;e++){
    const struct mesh_dependence *x=dependence+e;
    const int from_produced=!!(x->kinds&MESH_FROM_PRODUCED),from_own=!!(x->kinds&MESH_FROM_SOURCE_READ),
              to_produced=!!(x->kinds&MESH_TO_PRODUCED),to_own=!!(x->kinds&MESH_TO_SOURCE_READ);
    if(x->from>=count || x->to>=count || x->kinds&~kinds || (from_produced && from_own) || (to_produced && to_own) ||
       x->lag==UINT32_MAX || plan_event_node(declared,x->from,from_produced,from_own)!=
       plan_event_node(declared,x->to,to_produced,to_own))return EINVAL;
  }
  if(!function->outstanding || function->outstanding>=UINT32_MAX)return ENOMEM;
  const uint32_t n=nodes->count;
  p->declared=declared;p->transfers=count;p->given=values;p->givens=given;p->weights=weights;
  p->dependence=dependence;p->dependences=dependences;p->outstanding=function->outstanding;
  struct plan_key *key=malloc(((size_t)count+1)*sizeof *key);
  uint32_t *root=NULL,*demand=NULL,*placed=malloc(((size_t)count+1)*sizeof *placed);
  size_t *trees=NULL;
  double *up=NULL,*down=NULL,*share=NULL,*rate=NULL,*latency=NULL;
  int status=0;
  p->named=calloc((size_t)given+1,sizeof *p->named);
  p->uncovered=malloc(((size_t)given+1)*sizeof *p->uncovered);
  p->item=malloc(((size_t)given+1)*sizeof *p->item);
  p->demand=calloc((size_t)count+1,sizeof *p->demand);
  p->of=malloc(((size_t)count+1)*sizeof *p->of);
  if(!key || !placed || !p->named || !p->uncovered || !p->item || !p->demand || !p->of){status=ENOMEM;goto done;}
  for(uint32_t j=0;j<given;j++)p->uncovered[j]=(uint32_t)plan_blocks(m,values[j].section.bytes);
  /* 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
     mesh-collective.h struct mesh_strided and struct mesh_declared; "The revisions of 14b9a757", R-1), reversing the
     test of `bytes`: each declared transfer's two views well formed (plan_view: "`element` >= 1, every size[d] >=
     1"), naming as many bytes ("The two name as many bytes (EINVAL otherwise)"),
     at every rank, before anything is written.  What must stay: EHOSTUNREACH for an end outside the map; EINVAL for a
     transfer to its own source; ENOMEM for a range whose blocks pass 32 bits */
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+i;
    uint64_t from,to,end;
    if(t->source>=n || t->destination>=n){status=EHOSTUNREACH;goto done;}
    /* a transfer to its own source passes between no node pair, and nothing is copied */
    if(t->source==t->destination || plan_view(&t->from,&from,&end) || plan_view(&t->to,&to,&end) || from!=to){
      status=EINVAL;goto done;
    }
    if(plan_blocks(m,from)>=UINT32_MAX/m->block){status=ENOMEM;goto done;}
    key[i]=(struct plan_key){t->source,t->identity,i};
  }
  qsort(key,count,sizeof *key,plan_key_order);
  /* each declared transfer's range by its first transfer: key order holds a range's transfers together, each range's
     in declared order */
  for(uint32_t k=0;k<count;k++)
    placed[key[k].index]=!k || key[k].source!=key[k-1].source || key[k].identity!=key[k-1].identity?key[k].index:
                         placed[key[k-1].index];
  /* the demands numbered by their first declared transfer (design/compilation-coordination.md, "The documents'
     revisions of edd4b583", GO5, T6's other half): one opened at each
     transfer that is its range's first, in declared order */
  for(uint32_t i=0;i<count;i++){
    const struct mesh_declared *t=declared+i;
    uint64_t bytes,end;
    if(placed[i]==i){
      plan_view(&t->from,&bytes,&end);
      p->demand[p->demands++]=(struct plan_demand){.source=t->source,.identity=t->identity,.first=i,.mine=MESH_ABSENT,
                                                   .set=1,.bytes=bytes,.range=MESH_ABSENT,.relay=MESH_ABSENT};
    }
    p->of[i]=placed[i]==i?p->demands-1:p->of[placed[i]];
    struct plan_demand *d=p->demand+p->of[i];
    /* a set where every declared transfer of the range is MESH_PAGES, function B at every destination; any MESH_JOIN
       destination keeps the schedule's order (design/compilation-coordination.md, S1; design/aot-compilation-change-
       request.md, appendix item 5: "Function A's operands (joins) keep the schedule's order"), each of its pages at its
       own place in R5's one global order (R5: "A set's pages ... keep their places in that order") */
    if(t->kind!=MESH_PAGES)d->set=0;
    /* one range: its bytes alike, each destination once, its production and its source's read weighing alike
       (design/compilation-coordination.md, "The plan call", struct mesh_weight).  2026-11-06 (R-1; "The revisions of
       14b9a757"): its `from` alike (mesh-collective.h struct mesh_declared: "the same in every declared transfer of one
       range (EINVAL otherwise)"), where its `bytes` were.  What must stay: the other tests */
    if(!plan_alike(&t->from,&declared[d->first].from) || d->leaves>>t->destination&1 ||
       weights[i].produced!=weights[d->first].produced || weights[i].read!=weights[d->first].read){
      status=EINVAL;goto done;
    }
    d->leaves|=UINT64_C(1)<<t->destination;
    /* `value`: MESH_ABSENT where this node is neither the transfer's source nor its destination; else an execution
       value in the window of lag 0 (a value of lag k >= 1 is written by a producer the program orders by an event,
       which no declared transfer is: R4), the source's one value of every transfer of the range.
       2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 7; "The
       revisions of 4e78f6f4"): the range's blocks [first, first + P) within the value's, and the source's one `first`
       of every transfer of the range, where the value's bytes were the range's and "one range's alone".  Why: point
       7, "name, in each declared transfer, the pages of its destination value that it lands in, as NCCL's receives
       land at offsets of one result ... no concatenation and no copy of a part remains"; mesh-collective.h struct
       mesh_declared `first`.  What must stay: MESH_ABSENT away from the range's ends; an execution value in the
       window of lag 0
       2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
       design/plan-to-bridge-change-request.md R2, corrected 2026-11-06: "The plan call's rules that a lag-k value is
       "named by no declared transfer", that a transfer's value is "MESH_EXECUTION and wire" ... go"; "The revisions of
       14b9a757", R-1), reversing "an execution value in the window of lag 0" and `first`: the view at this node (`from`
       at the source, `to` at the destination) within its value's bytes (struct mesh_strided: "every extent within the
       value's bytes ... EINVAL otherwise"); a destination's value MESH_EXECUTION of `lag` 0 (struct mesh_declared:
       "EINVAL otherwise"); the source's one value of every transfer of the range, as before; any value a source (a
       MESH_INVARIANT one, or one of `lag` k >= 1, among them), placed in the window whatever its `wire` (plan_items).
       The implementer's decision, recorded: a value of `state` 1 a transfer names stays EINVAL, as the plan call's
       notes on `state` state it ("EINVAL but on a local MESH_INVARIANT value outside the window (wire 0, named by no
       declared transfer)"), which the 2026-11-06 notes do not reverse.  What must stay: MESH_ABSENT away from the
       range's ends */
    if(t->source!=p->self && t->destination!=p->self){
      if(t->value!=MESH_ABSENT){status=EINVAL;goto done;}
      continue;
    }
    if(t->value>=given){status=EINVAL;goto done;}
    const struct mesh_value *v=values+t->value;
    const struct mesh_strided *view=t->source==p->self?&t->from:&t->to;
    plan_view(view,&bytes,&end);
    if(end>v->section.bytes || v->state || (t->destination==p->self && (v->kind!=MESH_EXECUTION || v->lag)) ||
       (t->source==p->self && d->mine!=MESH_ABSENT && d->mine!=t->value)){
      status=EINVAL;goto done;
    }
    d->mine=t->value;
    d->view=view;
  }
  /* each value a range here names, its `named` (2026-11-06, R-1; "The revisions of 14b9a757": the covers' sort, its
     EINVAL for two ranges sharing a block, and struct mesh_share's tests go: mesh-collective.h, ""no two ranges at
     this node share a block of it" goes"; PB R2 at mesh 56ede97e, the share record "and its rules ... The record
     goes").  What must stay: every test before anything is written */
  for(uint32_t i=0;i<p->demands;i++)if(p->demand[i].mine!=MESH_ABSENT)p->named[p->demand[i].mine]=1;
  /* a local execution value's positions, a value of lag 0's alone (a value of lag k >= 1 is ordered by the program's
     events, not by its positions, which are not read: R4) */
  for(uint32_t j=0;j<given;j++)
    if(!p->named[j] && values[j].kind==MESH_EXECUTION && !values[j].lag && values[j].begin>values[j].end){
      status=EINVAL;goto done;
    }
  for(uint32_t i=0;i<p->demands;i++)if(p->demand[i].source==p->self)p->demand[i].release=(uint32_t)p->releases++;
  /* 2026-11-06 (R-1; "The revisions of 14b9a757", R-1): each range's instructions, the units of its routing below */
  if((status=plan_instructions(p,m)) || (status=plan_landings(p)))goto done;

  if(p->demands){
    for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++){
      const uint64_t c=nodes->capacity[a][b]<nodes->capacity[b][a]?nodes->capacity[a][b]:nodes->capacity[b][a];
      if(c){p->from[p->pairs]=a;p->to[p->pairs]=b;p->capacity[p->pairs]=(double)c;p->pairs++;}
    }
    const uint32_t E=p->pairs;
    /* a destination its source reaches by no Thunderbolt-5 path within the map: refused (mesh_graph_reach, once from
       each source; no breadth-first parents wanted here) */
    const struct plan_edges graph={p->from,p->to};
    uint32_t order[n];
    unsigned char reached[n];
    uint64_t reach[n],known=0;
    for(uint32_t i=0;i<p->demands;i++){
      const uint32_t s=p->demand[i].source;
      if(!(known>>s&1)){
        mesh_graph_reach(n,E,plan_end,&graph,s,order,reached,NULL);
        reach[s]=0;
        for(uint32_t v=0;v<n;v++)if(reached[v])reach[s]|=UINT64_C(1)<<v;
        known|=UINT64_C(1)<<s;
      }
      if(p->demand[i].leaves&~reach[s]){status=EHOSTUNREACH;goto done;}
    }
    const uint32_t rounds=(uint32_t)(mesh_graph_log(2*E>1?(double)(2*E):2.0)/(MESH_PLAN_EPS*MESH_PLAN_EPS))+1;
    const uint64_t most=(uint64_t)rounds*p->demands;
    const uint32_t room=most>=UINT32_MAX?UINT32_MAX-1:(uint32_t)most;
    root=malloc((size_t)p->demands*sizeof *root);
    up=calloc(p->demands,sizeof *up);
    down=malloc((size_t)p->demands*sizeof *down);
    demand=malloc((size_t)room*sizeof *demand);
    p->parent=malloc((size_t)room*n*sizeof *p->parent);
    share=malloc((size_t)room*sizeof *share);
    if(!root || !up || !down || !demand || !p->parent || !share){status=ENOMEM;goto done;}
    /* 2026-11-06 (R-1; "The revisions of 14b9a757"): each demand's bytes down its trees its B (mesh-graph.h
       mesh_graph_flow: "down[d] from it"), where it was its pages x Q.  The implementer's decision, recorded: a range
       has no pages of its own now, and B is the bytes that cross a tree.  What must stay: every rank's alike */
    for(uint32_t i=0;i<p->demands;i++){root[i]=p->demand[i].source;down[i]=(double)p->demand[i].bytes;}
    /* UINT32_MAX: its own allocations failed (its eps is positive, its roots members, every leaf reached and its room
       the most trees its rounds can find) */
    const uint32_t made=mesh_graph_flow(n,E,p->from,p->to,p->capacity,p->demands,root,up,down,plan_leaf,p->demand,
                                        MESH_PLAN_EPS,rounds,room,demand,p->parent,share);
    if(made==UINT32_MAX){status=ENOMEM;goto done;}
    p->made=made;
    p->tree=malloc(((size_t)made+1)*sizeof *p->tree);
    p->run=calloc((size_t)made+1,sizeof *p->run);
    rate=malloc(((size_t)made+1)*sizeof *rate);
    latency=calloc((size_t)made+1,sizeof *latency);
    trees=malloc(((size_t)p->demands+1)*sizeof *trees);
    if(!p->tree || !p->run || !rate || !latency || !trees){status=ENOMEM;goto done;}
    /* the solved trees grouped by demand, in the order the solve found them */
    mesh_graph_rows(p->demands,made,plan_at,demand,trees,p->tree);
    for(uint32_t i=0;i<p->demands;i++){
      p->demand[i].tree=(uint32_t)trees[i];
      p->demand[i].trees=(uint32_t)(trees[i+1]-trees[i]);
    }
    for(uint32_t j=0;j<made;j++)rate[j]=share[p->tree[j]];
    for(uint32_t i=0;i<p->demands;i++){
      struct plan_demand *d=p->demand+i;
      /* a tree's rate its share of the demand's flow; no tree is known to start later than another (latency 0).
         2026-11-06 (R-1; "The revisions of 14b9a757", R-1): the units its P instructions, where they were its pages.
         The implementer's decision, recorded (no document states another unit): mesh_graph_fill's earliest finish over
         equal units ("`units` equal units over `parts` parallel routes"), as it cut pages, each instruction one unit
         whatever its bytes, so every rank cuts them alike.  What must stay: every instruction on one tree */
      mesh_graph_fill(d->trees,latency+d->tree,rate+d->tree,d->instructions,p->run+d->tree);
      uint64_t cut=0;
      for(uint32_t j=d->tree;j<d->tree+d->trees;j++)cut+=p->run[j];
      if(cut!=d->instructions){status=EHOSTUNREACH;goto done;}
    }
  }
  /* the trees' communications, the template, no lag-0 cycle, the least lags and u; this node's entries of one copy
     counted; the plan's nodes; its actors' times: all before anything is written.  2026-11-06 (R-1; "The revisions of
     14b9a757"): and, after the least lags, the frames' items here (plan_sets) */
  if((status=plan_trees(p)) || (status=plan_template(p,dependence,dependences)) || (status=plan_acyclic(p)) ||
     (status=plan_unroll(p)) || (status=plan_sets(p,m)))goto done;
  plan_walk(p);
  uint64_t entries;
  /* 2026-11-04, owner C: the receives past a 32-bit count (struct mesh_state `receives`) are ENOMEM, as every other
     count past its word here, where they were ENOSPC with the list's room, which is gone (the entries are in the
     state section) */
  if(p->receives>=UINT32_MAX || __builtin_mul_overflow(p->entries,(uint64_t)p->copies,&entries) ||
     entries>=UINT64_MAX/(2*sizeof(struct mesh_hop))){
    status=ENOMEM;goto done;
  }
  /* the plan's nodes (a set of map indices) and its sender, the least of them (design/compilation-coordination.md,
     "The region", struct mesh_state `nodes` and `sender`; Q2), the same at every rank: every source and destination of
     a declared transfer and every node a tree the fill gave pages reaches (its `on`), this node's own where none.
     2026-11-05 ("The revisions of 3f106747", BS): computed here, before the rooms, where it followed the schedule.
     2026-11-05 ("The revisions of 4e78f6f4"): and its nodes in the program's IPC graph, `nodes`: these, or {s} for a
     plan with no declared transfer, at every rank (BS-5; plan_graph).  What must stay: as above */
  uint64_t spanned=0;
  for(uint32_t i=0;i<p->demands;i++)spanned|=UINT64_C(1)<<p->demand[i].source|p->demand[i].leaves|p->demand[i].on;
  if(!spanned)spanned=UINT64_C(1)<<p->self;
  if(spanned&~program){status=EINVAL;goto done;}
  p->spanned=spanned;
  p->nodes=p->demands?spanned:UINT64_C(1)<<__builtin_ctzll(program);
  status=plan_times(p);
done:
  free(key);free(placed);free(root);free(up);free(down);free(demand);free(share);free(rate);free(latency);free(trees);
  return status;
}

/* a plan's allocations */
static void plan_free(struct plan *p){
  free(p->named);free(p->uncovered);free(p->item);free(p->sync);free(p->cut);free(p->at);free(p->set_first);
  free(p->member);free(p->set_blocks);free(p->set_far);free(p->block_first);free(p->block_set);free(p->block_element);
  free(p->wait_row);free(p->demand);free(p->of);free(p->parent);
  free(p->tree);free(p->run);free(p->members);free(p->lead);free(p->comm);free(p->child);free(p->value);free(p->read);
  free(p->here);free(p->first);free(p->head);free(p->lag);free(p->between);free(p->farthest);free(p->entry);
}

/* The previous operand's release waited for (2026-11-05, design/compilation-coordination.md, "Review of 4e78f6f4
   against the requirement lines", B2, and "Answered in the documents", EB-1: "In RCU's form the writer waits for the
   grace period to end (`synchronize_rcu`), here the bridge's release of the old root.  So mesh_plan waits for it,
   bounded by MESH_PLATFORM_NS, and returns ETIMEDOUT past the bound"; "The revisions of 55b9cbad"), where mesh_plan
   returned EBUSY: while struct mesh_algo `published` (acquire) is not 0 and `released` (acquire) is not `published`,
   a wait on `released` (os_sync_wait_on_address_with_timeout, OS_SYNC_WAIT_ON_ADDRESS_SHARED, woken by the bridge
   parent's release, mesh-dataflow.h mesh_retired_release; as mesh-dataflow.c's start waits), its deadline
   MESH_PLATFORM_NS from the wait's start, ETIMEDOUT past it.  The SDK's os/os_sync_wait_on_address.h: `timeout_ns`
   "a timeout in nanoseconds", "Passing zero for the @timeout_ns results in an error" (so the wait is taken only
   before the deadline), and it returns where "the value at @addr is different than expected".  The record has one
   writer, the program's process (B2: "it needs no exchange and no lock: a single-writer register (L. Lamport, On
   interprocess communication, Distributed Computing 1(2), 1986)"), so no compare-and-swap; the program retires its own
   earlier operand first (mesh-dataflow.h mesh_retire).  A bounded spin: each turn ends at the deadline at the latest.
   0; ETIMEDOUT
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-7, R-8
   and R-10; "The revisions of 14b9a757", R-7, R-8 and R-10), reversing the wait's own deadline and its test:
   - the deadline is the caller's (mesh-collective.h mesh_plan `deadline`, absolute clock_gettime_nsec_np(
     CLOCK_MONOTONIC) nanoseconds), ETIMEDOUT at it, where it was MESH_PLATFORM_NS from the wait's start.  Why: R-7,
     "make()'s waits share one deadline"; mesh-collective.h, "The caller's one deadline covers its own earlier wait of
     the same preparation (rdma/mesh_compile.py's retirement) and this one";
   - the test is mesh.h mesh_algo_unreleased, "published and not yet released", where this wrote it again.  Why: R-10
     (W13: "written three times (mesh-collective.c 2091; mesh-dataflow.c 354; mesh-flow.c 2157), to be beside
     `mesh_algo_open`").  `released` is read before it, as the value the wait expects, so a release stored between the
     two makes the wait return at once (the SDK's "the value at @addr is different than expected"); `published` is this
     process's own word (R-8: "the program writes `retired`, the bridge writes only `released`"), and nothing here
     writes `retired` or `released`.
   What must stay: a bounded spin, each turn ending at the deadline at the latest; no compare-and-swap */
static int plan_released(struct hdr *m,uint64_t deadline){
  _Atomic uint64_t *released=&mesh_algo(m)->released;
  for(;;){
    const uint64_t seen=atomic_load_explicit(released,memory_order_acquire),now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(!mesh_algo_unreleased(m))return 0;
    if(now>=deadline)return ETIMEDOUT;
    os_sync_wait_on_address_with_timeout(released,seen,sizeof *released,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                         OS_CLOCK_MACH_ABSOLUTE_TIME,deadline-now);
  }
}

/* mesh-collective.h mesh_plan: the program's plan call (design/compilation-coordination.md, "The documents'
   corrections of 2026-11-05", points 4 to 8, and "The revisions of 4e78f6f4"; design/aot-compilation-change-
   request.md A8 and A9: "The compilation must carry one IR from partitioning to serialization, as an operand of its
   own"; point 5: "The plan call's form for a whole program is A8 and A9 ... With the program exported whole (point 1),
   there is one IR of the program and one serialized operand of it").  The IR is each compiled function's operands
   (struct mesh_function) and the program's calls of them in program order (`calls`, A10's graph); the serialized
   operand at this node is, for each function, its plan (A9: "for each adjacent node and direction, the ordered list of
   page instructions ... the page table's bindings and the plan's state section"), and its top level at the layout's
   one place (mesh.h struct mesh_algo, MSCCL-IR's `<algo>`), which the bridge reads "at a place its compilation fixed,
   and searches for nothing" (A9, "What the bridge runs: the self-timed schedule").  Each plan: its demands, routing,
   pages, template, u and sets (plan_function); its order, one execution's task graph scheduled (plan_schedule; design/plan-to-bridge-
   change-request.md, R3 to R5), and its instructions, the tasks and the communications, in one global topological
   order by their completion times in R5's linear program, the least first (MSCCLang §5.2, step 3; R5 at mesh
   0e56d8da), this node's entries in it, a set's at their own places (design/compilation-coordination.md, "The
   documents' revisions of edd4b583", GO1; "The revisions of 46ae8fdf", LP; "The revisions of 67ca86f8", SP: R5, "keep
   their places in that order").  Every rank computes the same demands, trees, pages, template, u, schedule, order and
   IPC graph from the same IR and map, and writes its own hops of them (plan_walk, plan_schedule).
   The plans' names (design/compilation-coordination.md, N1; design/aot-compilation-change-request.md, appendix item
   6): function k (from 0) is (group, k + 1), held in struct mesh_state `epoch`; its struct mesh_function `plan` its
   state section's region offset once the program's operand is published (N2), a constant of the plan.
   In this order:
   1. everything computed first: the calls' tests (EINVAL where one names no function or a function has none), each
      plan (plan_function, a malformed share's EINVAL among its tests), the program's IPC graph with every plan's
      synchronization edges, their rows and rooms (plan_graph), every value's rows and section (ENOMEM where they pass
      the page table's), each state section's layout (plan_layout), the program's one allocation (plan_place), each
      plan's entries filled (plan_walk) and in their places' order (plan_schedule, R5), every row's page and every
      value's `shares` (plan_rows);
   2. the previous operand's release waited for, at most MESH_PLATFORM_NS (plan_released): ETIMEDOUT past it, nothing
      written;
   3. each plan's state section started once, where it is placed (D3: its bytes before the entries 0, then struct
      mesh_state's words, `epoch` k + 1, `calls`, `waits`, `tells`, `syncs` and `rows` among them, `closed` 0), its
      synchronization edges (struct mesh_sync, `count` 0) and its waits' rows; every value's rows, each named clean and
      bound to its page (mesh_backing_bind, one row a block), every `state` 1 value's bytes 0 once (D3); each range's
      words at its own first row's publication (copy 0's row value.first + first: point 7); the entries u x n in
      plan_schedule's order, each with its `set` (S1, S2); the table's words, each plan's state section's region
      offset; no data page is written;
   4. the map read again: where its serial moved, or the reader failed, nothing is published: ESTALE, or the reader's
      error;
   5. struct mesh_algo `plans`, `group` and `table`, then the holder, hdr `client` and `client_start` from the
      context's `client` and `start`, then `published`, minted from hdr `serial`, stored last (release); each function's
      `plan`; hdr `plans` raised and woken (mesh.h mesh_notify).
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1, R-3,
   R-6, R-7, R-8 and R-10; design/aot-compilation-change-request.md A7, corrected 2026-11-06; "The revisions of
   14b9a757"), reversing the steps above where they differ:
   - step 1: each plan's views tested and flattened, its ranges' instructions (plan_instructions), routed and cut over
     its trees in instructions, and the frames' items here (plan_sets) among plan_function's work, struct mesh_share's
     EINVAL gone; the plans toward each map index (R-6) after the rows' pages;
   - step 2: the release waited for until the caller's `deadline` (R-7), its test mesh_algo_unreleased (R-10);
   - step 3: each declared transfer's range's words in the section's transfer table (struct mesh_state `transfers`),
     where they were a range's first row's publication, and no row's publication written; each entry its instruction's
     frame, `offset` and bytes (struct mesh_hop); the table's words then struct mesh_algo `adjacent`'s;
   - step 4: the serial the installed view's identifier (A7), ESTALE as written;
   - step 5: struct mesh_algo `nodes` and `adjacent` with `plans`, `group` and `table`, before `published`; nothing
     writes `retired` or `released` (R-8).
   What must stay: every failure before the writes but the second read's; `published` last.
   2026-11-05 (design/compilation-coordination.md, "Answered in the documents (2026-11-05, the coordinator, on this
   part's record)" and "Review of 4e78f6f4 against the requirement lines", G3, G4, B2, EB-1, K1, K4, V2 and G2; "The
   revisions of 55b9cbad"), reversing the steps above as 4e78f6f4 wrote them:
   - The rule is the publication's, in place of "every failure but ESTALE, the second map read's and the mapping's
     found before anything is written": "nothing is visible before the root's publication: one release store after
     every write, in C3's writing order, as RCU publishes a structure only once it is initialized (McKenney and
     Slingwine)" ("Answered in the documents").  Every write, the second map read, the record's words and the holder,
     `published` last; a failure before that store (ESTALE, the map reader's error) leaves nothing published, the pages
     written belonging to no published operand.  Every other failure is found before anything is written.
   - EBUSY where the root's `owner` was not 0 goes, for the bounded wait (B2, EB-1: "So mesh_plan waits for it,
     bounded by MESH_PLATFORM_NS, and returns ETIMEDOUT past the bound"), with no compare-and-swap (B2: "a
     single-writer register (L. Lamport, On interprocess communication, Distributed Computing 1(2), 1986)").
   - Steps 3 and 5's mapping go (mesh_view_compose, mesh_view_release, `*view`, struct mesh_value `view`), and the
     read of getpid() and mesh_started with them (G3: "The planner calls `mach_vm_remap` and writes virtual addresses
     into the IR, and reads `getpid()` ... Deciders see pages.  The plan writes rows, and the program's effector maps
     them"); the holder's words are the context's (mesh-dataflow.h struct mesh_ctx `start`, read by mesh_attach).
   - The record is struct mesh_algo at hdr `algo_off` (G4: "Its name is A9's ... Its fields are what A9 and the
     failport's lifecycle need: the operand's publication (B2's single writer) and the holder the heartbeat already
     knows"), where it was the root, struct mesh_program, whose `owner`, `closed`, `start` and `generation` this call
     wrote; mesh_plan writes nothing of `retired` or `released`, the program's retirement's and the bridge's.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", points 4 to 8; "The
   revisions of 4e78f6f4"), reversing the steps of the draft e6bd7db2:design/plan-bridge-coordination.md, §6, "Each plan
   in its claim" (one function a call; EEXIST where a plan of its name stood, every claim's head walked by
   mesh_head_after and mesh_head_state; its claim by mesh_claim_take, its rows after the claim's head and its pages two
   runs of the claim; its head's publication `state` stored last; mesh_claim_close on ESTALE): the call plans the
   whole program, every value placed by one allocation from row 0 and each space's first block, and stands once its
   root does.  Why: point 5: "place the whole program's values in one offline allocation at compile time.  These go:
   `mesh_claim_take`'s first fit and the claim's head record; `mesh_plan_head`'s walk of every claim head in
   `mesh_transfers_start` and `mesh_plan_retire`, a run-time check; `ENOMEM` from fragmentation.  Release at
   retirement stays deferred reclamation (RCU).  The program's group still names its plans"; "the serialized operand's
   place is a constant the region's layout names once".  What must stay: every failure but ESTALE, the second map
   read's and the mapping's found before anything is written; the map read again after everything is written and
   before the root stands; `plans` raised and woken once it stands.
   The notes before 4e78f6f4, each on the steps of one function's claim:
   2026-11-04, owner C2 (design/compilation-coordination.md, N1 to N5; design/aot-compilation-change-request.md,
   appendix item 6): reversing C-epoch's computation (the epoch one past the context's count, `mine`, the epoch kept
   from a cancelled standing plan, context->epoch raised once the plan stood and left at the named epoch less 1 on a
   failure after the writes), context->state, step 4's close of every earlier claim of the client, and step 6 (every
   earlier claim of this client closed once the plan stood).  Why: compiled functions coexist, each its own plan with
   its own claim, released only when the program retires that function or itself (R. Frostig, M. J. Johnson and C.
   Leary, SysML 2018; NVIDIA, CUDA Programming Guide §4.2.2; Pathways §3, §4.4); a plan is named by the program's
   ordinal of its function, the same at every rank; struct mesh_ctx holds no plan.  What must stay: every failure but
   ESTALE and the second map read's found before anything is written; the map read again after everything is written
   and before the plan stands; `plans` raised and woken once the plan stands.
   2026-11-04, owner C (design/compilation-coordination.md, "The region" and Q5): reversing the writing order of
   design/disjoint-pages-revision-coordination.md, "Writing a plan": the generation hdr `plan` + 1; hdr `hops` 0, a
   release fence and `plan` woken before the entries; the entries in hdr's hop area, ENOSPC past `hop_capacity`; hdr
   `state` and `plan` stored with them; `hops` stored, and `plan` woken, after the second map read.  Why: any number of
   plans stand at once, each in its own claim and found by its head (design/plan-to-bridge-change-request.md, R8;
   design/aot-compilation-change-request.md, A7), and hdr's one-plan fields are vacant.  What must stay: every failure
   but ESTALE and the second map read's found before anything is written; the map read again after everything is
   written and before the plan stands; the earlier claims closed only once the new plan stands (that last gone with
   step 6, 2026-11-04, owner C2: N3, N5, above).
   2026-11-04, sub-assignment 3: reversing 2026-11-03's (sub-assignments 3 and 5's) tail: `executions` as an input and
   its per-execution items; plan_section's check of sections mesh_section_create made (values now come in through
   `values`); the memset of every declared instance's pages before `hops`; the word-by-word zeroing of the supplied
   words and landed records; the call to mesh_plan_erased and its ETIMEDOUT; the layout's fixed areas of words.  Why:
   C4; a data page holds payload only and nothing is zeroed or reset while a plan stands (C3.7); a new plan's state is
   in new pages, so nothing is erased or waited for (C3.4).  What must stay: the demands, mesh_graph_reach,
   mesh_graph_flow and mesh_graph_fill as they were, so every rank's trees and template are the same; every failure but
   ESTALE and the second map read's found before anything is written.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS, PQ and RC): `program` and
   `outstanding` (mesh-collective.h): EINVAL where `program` is 0, lacks this node or lacks a node of the plan's (its
   `nodes`), ENOMEM where `outstanding` is 0 or a queue's room passes its words, before anything is written; struct
   mesh_state `sender` the least of `program`, where it was the least of the plan's `nodes`; the state section holds
   the plan's two queues, between the status and the entries (plan_layout), started 0 with every word before the
   entries (D3); every received range has its count words, a relay's too, so every receive entry names one.  Why:
   design/aot-compilation-change-request.md A7, "sender(m) is the program's one sender: the least of the program's
   nodes"; appendix item 2, "a queue of the plan's own.  It is laid out with the plan's pages when the plan is placed";
   item 5, "The code must name a page for round j by that count".  What must stay: every failure but ESTALE and the
   second map read's found before anything is written. */
int mesh_plan(struct mesh_ctx *context,uint64_t group,uint64_t program,struct mesh_function *functions,uint32_t count,
              const uint32_t *calls,uint32_t ncalls,uint64_t deadline){
  struct hdr *m=context->M;
  if(!group || !program || !count || !functions || !ncalls || !calls)return EINVAL;
  for(uint32_t j=0;j<ncalls;j++)if(calls[j]>=count)return EINVAL;
  struct plan *plans=calloc(count,sizeof *plans);
  size_t *row=malloc(((size_t)count+1)*sizeof *row),*order=malloc(((size_t)ncalls+1)*sizeof *order),*rows_by=NULL,
         *listed=NULL;
  struct plan_placement placed={0};
  struct mesh_nodes nodes,after;
  uint32_t *page=NULL,*adjacency=NULL,*toward=NULL,*place=NULL;
  int status=!plans || !row || !order?ENOMEM:0;
  if(status)goto done;
  /* the program's calls grouped by function, each function called at least once (mesh-collective.h mesh_plan) */
  mesh_graph_rows(count,ncalls,plan_at,calls,row,order);
  for(uint32_t f=0;f<count;f++)if(row[f]==row[f+1]){status=EINVAL;goto done;}
  if((status=mesh_nodes_read(context,&nodes)))goto done;
  if(nodes.self>=nodes.count){status=EHOSTUNREACH;goto done;}
  if(!(program>>nodes.self&1)){status=EINVAL;goto done;}
  const uint64_t block=plan_quantum(m);
  /* 1. each plan, the IPC graph, the rows, the layouts, the one allocation, the entries in their order.  2026-11-06
     (design/aot-compilation-change-request.md A7, corrected 2026-11-06: "The map a view is created from is the
     sequencer's, and the view's first message carries it ... Every member plans on those records"; mesh.h struct
     mesh_state `serial`, "the identifier of the installed view whose map the plan was made on (hdr `view`)"; "The
     revisions of 14b9a757", A7): each plan's `serial` is mesh_nodes_read's, the installed view's identifier, where it
     was the map serial, and the map is that view's records (mesh-dataflow.h mesh_nodes_read).  What must stay: every
     plan of the call made on one read */
  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    p->context=context;p->n=nodes.count;p->self=nodes.self;p->serial=nodes.serial;
    if((status=plan_function(p,functions+f,&nodes,program)))goto done;
  }
  if((status=plan_graph(plans,count,calls,ncalls,program,nodes.self,row,order)))goto done;
  /* every value's rows: count x P, count its copies (plan_copies), the program's from row 0, function by function and
     value by value, at most the page table's (2026-11-05, point 5: no claim's head row, no first fit over the rows),
     and its section (mesh-section.h: `first`, `pages` P x block, `bytes` as given, `count`, `stride` P), where the
     writes' step set it (2026-11-05, design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement
     lines", G2 and G3; "The revisions of 55b9cbad": decided before the rows' pages, plan_rows, which reads it); no
     `run` value is mapped here (G3) */
  uint64_t rows=0;
  for(uint32_t f=0;f<count;f++)for(uint32_t j=0;j<plans[f].givens;j++){
    struct mesh_value *v=plans[f].given+j;
    const uint32_t P=(uint32_t)plan_blocks(m,v->section.bytes),copies=plan_copies(plans+f,j);
    uint64_t blocks,end;
    if(__builtin_mul_overflow((uint64_t)copies,(uint64_t)P,&blocks) || __builtin_add_overflow(rows,blocks,&end) ||
       end>mesh_rows(m)){status=ENOMEM;goto done;}
    v->section=(struct mesh_section){(uint32_t)rows,P*m->block,v->section.bytes,copies,P};
    rows=end;
  }
  for(uint32_t f=0;f<count;f++)plan_layout(plans+f);
  if((status=plan_place(plans,count,m,&placed)))goto done;
  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    if(!(p->entry=malloc((p->entries+1)*sizeof *p->entry))){status=ENOMEM;goto done;}
    plan_walk(p);
    if(p->demands && (status=plan_schedule(p,p->weights,p->dependence,p->dependences)))goto done;
  }
  if(!(page=malloc(((size_t)rows+1)*sizeof *page))){status=ENOMEM;goto done;}
  if((status=plan_rows(plans,count,m,&placed,page,rows)))goto done;
  /* the plans toward each map index (2026-11-06, design/compilation-coordination.md, "The root cause of the copies, and
     the revised work list", R-6: "Each cable's processor executes its list of instructions in the compile-time order
     ... It scans no table of plans"; mesh.h struct mesh_algo `adjacent`; "The revisions of 14b9a757", R-6): n + 1
     offsets, then for each map index i the places f of the plans whose entries name node i (plan_walk's `toward`),
     ascending, as mesh-graph.h mesh_graph_rows' compressed rows keyed by map index over the records (i, f) taken plan
     by plan, so each row's places ascend; placed in the table's item (plan_items, plan_table) and written before
     `published`.  What must stay: every place below `plans` */
  const uint32_t n=nodes.count;
  uint64_t places=0;
  for(uint32_t f=0;f<count;f++)places+=(uint64_t)__builtin_popcountll(plans[f].toward);
  adjacency=malloc(((size_t)n+1+places)*sizeof *adjacency);
  toward=malloc((places+1)*sizeof *toward);
  place=malloc((places+1)*sizeof *place);
  rows_by=malloc(((size_t)n+1)*sizeof *rows_by);
  listed=malloc((places+1)*sizeof *listed);
  if(!adjacency || !toward || !place || !rows_by || !listed){status=ENOMEM;goto done;}
  for(uint32_t f=0,q=0;f<count;f++)
    for(uint64_t s=plans[f].toward;s;s&=s-1){toward[q]=(uint32_t)__builtin_ctzll(s);place[q++]=f;}
  mesh_graph_rows(n,places,plan_at,toward,rows_by,listed);
  for(uint32_t i=0;i<=n;i++)adjacency[i]=(uint32_t)rows_by[i];
  for(uint64_t q=0;q<places;q++)adjacency[n+1+q]=place[listed[q]];
  /* 2. the previous operand's release waited for until the caller's deadline (B2, EB-1, R-7; plan_released), before
     anything is written */
  if((status=plan_released(m,deadline)))goto done;

  /* 3. each plan's state section, its synchronization edges and their rows, its values' rows, its ranges'
     publications and its entries; then the table */
  for(uint32_t f=0;f<count;f++){
    struct plan *p=plans+f;
    const uint32_t u=p->copies,at=plan_page(&placed,m->block,p->state,0);
    const uint64_t base=m->data_off+(uint64_t)at*m->pgsz;
    p->base=base;
    /* the state section started once, where it is placed (D3): its bytes before the entries 0, then struct
       mesh_state's fields; nothing else of it is written by the plan but the entries, and no word of it is reset while
       the plan stands */
    memset(mesh_at(m,at),0,(size_t)p->hops);
    struct mesh_state *state=(struct mesh_state *)mesh_at(m,at);
    state->copies=u;
    state->receives=(uint32_t)p->receives;
    state->ranges=p->consumed;
    state->releases=(uint32_t)p->releases;
    state->phases=base+p->phases;
    state->counts=base+p->counts;
    state->supplied=base+p->supplied;
    state->transfers=base+p->publications;
    state->status=base+p->status;
    state->group=group;
    state->epoch=(uint64_t)f+1;
    state->serial=p->serial;
    state->hops=base+p->hops;
    state->entries=p->entries*u;
    state->nodes=p->spanned;
    state->sender=(uint32_t)__builtin_ctzll(program);
    state->outstanding=(uint32_t)p->outstanding;
    state->room=p->room;
    state->submit_room=p->submit_room;
    state->deliveries=base+p->deliveries;
    state->submits=base+p->submits;
    /* 2026-11-05 (point 4; "The revisions of 4e78f6f4"): the plan's synchronization edges here, `waits` then `tells`
       (struct mesh_sync, `count` 0 from the bytes started above), its calls, `closed` 0 (the bytes above), where it
       held one edge's `waits`, `tells` and `wait_delay`.  What must stay: every word started once */
    state->waits=p->waits;
    state->tells=p->tells;
    state->syncs=base+p->syncs;
    state->calls=p->calls;
    /* 2026-11-05 (118; K1; "The revisions of 55b9cbad"): each entry's `node`, `function`, `entry`, `every` and
       `base` (plan_graph), where it wrote `position`, `at` and `delay`; the waits' rows, `calls` + 1 words at `rows`
       (mesh.h struct mesh_state `rows`).  What must stay: every word started once */
    state->rows=base+p->rows;
    struct mesh_sync *sync=mesh_syncs(m,state);
    for(uint32_t k=0;k<p->waits+p->tells;k++){
      sync[k].node=p->sync[k].node;sync[k].function=p->sync[k].function;sync[k].entry=p->sync[k].entry;
      sync[k].every=p->sync[k].every;sync[k].base=p->sync[k].base;
    }
    memcpy(mesh_at(m,at)+p->rows,p->wait_row,((size_t)p->calls+1)*sizeof *p->wait_row);
    /* every value's rows, from the program's next, copy c's rows first + c x P + k each bound to its block's page
       (point 6: "bind every value's rows (mesh_backing_bind, one row a block)"), named clean as mesh_section_create
       named them (no device, no constant, no publication: every field of a row's publication 0 but its ranges' words
       below); a named value's block k of copy c, where a range lies, the range's routed copy c's block k - first (its
       demand's routed value here, a relay's among them, no row's), elsewhere its uncovered item's next block; a local
       value's copy c its item[j] + c; a `state` 1 value's bytes 0 once, block by block (D3: a local invariant outside
       the window alone).
       2026-11-05 (points 5 to 7; "The revisions of 4e78f6f4"): the rows the program's allocation numbers, one row a
       block placed anywhere, where they were the claim's after its head, bound copy by copy to one run of pages, and
       struct mesh_buffer's `owner`, `closed` and `pages` named 0 with `constant` (117: read and written no more).
       What must stay: every row named clean; no data page written but a `state` value's, once.
       2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G2, G3 and
       V2; "The revisions of 55b9cbad"): each row bound to its page as plan_rows decided it, a share's rows to its
       `of`'s pages among them, where this walked each value's ranges again; no row is gathered for a mapping and no
       `view` written (G3).  The implementer's decision, recorded (the header is silent): a `state` 1 value's bytes 0
       once over every row of it, a share's rows too, as D3 has it ("every `state` 1 value's bytes 0 once"): that write
       is the plan's, before the first execution, so it precedes `of`'s producer's first write (an invariant's "once,
       after the plan and before the first execution", design/disjoint-pages-change-request.md D.3.2).  What must
       stay: as above
       2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
       "The revisions of 14b9a757", R-1): no row's publication is written, the zeroing of every row's among them, nor
       a range's words at a row (mesh.h struct mesh_state `transfers`: "Every row's publication (mesh_publication_at) is
       written by nothing now and read by nothing"); the shares' rows go with struct mesh_share.  What must stay: every
       row named clean of its device and constant and bound to its page; a `state` 1 value's bytes 0 once */
    for(uint32_t j=0;j<p->givens;j++){
      const struct mesh_value *v=p->given+j;
      const struct mesh_section x=v->section;
      for(uint32_t c=0;c<x.count;c++)for(uint32_t k=0;k<x.stride;k++){
        const uint32_t r=x.first+c*x.stride+k;
        atomic_store_explicit(&mesh_page(m)[r].device,0,memory_order_relaxed);
        mesh_buffers(m)[r].constant=0;
        mesh_backing_bind(context,r,m->block,page[r],c);
        if(v->state){
          const uint64_t left=v->section.bytes-(uint64_t)k*block;
          memset(mesh_at(m,page[r]),0,(size_t)(left<block?left:block));
        }
      }
    }
    /* each declared transfer's range's words here (2026-11-06, R-1; mesh.h struct mesh_state `transfers`; "The
       revisions of 14b9a757", R-1), entry i of the section's `publications` (64 bytes each, 0 from the bytes started
       above): where this node is declared[i]'s source, `sends` 1 and `targets[0]` {its range's supplied word, 1, 0};
       where it is declared[i]'s destination, `pages` P (its range's instructions it receives a copy), `phases` its
       copy-0 group and `count` its copy-0 count word; `state` the section where either holds; every other field 0.
       Why: mesh.h, "each written once by mesh_plan, read by index" (mesh_transfer_at), where a range's first row's
       publication held them; ranges may share a block, so a row names no one range.  What must stay: written before
       `published` */
    for(uint32_t i=0;i<p->transfers;i++){
      const struct mesh_declared *t=p->declared+i;
      const struct plan_demand *d=p->demand+p->of[i];
      struct mesh_publication *entry=(struct mesh_publication *)(void *)(mesh_at(m,at)+p->publications+
                                                                         (size_t)i*sizeof(struct mesh_publication));
      if(t->source==p->self){
        entry->sends=1;
        entry->targets[0]=(struct mesh_target){.word=plan_supplied(p,d),.count=1,.stride=0};
        entry->state=base;
      }else if(t->destination==p->self){
        entry->pages=d->received;
        entry->phases=plan_group(p,d,0);
        entry->count=plan_count(p,d,0);
        entry->state=base;
      }
    }
    /* the entries, copy c's n at [c n, c n + n) of the section's `hops`, in plan_schedule's order (R5), each naming
       copy c's page and words (mesh.h struct mesh_hop): a receive its field's word and low bit in its range's copy-c
       group, its range's copy-c count word and P_g; a forwarding send its receive's word and field; a send from the
       range's source its release's supplied word; each its page's bytes, page k of a range of B bytes min(Q, B - k Q),
       Q = block x pgsz (design/compilation-coordination.md, T7); each its entry's `set`, every copy's the same (S1,
       S2); its page the routed copy's block k's (plan_page, point 6)
       2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1;
       mesh.h struct mesh_hop `offset`; "The revisions of 14b9a757", R-1), reversing each hop's page k and its bytes
       min(Q, B - k Q): each hop is its instruction k (plan_instructions), its `bytes` the instruction's, cut[k + 1] -
       cut[k], its `page` the frame of the block holding the instruction at this node and its `offset` the byte within
       that block.  At a relay that is byte cut[k] of its routed value, laid out in the range's order, block cut[k] / Q
       of copy c of its set (plan_sets); at the source or a destination, byte at[k] of its value, the row of block
       at[k] / Q of its copy c mod its copies (a MESH_INVARIANT source's copy 0 in every copy, mesh-collective.h: "every
       copy's send reads its one copy"), the frame plan_rows bound to it.  Why: R-1, "The bridge's instructions are
       those blocks (aot A9: MSCCL-IR's steps with offset and count)"; PB R2, corrected 2026-11-06, "A page instruction
       names its bytes at their offset in its page".  What must stay: the words, fields, counts and sets as above */
    struct mesh_hop *hop=mesh_hops(m,state);
    for(uint32_t c=0;c<u;c++)for(uint64_t j=0;j<p->entries;j++){
      const struct plan_entry *e=p->entry+j;
      const struct plan_demand *d=p->demand+e->demand;
      const uint64_t *cut=p->cut+d->cut+e->instruction;
      uint64_t byte=cut[0];
      uint32_t frame;
      if(d->mine==MESH_ABSENT)frame=plan_page(&placed,m->block,p->set_item+d->relay*u+c,byte/block);
      else{
        const struct mesh_section x=p->given[d->mine].section;
        byte=p->at[d->at+e->instruction];
        frame=page[x.first+(size_t)(c%x.count)*x.stride+byte/block];
      }
      struct mesh_hop h={.page=frame,.node=e->node,.direction=e->direction,.field=MESH_ABSENT,
                         .bytes=(uint32_t)(cut[1]-cut[0]),.serial=p->serial,.set=e->set,
                         .offset=(uint32_t)(byte%block)};
      if(e->field==MESH_ABSENT)h.word=plan_supplied(p,d);
      else{
        h.word=plan_group(p,d,c)+(uint64_t)e->field/32*sizeof(uint64_t);
        h.field=2*(e->field%32);
        if(e->direction==MESH_RECEIVE){
          h.count=plan_count(p,d,c);
          h.pages=d->received;
        }
      }
      *hop++=h;
    }
  }
  /* the program's table (mesh.h struct mesh_algo `table`): word k function k + 1's state section's region offset.
     2026-11-06 (R-6; "The revisions of 14b9a757", R-6): then struct mesh_algo `adjacent`'s words (above).  What must
     stay: word k at k */
  const uint32_t table=plan_page(&placed,m->block,placed.table,0);
  uint64_t *word=(uint64_t *)(void *)mesh_at(m,table);
  for(uint32_t f=0;f<count;f++)word[f]=plans[f].base;
  memcpy(word+count,adjacency,((size_t)n+1+places)*sizeof *adjacency);
  /* 4. the map read again, after every write: where it moved, or the reader failed, nothing is published (a bridge
     refuses a plan whose `serial` is not `map_serial` as it loads it, and cancels one that stands where the serial
     moves: mesh.h, "Cancellation").  2026-11-06 (A7, corrected 2026-11-06; "The revisions of 14b9a757", A7): the
     serial the installed view's identifier (mesh-dataflow.h mesh_nodes_read), so ESTALE where the installed view moved
     while planning, as written; a bridge cancels a plan whose `serial` is not hdr `view` (mesh-bridge.h plan_stale).
     What must stay: the second read after every write, ESTALE as written */
  if(!(status=mesh_nodes_read(context,&after)) && after.serial!=nodes.serial)status=ESTALE;
  if(status)goto done;
  /* 5. the record's words and the holder, then `published` last (release), so the bridge reads every plan of it
     written (mesh.h struct mesh_algo, mesh_algo_plan; the publication rule, "one release store after every write";
     G4); each function's `plan`; `plans` raised and woken.  2026-11-06 (R-3, R-6 and R-8; "The revisions of
     14b9a757"): `nodes`, the map's n, and `adjacent`, after the table's `count` words, beside `plans`, `group` and
     `table`, all before `published` (mesh.h: "`plans`, `group`, `table`, `nodes` and `adjacent` written before
     `published`"), which a reader acquires before it reads any of them (R-3); nothing here writes `retired` or
     `released` (R-8).  What must stay: `published` stored last */
  struct mesh_algo *algo=mesh_algo(m);
  algo->plans=count;
  algo->nodes=n;
  algo->group=group;
  algo->table=m->data_off+(uint64_t)table*m->pgsz;
  algo->adjacent=algo->table+(uint64_t)count*sizeof(uint64_t);
  atomic_store_explicit(&m->client,context->client,memory_order_relaxed);
  atomic_store_explicit(&m->client_start,context->start,memory_order_relaxed);
  atomic_store_explicit(&algo->published,atomic_fetch_add(&m->serial,1)+1,memory_order_release);
  for(uint32_t f=0;f<count;f++)functions[f].plan=plans[f].base;
  mesh_notify(&m->plans);
done:
  for(uint32_t f=0;plans && f<count;f++)plan_free(plans+f);
  free(plans);free(row);free(order);free(placed.item);free(placed.count);free(placed.chain);free(page);
  free(adjacency);free(toward);free(place);free(rows_by);free(listed);
  return status;
}
