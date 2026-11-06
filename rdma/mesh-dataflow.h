#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
/* mesh_started, mesh_holds, mesh_gone (sub-assignment 4's header; tag pre-return-20261103 rdma/mesh-dataflow.h) */
#include "mesh-holders.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_range { uint32_t first,count; };
/* The program's context: the region mapped (`M`, `len`, `fd`) and the client word it minted (`client`, each of its
   claims' head owner).  It holds no rows and no pages: a program's rows and pages are its plans' claims (struct
   mesh_claim, mesh_claim_take), each taken by mesh_plan for exactly what that plan places
   (design/disjoint-pages-change-request.md, Appendix C, C3.4; design/disjoint-pages-revision-coordination.md, "The
   program's claim and its sections").  It holds no plan: a plan's handle is its state section's region offset,
   mesh_plan's `*plan`, which mesh_transfers_start, mesh-metal.h and mesh_plan_retire take
   (design/compilation-coordination.md, "The appendix's revisions", N2, N5).
   2026-11-04, owner L (design/plan-to-bridge-change-request.md R8): `padding`, `epoch` and `state` appended, where the
   region's one plan was hdr `state` and the client word hdr `client`.  What must stay: no rows or pages in the
   context.
   2026-11-04, owner L2 (design/aot-compilation-change-request.md, appendix item 6; N5): `epoch` and `state` gone
   (32 bytes).  Why: compiled functions coexist, each its own plan (item 6), so the context holds no standing plan and
   no count of plans; a plan is named by its function (mesh-collective.h mesh_plan, N1).  What must stay: no rows,
   pages or plans in the context.
   2026-11-04, sub-assignment 5: reversing sub-assignment 7's 2026-11-03 form, where the context held the program's one
   claim, taken at attach (`rows`, `window`, `local`: every free row and the widest free run of pages in the window and
   outside it) and laid out by cursors (`row_cursor`, `local_cursor`).  Why: C3.4, "Nothing is claimed at attach", and
   C4 (mesh_claim's claim at attach of everything free; the claim's local run, where values were laid out when
   created).  What must stay: no call site but mesh_plan's claim takes rows or pages from the region.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS): `clock` appended (40 bytes), the
   program's logical clock, which each of its starts that raises a plan's `executions` increments and stamps on the
   submission (mesh-call.h mesh_transfers_start): L. Lamport, Time, clocks, and the ordering of events in a distributed
   system, CACM 21(7), 1978, IR1, "Each process Pi increments Ci between any two successive events"; 0 at attach.  Why:
   design/aot-compilation-change-request.md A7: sender(m) "TO-broadcasts the program's submissions in the order the
   program makes them", and the order among one program's plans reaches its sender's bridge only so (mesh-records.c
   orders_submit).  What must stay: no rows, pages or plans in the context.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 4e78f6f4"): a program's rows and pages are its
   one allocation's (mesh-collective.h mesh_plan), and its plans are named by its serialized operand's root at the
   layout's place (mesh.h struct mesh_program), where the notes above name its plans' claims.  What must stay: no rows,
   pages or plans in the context.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3 and G4; "The
   revisions of 55b9cbad"): `start` appended (48 bytes), this process's start (mesh-holders.h mesh_started(getpid())),
   read by mesh_attach beside the client word it mints; with `client` the holder of the program's operand, which
   mesh_plan copies into hdr `client` and `client_start` as it publishes (mesh.h struct mesh_algo's note).  Why: G3,
   "The planner ... reads `getpid()` ... Deciders see pages", so the process's identity is read where the region is
   mapped, an effector's work, and the planner copies words; G4, "the holder the heartbeat already knows".  What must
   stay: no rows, pages or plans in the context */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; int fd; uint32_t padding; uint64_t clock,start; };
_Static_assert(sizeof(struct mesh_ctx)==48 && offsetof(struct mesh_ctx,fd)==24 && offsetof(struct mesh_ctx,padding)==28 &&
  offsetof(struct mesh_ctx,clock)==32 && offsetof(struct mesh_ctx,start)==40,"mesh_ctx: rdma/mesh.py Context");
/* design/prepared-machine.md#M09 */
/* The two arena ranges.  [0,wire_pages) is the registered window and the only memory an SGE may
   name; [wire_pages,arena) is addressable and never registered.  An undeclared window is the whole
   arena, and then there is one range over it, exactly as before the split. */
static inline struct mesh_range mesh_arena_range(const struct hdr *m,int wire){
  uint32_t arena=mesh_arena_pages(m),window=m->wire_pages>arena?arena:m->wire_pages;
  uint32_t begin=wire||window==arena?0:window,end=wire?window:arena;
  return (struct mesh_range){begin,end-begin};
}
static inline uint32_t mesh_block_pages(const struct mesh_ctx *c){ return c->M->block; }
static inline void *mesh_page_address(struct mesh_ctx *c,uint32_t page){ return mesh_at(c->M,page); }
/* the region attached (design/plan-to-bridge-change-request.md R8; design/compilation-coordination.md, "The region"):
   the region mapped and its magic, version and length checked; the client word minted, ((hdr `serial` + 1) &
   0x7fffffff) << 32 | pid, bit 63 set where hdr `device_client`'s is clear, as before; the version read again.  It
   holds nothing and writes nothing in the region but hdr `serial`: no client word, no row, no page.  0, EINVAL where
   the region is not of this layout or expired as it was attached (it is let go), else the errno.
   2026-11-04, owner L (R8): reversing the 109 form, the compare-and-swap of hdr `client` (EADDRINUSE where another live
   client held it, an earlier gone client retired first) and hdr `client_start` written.  Why: "nothing happens when a
   program joins or leaves because a program is a sequence of like, function calls"; a program holds its plans'
   claims and nothing else.  What must stay: the version read again after the client word is minted.
   2026-11-04, owner L2 (design/compilation-coordination.md, "The appendix's revisions", N5): no `epoch` or `state`
   set, the context holding neither.  What must stay: nothing held at attach.
   2026-11-05 ("The revisions of 55b9cbad", G3 and G4): the context's `start` set to mesh_started(getpid()) beside the
   client word, nothing written in the region for it.  What must stay: nothing held at attach */
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* design/algorithm-sources.md#programtensor */
/* the client retired (design/plan-to-bridge-change-request.md R8; e6bd7db2's section 6, as design/compilation-
   coordination.md, Q5 takes it): every claim whose head's `owner` is `client` closed (struct mesh_buffer `closed`,
   release), then hdr `plans` raised and woken; nothing is let go here, the bridge releasing each closed claim once no
   port holds it (mesh_retired_release).
   2026-11-04, owner L: reversing the 109 form, the retirer's compare-and-swap of hdr `client`, hdr `hops` and `state`
   stored 0 before the claims closed, and `client` cleared after.  Why: those words are vacant (mesh.h 110); a plan's
   unpublishing is its claim's close, which the ports read through `plans`.  What must stay: nothing let go here.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): the program's root (mesh.h struct mesh_program) closed (`closed` 1, release) where its
   `owner` is `client`, then hdr `plans` raised and woken; nothing is let go here, the bridge releasing the root once
   no port holds a plan of it (mesh_retired_release).  Why: point 5, "These go: `mesh_claim_take`'s first fit and the
   claim's head record ... Release at retirement stays deferred reclamation"; one program's operand at the layout's
   place.  What must stay: nothing let go here.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4; "The
   revisions of 55b9cbad"): the program's operand (mesh.h struct mesh_algo) retired where hdr `client` (its holder,
   acquire) is `client` and its `published` (acquire) is not 0: `retired` = `published` (release), then hdr `plans`
   raised and woken.  Its callers are the program's process (its detach, or its own retirement before it plans again)
   and the bridge parent for a holder that is gone (mesh-records.c clients_turn, mesh-link.c region_adopt: the
   failport's lifecycle, as the tag's mesh_retire retired a gone client); both store the one value `published`, so
   `retired` only names the operand published.  What must stay: nothing let go here */
void mesh_retire(struct hdr *,uint64_t client);
/* the program's retirement of one compiled function (design/aot-compilation-change-request.md, appendix item 6: a
   function's claim released only when the program retires that function or itself; design/compilation-
   coordination.md, "The appendix's revisions", N3, as NVIDIA, CUDA Programming Guide §4.2.3.1, cudaGraphExecDestroy(),
   destroys one instantiated graph).
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"), reversing the claim's close and its walk of the claims' heads (mesh_plan_head, ENOENT):
   the plan's state section at `plan` (mesh_plan's, a constant of the plan) `closed` 1 (release), then hdr `plans`
   raised and woken (mesh.h mesh_notify), so the ports let it go; its pages stay the program's, as every page of the
   program is placed once at compile time and nothing takes pages at run time, until the program's root is released
   (mesh_retired_release).  Why: point 5: "`mesh_plan_head`'s walk of every claim head in `mesh_transfers_start` and
   `mesh_plan_retire`, a run-time check", goes; appendix item 6 as corrected: "A function's pages are released when
   the program retires it or itself, by deferred reclamation".  What must stay: nothing let go here */
void mesh_plan_retire(struct mesh_ctx *,uint64_t plan);
/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): mesh_head_after, mesh_head_state, struct mesh_claim, mesh_claim_take and mesh_claim_close
   are removed: the claims' heads, their walk, the first fit over the rows' and pages' bitmaps (Knuth vol. 1 §2.5) and
   ENOMEM from fragmentation.  Why: point 5, "place the whole program's values in one offline allocation at compile
   time.  These go: `mesh_claim_take`'s first fit and the claim's head record; `mesh_plan_head`'s walk of every claim
   head ...; `ENOMEM` from fragmentation"; the bridge reads the program's operand at the layout's one place (mesh.h
   struct mesh_program, mesh_program_plan).  What must stay: no call site but mesh_plan names rows or pages */
/* rows [first, first + pages / block) of a claim bound to arena pages [page, page + pages), block by block, with
   `index` the copy (an entry's `mapping` high half): mesh_plan's binding of the rows and pages it placed; nothing is
   taken here, only named.
   2026-11-05 ("The revisions of 4e78f6f4"): the rows the program's allocation numbered, where they were a claim's;
   mesh_plan binds each row alone (`pages` one block), its block placed anywhere (point 6).  What must stay: nothing
   taken here, only named */
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
/* The mapping of a value typed to be seen as one run (2026-11-05; design/compilation-coordination.md, "The documents'
   corrections of 2026-11-05", point 6: "Give an operand typed as one run its run as a mapping written when the plan is
   made ... consecutive virtual pages over its pages, as `ncclMeshPageTableCompose`'s `mach_vm_remap` view was on
   2026-11-02 (PB R2)"; design/plan-to-bridge-change-request.md R2, corrected; restored from mesh 1b372d09^
   rdma/nccl-mesh.c ncclMeshPageTableCompose, its reservation and its remap a run, as it was): an effector, given rows
   and deciding nothing (the operator, 2026-11-02, design/page-maps-2026-11-02.md: "effectors know about pointers and
   pages but never make any decisions ever").  Out: `*view`, a reservation of `count` blocks (`address` its first byte,
   `bytes` count x block x pgsz: mmap of PROT_NONE anonymous memory, ncclMeshPageTableCompose's), then rows[k]'s
   block of pages remapped at address + k x block x pgsz for k < `count` (mach_vm_remap of this task's mapping of the
   region, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, copy FALSE, VM_INHERIT_NONE, as ncclMeshPageTableCompose's), each row's
   block read from its page-table binding (struct mesh_page_entry `address`); {NULL, 0} where `count` is 0.  0; ENOMEM
   where Mach gives KERN_NO_SPACE or KERN_RESOURCE_SHORTAGE; EINVAL for any other refusal (ncclMeshPageTableCompose's
   mapping of its codes); the reservation's errno.  Nothing stays mapped on a failure, and `*view` is {NULL, 0}.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3: "The
   planner calls `mach_vm_remap` and writes virtual addresses into the IR ... Deciders see pages.  The plan writes
   rows, and the program's effector maps them"; "The revisions of 55b9cbad"): called by the program's effector
   (tools/e2b_replay.m, rdma/mesh_compile.py through rdma/mesh.py) over its `run` values' rows once mesh_plan has
   written them, where mesh_plan called it; mesh_plan neither maps nor names an address.  What must stay: an effector,
   given rows and deciding nothing */
struct mesh_view { void *address; size_t bytes; };
int mesh_view_compose(struct mesh_ctx *,const uint32_t *rows,size_t count,struct mesh_view *view);
/* the program's mapping released (munmap of `address`, `bytes`), once its holder retires the program and no kernel
   reads it; nothing where `address` is NULL */
void mesh_view_release(struct mesh_view);
/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *,uint32_t row,uint64_t address);
/* 2026-11-04, sub-assignment 5: mesh_publication_prepare is removed (the GPU's publication records went with layout
   109, mesh.h; its one caller, mesh-metal.m, reads a range's first row's publication itself).  What must stay: a
   range's release is its supplied word, which its first row's publication names */
/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5: "Release at
   retirement stays deferred reclamation (RCU)"; "The revisions of 4e78f6f4"), reversing the claims of the note below:
   the program's root (mesh.h struct mesh_program) released where it is closed: a seq_cst fence, then kept where any
   port's `prepared` is its `generation` or any of its plans' struct mesh_state `held` is not 0, else its `owner` 0
   (release), so a next program may plan; 1 kept, else 0.  The bridge parent its one caller and its root's one
   releaser.  What must stay: the fence after the close is read and before the ports are; nobody waits for it.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", B2 and G4, and
   "Answered in the documents", EB-1; "The revisions of 55b9cbad"), reversing "its `owner` 0" and "nobody waits for
   it": where struct mesh_algo `published` (acquire) is not 0, `retired` (acquire) is `published` and `released` is
   not, a seq_cst fence, then kept where any port's `prepared` is `published` or any of its plans' `held` is not 0,
   else `released` = `published` (release) and woken (os_sync_wake_by_address_all, region-shared); 1 kept, else 0.
   The bridge parent its one caller and `released`'s one writer.  mesh_plan waits for that store before it writes a
   next operand (mesh-collective.h), as RCU's writer waits for the grace period to end (`synchronize_rcu`; EB-1: "In
   RCU's form the writer waits for the grace period to end (`synchronize_rcu`), here the bridge's release of the old
   root").  What must stay: the fence after the retirement is read and before the ports are.
   The note before 4e78f6f4: */
/* the bridge's release of closed claims (deferred reclamation: RCU's call_rcu, P. E. McKenney and J. D. Slingwine,
   Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004; the contract's D10; the hazard pointer's announcement
   read before reclamation, M. M. Michael, IEEE TPDS 15(6), 2004): for each closed claim, a seq_cst fence, then the claim
   kept where any port's `prepared` is its generation or its plan's struct mesh_state `held` is not 0, and released
   otherwise, its pages' bits, then its rows' bits, by the winner of the head's owner exchange; the closed claims kept.
   The bridge parent's (mesh-records.c clients_turn, mesh-link.c region_adopt); nobody waits for it
   (design/compilation-coordination.md, "The region") */
uint32_t mesh_retired_release(struct hdr *);

/* The map reader (design/page-queue-change-request.md, "What the collective decides, and what the bridge decides"): the
   collective sees node-level adjacency and each node pair's capacity, nothing else of how the bridge carries pages.
   `serial` is the map's (hdr `map_serial`, the one a plan is made against: struct mesh_hop `serial`); `count` the map's
   nodes, at most MESH_NODES_MOST; `node[i]` the node id of index i, a node's index its place in ascending node-id order
   among the map's records; `self` this node's index; `capacity[i][j]` == `capacity[j][i]` the bits a second of the
   cables between i and j, summed, each a cable both records list (mesh_cables) at the lesser of its two bandwidths;
   0 where none joins them. */
#define MESH_NODES_MOST 32
struct mesh_nodes { uint64_t serial; uint32_t count,self; uint32_t node[MESH_NODES_MOST];
                    uint64_t capacity[MESH_NODES_MOST][MESH_NODES_MOST]; /* bits/s, a pair's cables summed; 0 none */ };
/* 0, or an error value: ETIMEDOUT where no whole copy was read within MESH_PLATFORM_NS of its start, ESRCH where the
   bridge left its write open (its process gone), E2BIG past MESH_NODES_MOST records, EINVAL where the records lie
   outside the region, ENOENT where the map holds no record of this node (hdr `node`); read under the map seqlock
   (mesh.h mesh_map_whole) */
int mesh_nodes_read(struct mesh_ctx *,struct mesh_nodes *);
/* mesh_joined, struct mesh_cable and mesh_cables, which name records and cables, are mesh-map.h's, bridge-side (review
   R1, C1) */
#endif
