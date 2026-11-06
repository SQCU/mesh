#include <signal.h>
#include "mesh-dataflow.h"
/* the library's own internals that reduce the map to node pairs (mesh_cables, mesh_joined, mesh_nodes_read) and wait on
   the ports' data paths (mesh_links, mesh_transfers_start): the bridge-side region types, which no client header
   includes (review R1, C1) */
#include "mesh-map.h"
/* mesh_transfers_start's declaration: defined here, beside the map reader it reads */
#include "mesh-call.h"
/* mesh_view_values' declaration, defined here beside mesh_view_compose (2026-11-06, design/compilation-coordination.md,
   "The root cause of the copies, and the revised work list", R-10; "The revisions of 14b9a757") */
#include "mesh-collective.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
/* mach_vm_remap, mach_task_self (mesh_view_compose) */
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
/* design/algorithm-sources.md#programtensor */
/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): a plan's claim and its record in a head row (sub-assignment 5's and owner L's form of
   2026-11-04: any number of claims, each taken by first fit over MESH_ROW_OWN's and the arena's bits at its plan call,
   its head's `owner`, `closed`, `pages` and `start`, its entry's runs and generation and its publication's `state`),
   the walk of the heads (mesh_head_after, mesh_bit_next) and the standing plan read at a head (mesh_head_state) are
   removed.  Why: point 5: "place the whole program's values in one offline allocation at compile time.  These go:
   `mesh_claim_take`'s first fit and the claim's head record; `mesh_plan_head`'s walk of every claim head ...;
   `ENOMEM` from fragmentation"; "the serialized operand's place is a constant the region's layout names once, as the
   bridge's own pages are "laid out once with the region" (disjoint-pages A3).  Nothing walks heads to find it."  The
   program's record is its serialized operand's root (mesh.h struct mesh_program, hdr `program_off`), written by
   mesh_plan, closed by the program's retirement (mesh_retire), each of its plans read at its place
   (mesh_program_plan) and each closed alone by mesh_plan_retire, and the root released by the bridge parent once no
   port holds a plan of it (mesh_retired_release).  What must stay: nothing else takes rows or pages from the region;
   the bridge's own areas laid out with it (mesh-map.h mesh_layout) */

/* design/algorithm-sources.md#programtensor */
/* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 "Sharing the region between programs"
   (sub-assignment 7): the client's rows are no longer let go here (main's form cleared each owned row's bit of
   MESH_ROW_OWN, the pages kept by MESH_ROW_HOT until a reclaim); nothing of a claim is let go until the bridge's
   release (mesh_retired_release).
   2026-11-04, sub-assignment 5: every claim the client owns is closed, where sub-assignment 7's 2026-11-03 form closed
   the first head it found, the client's one claim, taken at attach.  Why: a client holds a claim a plan, and a retired
   plan's claim stands beside the next plan's until the bridge releases it (design/disjoint-pages-change-request.md,
   Appendix C, C3.4).  What must stay: a retired claim's rows and pages held until that release.
   2026-11-04, owner L (design/plan-to-bridge-change-request.md R8; e6bd7db2's section 6, "Retirement and the deferred
   release", as design/compilation-coordination.md, Q5 takes it): every claim whose owner is the client closed
   (release), then hdr `plans` raised and woken (mesh.h mesh_notify), and the bridge's control word notified (M26, its
   event loop's wake for the release); reversing sub-assignment 5's rulings on owner 7's finding and review 15's
   finding 2, the compare-and-swap of hdr `client` to the retirer's word, hdr `hops` 0 and `state` 0 (release), a
   fence and hdr `plan` woken before the claims closed, and `client` cleared after.  Why: those words are vacant
   (mesh.h 110); each plan is in its claim, and a plan stands only while its claim is open (mesh_head_state), so the
   close is the unpublishing, and no reader is led from the header to a state section the bridge may release.  A
   claim's pages are each plan's (R8: "each plan claims exactly its pages and releases them on retirement, by deferred
   reclamation"; P. E. McKenney and J. D. Slingwine, Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004).
   What must stay: nothing let go here; a closed claim held until the bridge's release.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"), reversing the notes above where they close claims: the program's root (mesh.h struct
   mesh_program, at the layout's one place) closed (`closed` 1, release) where its `owner` (acquire) is `client`, then
   hdr `plans` raised and woken and the bridge's control word notified, as before.  Why: point 5, "These go:
   `mesh_claim_take`'s first fit and the claim's head record; `mesh_plan_head`'s walk of every claim head"; "Release
   at retirement stays deferred reclamation (RCU)"; the root is the program's one record (A9, "What the bridge runs":
   "the bridge reads the serialized operand at a place its compilation fixed, and searches for nothing").  What must
   stay: nothing let go here; a closed root held until the bridge's release.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4; "The
   revisions of 55b9cbad"), reversing the root's `owner` and `closed`: the program's operand (mesh.h struct mesh_algo)
   retired where its holder, hdr `client` (acquire), is `client` and its `published` (acquire) is not 0: `retired` =
   `published` (release), then hdr `plans` raised and woken and the bridge's control word notified, as before.  Why:
   G4, "keeps the claim head record's `owner`, `closed` and start under a new name, which point 5 removes ... Its
   fields are what A9 and the failport's lifecycle need: the operand's publication (B2's single writer) and the holder
   the heartbeat already knows, reused as the failport's code is".  Its callers store the one value `published`: the
   program's process (mesh_detach, or its retirement before it plans again) and the bridge parent for a gone holder
   (mesh-records.c clients_turn, mesh-link.c region_adopt), so `retired` only ever names the operand published.  What
   must stay: nothing let go here; the retired operand held until the bridge's release
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-8: "One
   writer a word (W11; Lamport's single-writer registers): the program writes `retired`, the bridge writes only
   `released`, and a gone holder is read from the heartbeat's words"; "The revisions of 14b9a757", R-8), reversing "and
   the bridge parent for a gone holder": the program's process is this function's one caller (mesh_detach, and its
   retirement before it plans again: rdma/mesh_compile.py RETIRE), so `retired` has one writer.  The bridge parent
   reads a gone holder by mesh-holders.h mesh_algo_holder and mesh_holds, raises `plans` for it and releases its
   operand (mesh_retired_release), writing no `retired`.  The publication is read before the holder (R-3).  What must
   stay: nothing let go here; the retired operand held until the bridge's release */
void mesh_retire(struct hdr *m,uint64_t client){
  struct mesh_algo *algo=mesh_algo(m);
  const uint64_t published=atomic_load_explicit(&algo->published,memory_order_acquire);
  if(atomic_load_explicit(&m->client,memory_order_acquire)==client && published)
    atomic_store_explicit(&algo->retired,published,memory_order_release);
  mesh_notify(&m->plans);
  /* design/prepared-machine.md#M26 */
  mesh_control_notify(m);
}

/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): mesh_bits_take, mesh_run_take (first fit: D. E. Knuth, The Art of Computer Programming,
   vol. 1, 2.5), mesh_claim_take, mesh_claim_close and mesh_plan_head (the test of a handle by a walk of every claim
   head) are removed, with the claim's record in its head row (its `owner`, `closed`, `pages` and `start`; its entry's
   window and local runs and generation; its publication's `state`).  Why: point 5, "These go: `mesh_claim_take`'s
   first fit and the claim's head record; `mesh_plan_head`'s walk of every claim head in `mesh_transfers_start` and
   `mesh_plan_retire`, a run-time check; `ENOMEM` from fragmentation"; design/plan-to-bridge-change-request.md R8,
   corrected: "Nothing takes pages at run time, and no handle is looked up at run time".  What must stay: no call site
   but mesh_plan names rows or pages */

/* mesh-dataflow.h mesh_plan_retire: the program's retirement of one compiled function
   (design/aot-compilation-change-request.md, appendix item 6; design/compilation-coordination.md, "The appendix's
   revisions", N3, as NVIDIA, CUDA Programming Guide §4.2.3.1, cudaGraphExecDestroy(), destroys one instantiated
   graph).
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"), reversing the head's close by mesh_plan_head's walk and its ENOENT: the plan's state section
   at `plan`, a constant of the plan (mesh-collective.h struct mesh_function `plan`), `closed` 1 (release), then hdr
   `plans` raised and woken (mesh.h mesh_notify), so each port ends it (mesh_program_plan reads it closed) and clears
   its `held` bit; its pages stay the program's until the program's root is released (mesh_retired_release).  Why:
   point 5, "`mesh_plan_head`'s walk of every claim head in `mesh_transfers_start` and `mesh_plan_retire`, a run-time
   check", goes; appendix item 6, as corrected: "A function's pages are released when the program retires it or
   itself, by deferred reclamation".  What must stay: the close a release, `plans` raised and woken after it; nothing
   let go here */
void mesh_plan_retire(struct mesh_ctx *c,uint64_t plan){
  atomic_store_explicit(&mesh_state_at(c->M,plan)->closed,1,memory_order_release);
  mesh_notify(&c->M->plans);
}

/* 2026-11-04, sub-assignment 5: the claim at attach (mesh_claim) and the release before it, waited for while a port's
   `prepared` held a plan, bounded by MESH_PLATFORM_NS (sub-assignment 7's 2026-11-03 form; review 12, finding 3), are
   gone: attach claims nothing, and a retired client's claim is released on the bridge's own turn, which nobody waits
   for.  Why: design/disjoint-pages-change-request.md, Appendix C, C3.4 and C3.6 ("A port that keeps a retired client's
   plan past MESH_PLATFORM_NS: gone with the claim at attach"), and C4.
   2026-11-04, owner L (design/plan-to-bridge-change-request.md R8: "Attaching holds nothing, and there is no client
   word and no EADDRINUSE"; design/compilation-coordination.md, "The region"; e6bd7db2's section 6, "Attach takes
   nothing"): reversing sub-assignment 5's form, the compare-and-swap of hdr `client` (EADDRINUSE where another live
   client held it, a gone one retired first) and hdr `client_start` written.  Why: a program holds its plans' claims and
   nothing else, and any number of programs attach at once.  The client word is minted from hdr `serial` as before, bit
   63 `device_client`'s, and names each of the program's claims (its heads' owner word).  The version is read again after the mint (tag pre-return-20261103 rdma/mesh-dataflow.c mesh_attach: a client
   attaching as the heartbeat expires the region reads it expired here, and is let go, nothing of it held).  What must
   stay: nothing written in the region but hdr `serial`; the version read again.
   2026-11-04, owner L2 (design/compilation-coordination.md, "The appendix's revisions", N5): the context's `epoch` and
   `state`, which started at 0 here, gone with struct mesh_ctx's fields.  Why: the context holds no plan; a plan's
   handle is mesh_plan's `*plan` (N2).  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G3 and G4; "The
   revisions of 55b9cbad"): the context's `start` set to this process's start, mesh_started(getpid()) (mesh-holders.h,
   the failport's own probe), beside the client word, and nothing written in the region for it; mesh_plan copies both
   into the operand's holder, hdr `client` and `client_start`, as it publishes.  Why: G3, "The planner ... reads
   `getpid()` ... Deciders see pages"; G4, "the holder the heartbeat already knows, reused as the failport's code is".
   What must stay: nothing written in the region but hdr `serial`; the version read again */
int mesh_attach(struct mesh_ctx *c,const char *name){
  if(c->M) return 0;
  if(!name) name=getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int file=shm_open(name,O_RDWR,MESH_MODE);
  if(file<0) return errno;
  struct stat info;
  if(fstat(file,&info)){ int error=errno; close(file); return error; }
  struct hdr *memory=mmap(NULL,(size_t)info.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,file,0);
  int error=errno;
  if(memory==MAP_FAILED){ close(file); return error; }
  if((size_t)info.st_size<sizeof *memory || memory->magic!=MESH_MAGIC || memory->version!=MESH_VERSION || memory->length>(uint64_t)info.st_size){
    munmap(memory,(size_t)info.st_size); close(file); return EINVAL;
  }
  uint64_t client=(((atomic_fetch_add_explicit(&memory->serial,1,memory_order_relaxed)+1)&UINT64_C(0x7fffffff))<<32)|(uint32_t)getpid();
  const uint64_t device=atomic_load_explicit(&memory->device_client,memory_order_seq_cst);
  client|=(~device)&(UINT64_C(1)<<63);
  if(__atomic_load_n(&memory->version,__ATOMIC_SEQ_CST)!=MESH_VERSION){ munmap(memory,(size_t)info.st_size); close(file); return EINVAL; }
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.client=client,.fd=file,.start=mesh_started(getpid())};
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_retire(c->M,c->client);
  int status=munmap(c->M,c->len);
  int error=status?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* design/algorithm-sources.md#programtensor */
/* the plan's binding of a copy's rows to its pages (mesh-collective.h mesh_plan): no page is taken here, only named */
void mesh_backing_bind(struct mesh_ctx *c,uint32_t first,uint32_t pages,uint32_t page,uint32_t index){
  for(uint32_t offset=0;offset<pages;offset+=c->M->block){
    struct mesh_page_entry *entry=&mesh_page(c->M)[first+offset/c->M->block];
    atomic_store_explicit(&entry->mapping,((uint64_t)index<<32)|(page+offset),memory_order_relaxed);
    atomic_store_explicit(&entry->address,(uintptr_t)mesh_at(c->M,page+offset)-(uintptr_t)c->M,memory_order_relaxed);
  }
}

/* mesh-dataflow.h mesh_view_compose: ncclMeshPageTableCompose's reservation and remap (mesh 1b372d09^ rdma/nccl-
   mesh.c), restored as an effector (2026-11-05; design/compilation-coordination.md, "The documents' corrections of
   2026-11-05", point 6: "Give an operand typed as one run its run as a mapping written when the plan is made:
   consecutive virtual pages over its pages, as `ncclMeshPageTableCompose`'s `mach_vm_remap` view was on 2026-11-02
   (PB R2)"; design/plan-to-bridge-change-request.md R2, corrected: "Such an operand's run is address translation, fixed
   when the plan is made"; "The revisions of 4e78f6f4").  Its reservation is that function's, mmap of `bytes` of
   PROT_NONE anonymous memory, MAP_PRIVATE | MAP_ANON (sys/mman.h), aligned to the system's page as mmap gives it
   (that function's `alignment` operand, which nothing here chooses, is the page); each row's block then remapped over
   it at its place k x block x pgsz with that function's call, mach_vm_remap(mach_task_self(), &at, size, 0,
   VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self(), source, FALSE, &current, &maximum, VM_INHERIT_NONE) (the
   SDK's mach/mach_vm.h: "kern_return_t mach_vm_remap (vm_map_t target_task, mach_vm_address_t *target_address,
   mach_vm_size_t size, mach_vm_offset_t mask, int flags, vm_map_t src_task, mach_vm_address_t src_address, boolean_t
   copy, vm_prot_t *cur_protection, vm_prot_t *max_protection, vm_inherit_t inheritance)"; mach/vm_statistics.h
   VM_FLAGS_FIXED, VM_FLAGS_OVERWRITE "delete any existing mappings first"; mach/vm_inherit.h VM_INHERIT_NONE "absent
   from child"), its source this task's mapping of the region at the row's binding (struct mesh_page_entry `address`, a
   region offset: mesh_backing_bind), and that function's mapping of a refusal, KERN_NO_SPACE or
   KERN_RESOURCE_SHORTAGE ENOMEM, any other EINVAL (mach/kern_return.h), with the reservation unmapped whole.  It
   decides nothing: the rows, their order and their count are the caller's (mesh_plan's), and a row's block is where
   its binding says (design/page-maps-2026-11-02.md at 6255e92b: "effectors know about pointers and pages but never
   make any decisions ever").  What must stay: no copy (copy FALSE: the view shares the region's pages); nothing mapped
   after a failure */
int mesh_view_compose(struct mesh_ctx *c,const uint32_t *rows,size_t count,struct mesh_view *view){
  *view=(struct mesh_view){NULL,0};
  if(!count)return 0;
  struct hdr *m=c->M;
  const size_t block=(size_t)m->block*m->pgsz,bytes=count*block;
  char *base=mmap(NULL,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
  if(base==MAP_FAILED)return errno;
  /* 2026-11-05 ("The revisions of 4e78f6f4", the coordinating implementer's hunk): one remap a run of rows whose
     blocks are consecutive in the region, as ncclMeshPageTableCompose remapped each run of its physical runs
     (1b372d09^ rdma/nccl-mesh.c: `mesh_page_run_add`, which merged a page adjacent to the run before it, then one
     mach_vm_remap a run), where each block was remapped alone.  The allocator's chains give an item ordered with no
     other consecutive blocks (mesh-graph.h mesh_graph_chains), so a weight's blocks are one run.  What must stay:
     row k's block at base + k x block */
  for(size_t k=0,run;k<count;k+=run){
    mach_vm_address_t at=(mach_vm_address_t)(uintptr_t)(base+k*block);
    const uint64_t first=atomic_load_explicit(&mesh_page(m)[rows[k]].address,memory_order_relaxed);
    for(run=1;k+run<count &&
              atomic_load_explicit(&mesh_page(m)[rows[k+run]].address,memory_order_relaxed)==first+run*block;run++);
    const uintptr_t source=(uintptr_t)m+first;
    vm_prot_t current,maximum;
    const kern_return_t mapped=mach_vm_remap(mach_task_self(),&at,run*block,0,VM_FLAGS_FIXED|VM_FLAGS_OVERWRITE,
                                             mach_task_self(),source,FALSE,&current,&maximum,VM_INHERIT_NONE);
    if(mapped!=KERN_SUCCESS){
      munmap(base,bytes);
      return mapped==KERN_NO_SPACE || mapped==KERN_RESOURCE_SHORTAGE?ENOMEM:EINVAL;
    }
  }
  *view=(struct mesh_view){base,bytes};
  return 0;
}
/* mesh-dataflow.h mesh_view_release: ncclMeshPageTableDestroy's unmapping of its view (1b372d09^, through
   ncclMemFree), munmap of the reservation whole (sys/mman.h), which takes the remapped blocks with it; nothing where
   `address` is NULL (an empty view) */
void mesh_view_release(struct mesh_view view){
  if(view.address)munmap(view.address,view.bytes);
}
/* mesh-collective.h mesh_view_values: the program's one mapping of its `run` values (2026-11-06,
   design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-10: "the mapping's
   row list in one mesh-dataflow helper over the plan's rows"; W13: "the mapping's row list in two effectors
   (mesh_compile.py 2289; e2b_replay.m 1437), to be one mesh-dataflow helper"; "The revisions of 14b9a757", R-10).  The
   row list the two effectors each built (rdma/mesh_compile.py _mapped; metal-microbench tools/e2b_replay.m main), in
   their one order, which mesh_plan's mapping had: function by function, value by value in `values`' order, each `run`
   value's copies in order, block k of copy c its row section.first + c x section.stride + k (mesh-section.h: "copy c's
   rows first + c x P + k, k < P"), which for c < count and k < P are first + j, j < count x P; so value i's copy 0 lies
   at the count of rows before it times block x pgsz, and its copy c section.pages x pgsz after that; offset[i]
   UINT64_MAX where `run` is 0.  The list is this call's own, freed
   once mesh_view_compose has read it ("reads it and keeps nothing", as e2b's was), and it is the mapping's, never a
   program value: no page of the region is taken.  An effector: the rows, their order and their count are the plan's,
   and nothing is decided here (design/page-maps-2026-11-02.md at 6255e92b: "effectors know about pointers and pages but
   never make any decisions ever").  0, or mesh_view_compose's error (ENOMEM also where the list cannot be held), with
   nothing mapped, `*view` {NULL, 0} and every offset UINT64_MAX.  What must stay: one mapping a program, every `run`
   value's copies consecutive in it in row order */
int mesh_view_values(struct mesh_ctx *c,const struct mesh_function *functions,uint32_t count,uint64_t *offset,
                     struct mesh_view *view){
  const size_t block=(size_t)c->M->block*c->M->pgsz;
  size_t rows=0,values=0;
  for(uint32_t f=0;f<count;f++)
    for(uint32_t i=0;i<functions[f].values;i++,values++){
      const struct mesh_section s=functions[f].value[i].section;
      offset[values]=functions[f].value[i].run?rows*block:UINT64_MAX;
      if(functions[f].value[i].run)rows+=(size_t)s.count*s.stride;
    }
  uint32_t *row=malloc((rows?rows:1)*sizeof *row);
  int error=row?0:ENOMEM;
  for(uint32_t f=0,at=0;!error && f<count;f++)
    for(uint32_t i=0;i<functions[f].values;i++){
      const struct mesh_section s=functions[f].value[i].section;
      for(uint32_t k=0;functions[f].value[i].run && k<s.count*s.stride;k++)row[at++]=s.first+k;
    }
  if(!error)error=mesh_view_compose(c,row,rows,view);
  free(row);
  if(error){
    *view=(struct mesh_view){NULL,0};
    for(size_t i=0;i<values;i++)offset[i]=UINT64_MAX;
  }
  return error;
}

/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *c,uint32_t row,uint64_t address){
  if(row!=MESH_ABSENT)atomic_store_explicit(&mesh_page(c->M)[row].device,address,memory_order_relaxed);
}

/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_holding_retained, adapted to the page lists: whether a closed
   claim's pages may still be named by a request a port's child posted (a receive not landed, a SEND not done).  At the
   tag, a holding while its port child lived; here, a port loading the claim's plan (its `prepared`) or
   holding it (its bit of the plan's `held`), as the dated note below states.  Every port of the link table is read,
   not only those a plan's hops name: a port the map no longer lists may still hold it.  A port in phase MESH_STOPPED
   retains too, though the review (R1, round 2) asked to skip it: a refusal (mesh-flow.c) holds no request, but a port
   lost while it serves is MESH_STOPPED as well (mesh-ports.c port_lost: its child recycled, `open` 0), and its
   holding stays while that child lives, its receives posted until the reap; the region tells the two apart by
   nothing.
   2026-11-04, sub-assignment 5: the rule as given, a port's `prepared` equal to the claim's generation, where
   sub-assignment 7's 2026-11-03 form retained every closed claim while any port held any plan (main's header kept no
   generation of a retired client's plan).  Why: a claim's head records its generation (mesh_claim_take), and a
   retired plan's claim is released while the next plan is served (design/disjoint-pages-change-request.md, Appendix
   C, C3.4).
   2026-11-04, owner L (design/compilation-coordination.md, "The region" and Q5; e6bd7db2's section 6, "Retirement and
   the deferred release", and its D3): a port's `prepared` is the generation of the claim it is loading now (0 none),
   set and cleared within one load, and a port that holds a plan has its bit of the plan's struct mesh_state `held`,
   which it clears as its last access once it has ended the plan (mesh-flow.c; a port lost with requests outstanding
   keeps its bit until its child is gone).  So a closed claim is kept where any port's `prepared` is its generation
   (a load in flight: the hazard pointer's announcement, M. M. Michael, IEEE TPDS 15(6), 2004) or its plan's `held` is
   not 0; its plan's state section is its head's publication `state`, 0 where the plan never stood (no port loaded
   it).  The ports' `prepared` are read first (acquire), then `held` (acquire): a port sets its `held` bit before it
   withdraws `prepared`, so a `prepared` read after the withdrawal is followed by a `held` read that sees the bit.
   Reversing the 109 form, where `prepared` held the one plan's generation while the port served it.  Why: any number
   of plans stand at once, and a port holds each of them in that plan's words.  What must stay: no STOPPED port
   skipped while a loss sets it; every port of the link table read, not only those a plan's hops name.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): the program's root kept where any port's `prepared` is the root's `generation` (a port
   loading a plan of it: mesh-flow.c plan_load) or any of its plans' `held` is not 0, each plan's state section read
   at its place, the root's `table` word k for k below its `plans`, closed or standing (a plan the program retired
   alone, struct mesh_state `closed` 1, is held until every port ends it), where a claim's head named one generation
   and one plan.  Why: point 5, "Release at retirement stays deferred reclamation (RCU)"; the root at the layout's one
   place (mesh.h struct mesh_program).  What must stay: the ports' `prepared` read first, then `held`, each with
   acquire; every port of the link table read.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4; "The
   revisions of 55b9cbad"): the operand (mesh.h struct mesh_algo) kept where any port's `prepared` is its `published`
   (the generation the ports announce, struct mesh_port_info) or any of its plans' `held` is not 0, each plan's state
   section at the operand's `table` word k for k below its `plans` (mesh_algo_state), where the root's `generation` and
   table were read.  What must stay: as above */
static int mesh_holding_retained(struct hdr *m,const struct mesh_algo *algo,uint64_t published){
  const uint32_t links=m->links>MESH_PORTS?m->links:MESH_PORTS;
  for(uint32_t p=0;p<links;p++)
    if(atomic_load_explicit(&mesh_links(m)[p].port.prepared,memory_order_acquire)==published)return 1;
  for(uint32_t k=0;k<algo->plans;k++)
    if(atomic_load_explicit(&mesh_algo_state(m,algo,k)->held,memory_order_acquire))return 1;
  return 0;
}

/* design/algorithm-sources.md#programtensor */
/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_retired_release, adapted: closed claims released, but none while
   a port's requests may name their pages (mesh_holding_retained); the count kept, which the tag's clients_turn keeps
   `unreleased` by.  Main's form released every closed row at once.  This is deferred reclamation (RCU's call_rcu: P. E.
   McKenney and J. D. Slingwine, Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004, an epoch announced and
   read again; design/disjoint-pages-revision-coordination.md, D10), on the bridge parent's one event-loop thread
   (mesh-records.c clients_turn, mesh-link.c region_adopt), and nobody waits for it.
   - A claim is released whole, from its head row (mesh_claim_take's record): its window and local runs' bits of the
     arena, then its rows' bits of MESH_ROW_OWN, last, so no later claim takes a row whose head fields are still being
     read.  The head's fields are read before its owner word is taken back: they are stored before the owner word and
     changed only by the release that takes it, so a release that loses the exchange reads nothing another claim wrote.
     The winner returns the head's entry to a free row's form (mesh-link.c: `mapping` MESH_ABSENT, `address` and
     `device` 0) and its `pages`, `start` and `closed` to 0 before the rows' bits are cleared.
   - The releaser's half of the ports' announcement (D10; D3): a port stores its `prepared`, fences (seq_cst), and
     reads the claim's head again (its owner, generation, `closed` and publication `state`), withdrawing where it
     changed; the releaser reads a claim closed (acquire: what its closer stored before happens before), fences
     (seq_cst), and then reads the ports' `prepared` and the plan's `held` (mesh_holding_retained).  Of the two fences
     one precedes the other, so either the port reads the close and withdraws or the releaser reads its `prepared` and
     keeps the claim (C++20 [atomics.order] p4.4: a fence that happens before
     a load coherence-ordered before a store that happens before another fence precedes that fence).  One fence for
     each closed claim, after its acquire and before its ports are read, not one before the walk: a fence before that
     acquire orders nothing its closer stored.
   2026-11-04, sub-assignment 5: reversing sub-assignment 7's 2026-11-03 form, which released a retired client's one
   claim from two head rows (`pages` and `mapping` of each, the rows' count in the first's `mapping` high half), every
   claim retained while any port held any plan, and which a client's mesh_attach also called.  Why: claims per plan
   with their generation (design/disjoint-pages-change-request.md, Appendix C, C3.4), and the release the bridge
   parent's alone (D10; the coordinator's ruling on owner 7's finding).  What must stay: mesh_holding_retained's rule;
   the rows let go after the pages, and only by the exchange's winner.
   2026-11-04, owner L (design/compilation-coordination.md, "The region"): a claim is kept where a port's `prepared` is
   its generation or its plan's `held` is not 0 (mesh_holding_retained), the walk over the heads alone
   (mesh_head_after, the next head read before the release), and the head's `start` returned to 0 with its other
   fields.  What must stay: one seq_cst fence for each closed claim, after its acquire and before its ports and its
   `held` are read.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5: "Release at
   retirement stays deferred reclamation (RCU)"; "The revisions of 4e78f6f4"), reversing the notes above where they
   release claims: the program's root (mesh.h struct mesh_program) released where it is closed (`owner` not 0, then
   `closed` 1, each read with acquire): a seq_cst fence, then the root kept where any port's `prepared` is its
   `generation` or any of its plans' `held` is not 0 (mesh_holding_retained), else its `owner` 0 (release), so a next
   program may plan (mesh-collective.h mesh_plan: EBUSY while it is not 0); 1 where the root is kept, else 0.  Nothing
   else is let go: the program's rows and pages are its one allocation's, numbered from the region's first row and
   first block (mesh-collective.h mesh_plan), and the next program's plan call writes every word it uses, so no bit,
   row or entry is returned to a free form (the first fit's bitmaps go: mesh.h MESH_ROW_OWN's note).  The bridge
   parent the one caller (mesh-records.c clients_turn, mesh-link.c region_adopt) and the root's one releaser, so the
   store needs no exchange.  The fence's pairing is as above: a port announces the generation in its `prepared`,
   fences and reads the root again (mesh-flow.c plan_load), and the releaser reads the root closed, fences and reads
   the ports' `prepared`, so either the port reads the close and withdraws or the release reads its announcement.  Why:
   point 5; design/aot-compilation-change-request.md, appendix item 6, as corrected: "A function's pages are released
   when the program retires it or itself, by deferred reclamation, as R8 cites"; the root at the layout's one place
   (A9, "What the bridge runs": "searches for nothing").  What must stay: the fence after the close is read and before
   the ports and `held` are read; nobody waits for it.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", B2 and G4, and
   "Answered in the documents", EB-1; "The revisions of 55b9cbad"), reversing the root's `owner` 0 and "nobody waits
   for it": where the operand's `published` (acquire) is not 0, its `retired` (acquire) is `published` and its
   `released` (acquire) is not: a seq_cst fence, then kept where any port's `prepared` is `published` or any of its
   plans' `held` is not 0 (mesh_holding_retained), else `released` = `published` (release) and its waiters woken
   (os_sync_wake_by_address_all, region-shared); 1 kept, else 0.  mesh_plan waits on `released` before it writes a next
   operand (mesh-collective.h), as RCU's writer waits for the grace period to end.  Why: EB-1, "An EBUSY that hands a
   retry to the program's next call is not deferred reclamation's form.  In RCU's form the writer waits for the grace
   period to end (`synchronize_rcu`), here the bridge's release of the old root"; B2, "The replan must wait for the
   release, bounded by MESH_PLATFORM_NS".  The bridge parent is `released`'s one writer (C3's rule 2, "Each state word
   has one writer"), so the store needs no exchange.  The fence pairs as above: a port announces `published` in its
   `prepared`, fences and reads the operand again (mesh-flow.c plan_load), and this side reads it retired, fences and
   reads the ports.  What must stay: the fence after the retirement is read and before the ports and `held` are read;
   the store after them, then the wake
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-8 and
   R-10; "The revisions of 14b9a757"), reversing "its `retired` (acquire) is `published`" as the one case released:
   - The operand is released where it is published and not yet released (mesh.h mesh_algo_unreleased, R-10's one form
     of ""published and not yet released"", where this function read `published` and `released` itself) and its holder
     no longer holds it: the holder as mesh-holders.h mesh_algo_holder reads it ({0} where the operand is not open, so
     where the program retired it), gone by mesh_holds(pid, start) alone.  Why: R-8, "the program writes `retired`, the
     bridge writes only `released`, and a gone holder is read from the heartbeat's words" (W11), so a gone holder's
     operand is published and unretired here, and its release is this test's, not a `retired` the bridge wrote.
   - Every word of the operand is read after the publication's acquire (R-3: "every field of the published operand is
     read through the acquired publication, never before it"; mesh_holding_retained's `plans` after it).
   What must stay: the fence after the release's test and before the ports and `held` are read; the store after them,
   then the wake; `released` this function's alone */
uint32_t mesh_retired_release(struct hdr *m){
  struct mesh_algo *algo=mesh_algo(m);
  const uint64_t published=mesh_algo_unreleased(m);
  const struct mesh_algo_holder holder=mesh_algo_holder(m);
  if(!published || mesh_holds(holder.pid,holder.start))return 0;
  atomic_thread_fence(memory_order_seq_cst);
  if(mesh_holding_retained(m,algo,published))return 1;
  atomic_store_explicit(&algo->released,published,memory_order_release);
  os_sync_wake_by_address_all(&algo->released,sizeof algo->released,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  return 0;
}

/* 2026-11-04, sub-assignment 5: mesh_publication_prepare is removed, which read a row's supplied words into the GPU's
   publication records (struct prepared_publication, gone with layout 109's MESH_SUPPLIED and MESH_WORD_STRIDE: mesh.h).
   Its one caller, mesh-metal.m, reads a range's first row's publication itself (its `targets[0]` the supplied word,
   mesh.h struct mesh_publication).  What must stay: one store releases a range */

/* The map reader and the client's start (mesh-dataflow.h mesh_nodes_read, mesh-call.h mesh_transfers_start).  The
   map's records, a cable's both ends and its ports stay in this file: the collective is given node pairs and summed
   capacity (struct mesh_nodes), and the start a status.  No graph algorithm is here (design/rubrics.md C9; the C9
   review of 7da3c125 asked these into mesh-graph.h): they read the map's record format into the node-pair weights
   mesh-graph.h's functions take as operands, and mesh-graph.h knows no cable.  The matching of two records' cables is
   one copy, mesh_cables, which the join (mesh_join: mesh_joined, the bridge's map in mesh-records.c), the reader's
   capacity and the bridges' rule (mesh-flow.c link_prepare) share. */

/* the cables both records list for each other (mesh.h struct mesh_reach: each record lists a cable from its own side,
   so b's entry names a's node with the two ports crossed), in `out` (MESH_PORTS of room: a record lists at most
   MESH_PORTS) sorted by key, the lower node id's port index, then the higher's; each one's `bandwidth` and `qps` the
   lesser of the two records' (the pairing's, the same at both ends; the bridge's words); their count.  A cable one record
   lists alone (the other's record not yet heard) is none: both ends' rule reads a cable from both records */
uint32_t mesh_cables(const struct mesh_record *a,const struct mesh_record *b,struct mesh_cable *out){
  const int lower=a->node<b->node;
  uint32_t n=0;
  for(uint32_t i=0;i<a->reached && i<MESH_PORTS;i++){
    const struct mesh_reach *e=a->reach+i;
    if(e->node!=b->node)continue;
    for(uint32_t k=0;k<b->reached && k<MESH_PORTS;k++){
      const struct mesh_reach *f=b->reach+k;
      if(f->node!=a->node || f->port!=e->peer_port || f->peer_port!=e->port)continue;
      const struct mesh_cable cable={.key={lower?e->port:e->peer_port,lower?e->peer_port:e->port},
        .qps=e->qps<f->qps?e->qps:f->qps,.bandwidth=e->bandwidth<f->bandwidth?e->bandwidth:f->bandwidth};
      uint32_t at=n++;
      for(;at && (out[at-1].key[0]>cable.key[0] || (out[at-1].key[0]==cable.key[0] && out[at-1].key[1]>cable.key[1]));at--)
        out[at]=out[at-1];
      out[at]=cable;
      break;
    }
  }
  return n;
}

/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_join and mesh_joined, mesh_joined verbatim; mesh_join over
   mesh_cables, so the matching is one copy: its count the cables both records list for each other (the tag's, the
   fewer of the two ends' counts of entries naming the other, mesh_paired, which is gone), *bandwidth the least of
   theirs that either reports (0: none reports one) */
/* the one join of two nodes' records: the cables each one's bridge reaches the other over, 0 where none */
static uint32_t mesh_join(struct mesh_record *a,struct mesh_record *b,uint64_t *bandwidth){
  struct mesh_cable cables[MESH_PORTS];
  const uint32_t joined=mesh_cables(a,b,cables);
  uint64_t slowest=0;
  for(uint32_t c=0;c<joined;c++)if(cables[c].bandwidth && (!slowest || cables[c].bandwidth<slowest))slowest=cables[c].bandwidth;
  if(joined)*bandwidth=slowest;
  return joined;
}
int mesh_joined(struct mesh_record *a,struct mesh_record *b,double *beta){
  uint64_t bandwidth=0;
  const uint32_t joined=mesh_join(a,b,&bandwidth);
  if(joined)*beta=bandwidth?8e9/(double)bandwidth:0;
  return (int)joined;
}

/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_records_read, adapted: the map's records as one serial left them,
   the seqlock's reader (mesh.h mesh_map_whole), copied into r (MESH_NODES_MOST of room) with their count and the map's
   serial (hdr `map_serial`, moved only while `map` is odd); the read ended by `deadline` (the monotonic clock's ns),
   which its caller sets MESH_PLATFORM_NS past the start of its wait.  It differs from the tag's: no copy is allocated
   (the room is the reader's MESH_NODES_MOST, E2BIG past it); main has no expiry to register (mesh_expiry_set), so
   past the deadline it returns ETIMEDOUT, or ESRCH where the bridge's process is gone (its write left open); and an
   open write is waited on until the deadline, where the tag looked again each MESH_PAUSE_NS
   2026-11-06 (design/aot-compilation-change-request.md A7, corrected 2026-11-06: "The map a view is created from is the
   sequencer's, and the view's first message carries it: the records the sequencer's map holds when it creates the
   view.  Every member plans on those records"; "So every member of a view computes the same plan from the same
   operands"; "The revisions of 14b9a757", A7), reversing the map's records: the installed view's records (mesh.h hdr
   `view_nodes` of them at `view_off`, mesh-map.h mesh_view_records) and its identifier (hdr `view`) as `*serial`, read
   whole under the view's seqlock `view_map` as the map's are under `map`: `view_map` read (acquire) even, the words
   read, an acquire fence and `view_map` read again the same (mesh.h mesh_map_whole's two steps, which that function
   takes for `map` alone).  ESTALE where no view is installed (`view` 0), after a whole read: the implementer's decision
   (sub-agent D2's), from A7: "the program's sender TO-broadcasts nothing before its first view (Initial View Event)",
   and Chockler, Keidar and Vitenberg's "no send events occur before the first view_chng event" (Property 3.3), so a
   plan, whose sends are the view's, is made in some view; ESTALE is mesh_plan's "plan again" (mesh-collective.h).
   What must stay: one whole copy under the seqlock; the deadline's ESRCH and ETIMEDOUT as before */
static int mesh_view_copy(struct mesh_ctx *c,uint64_t deadline,struct mesh_record *r,uint32_t *count,uint64_t *serial){
  struct hdr *m=c->M;
  for(;;){
    const uint64_t before=atomic_load_explicit(&m->view_map,memory_order_acquire),now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(now>=deadline)return mesh_gone((pid_t)atomic_load_explicit(&m->bridge_pid,memory_order_acquire))?ESRCH:ETIMEDOUT;
    if(before&1){
      os_sync_wait_on_address_with_timeout(&m->view_map,before,sizeof m->view_map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                           OS_CLOCK_MACH_ABSOLUTE_TIME,deadline-now);
      continue;
    }
    const uint64_t nodes=atomic_load_explicit(&m->view_nodes,memory_order_relaxed),at=m->view_off;
    *serial=atomic_load_explicit(&m->view,memory_order_relaxed);
    const int fits=at>=sizeof *m && at<=c->len && nodes<=(c->len-at)/sizeof *r;
    if(fits && nodes<=MESH_NODES_MOST)memcpy(r,(char *)m+at,nodes*sizeof *r);
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&m->view_map,memory_order_relaxed)!=before)continue;
    if(!fits)return EINVAL;
    if(nodes>MESH_NODES_MOST)return E2BIG;
    if(!*serial)return ESTALE;
    *count=(uint32_t)nodes;return 0;
  }
}

/* the records' indices in ascending node-id order: order[i] the record of map index i */
static void mesh_map_order(const struct mesh_record *r,uint32_t count,uint32_t *order){
  for(uint32_t i=0;i<count;i++){
    uint32_t k=i;
    for(;k && r[order[k-1]].node>r[i].node;k--)order[k]=order[k-1];
    order[k]=i;
  }
}

/* a pair's capacity, the sum of its cables' bandwidths (mesh_cables), the rates the bridges' rule spreads its pages
   over; 0 where none joins them */
static uint64_t mesh_capacity(const struct mesh_record *a,const struct mesh_record *b){
  struct mesh_cable cables[MESH_PORTS];
  uint64_t sum=0;
  for(uint32_t c=mesh_cables(a,b,cables);c;c--)sum+=cables[c-1].bandwidth;
  return sum;
}

/* mesh-dataflow.h mesh_nodes_read.  2026-11-06 (design/aot-compilation-change-request.md A7, corrected 2026-11-06;
   mesh.h hdr `view`: "A plan is made on those records (mesh-dataflow.h mesh_nodes_read), its `serial` `view`"; "The
   revisions of 14b9a757", A7), reversing "`serial` is the map's (hdr `map_serial`)": the nodes, their indices,
   capacities and this node's index are the installed view's records' (mesh_view_copy), read as the map's were, and
   `serial` is the view's identifier; ESTALE where no view is installed, besides mesh-dataflow.h's errors.  What must
   stay: a node's index its place in ascending node-id order among the records; the pairs' capacities mesh_cables' */
int mesh_nodes_read(struct mesh_ctx *c,struct mesh_nodes *out){
  struct mesh_record r[MESH_NODES_MOST];
  uint32_t count,order[MESH_NODES_MOST];
  uint64_t serial;
  int error=mesh_view_copy(c,clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS,r,&count,&serial);
  if(error)return error;
  mesh_map_order(r,count,order);
  *out=(struct mesh_nodes){.serial=serial,.count=count,.self=MESH_ABSENT};
  for(uint32_t i=0;i<count;i++){
    out->node[i]=r[order[i]].node;
    if(out->node[i]==c->M->node)out->self=i;
    for(uint32_t j=0;j<i;j++)out->capacity[i][j]=out->capacity[j][i]=mesh_capacity(r+order[i],r+order[j]);
  }
  return out->self==MESH_ABSENT?ENOENT:0;
}

/* 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", SW): mesh_ports_held (the start's wait,
   bounded by MESH_PLATFORM_NS, until every port whose peer a hop names held the plan, its `held` bit, or refused it,
   its `refused[p]` code returned) and mesh_ports_named (that wait's port set, its last caller gone) are removed, with
   the reason sub-assignment 5 gave the wait (a port that pairs again loading the same plan, refused where its
   `executions` is above 0, which held only where the start raised it after every named port had loaded).  Why:
   design/aot-compilation-change-request.md, appendix item 2, "The code must: ... remove the start's wait for every
   port to load the plan, and the late-loading port's refusal of it (`mesh_ports_held`).  Nothing requires them.  "a
   send that arrives first is held until its receive is posted (§3), and a loss cancels the plan anyway""; the
   refusal is mesh-flow.c link_prepare's.  What must stay: the raise, then the fence, then the serial's read
   (mesh_transfers_start); ESTALE where the plan's serial is not the map's */
/* 2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, D.3.6; design/disjoint-pages-rulings-
   coordination.md, R3): after the raise of the plan's `executions` and the wake, a seq_cst fence, then hdr
   `map_serial` read; where it is not the plan's serial (struct mesh_state `serial`, every entry's, the map the plan
   was made from), ESTALE, the raise standing.  Why: a plan stands only while its map does.  Where the map moves, the
   bridge moves the serial, fences (seq_cst) and reads the plan's `executions`, then cancels every receive of the plan
   not landed for the executions below it, each an arrival on its count with no phase written (mesh-flow.c
   plan_stale); a raise after that read would leave the raised executions uncancelled, and their readers would end at
   their deadline (ETIMEDOUT), not at once.  The start raises, fences and reads the serial: of the two fences one comes
   first, and a load after the later one sees the store before the earlier one, so at least one side sees the other.
   Either the cancellation covers the raised executions, or the start reads the moved serial and returns ESTALE, and
   the program runs none of them and plans again on the map as it is.  The raise left standing reads no execution as
   landed that has not: `executions` only rises (C3.7, no state word set back while its plan stands), the program
   reads and releases nothing of an execution it does not run, and the plan's end (its claim's close) cancels every
   receive of it not landed.  That is its safety, not that nothing follows: the bridge has already set the raised
   executions' receives, which end with the plan as any cancelled run's do, by the bridge's own means (review 8's
   finding 1; review 10's finding 5).  The pairing is D10's announcement, fence and re-read
   (design/disjoint-pages-revision-coordination.md; K. Fraser, Practical lock-freedom, 2004), and the rule is the one
   mesh_ports_held applies before its wait (its ESTALE), applied after the raise; nobody waits.  The fence after the
   wake, so every raise is woken as before.
   2026-11-04, owner L (design/aot-compilation-change-request.md A7; design/compilation-coordination.md, Q1 and "The
   region"): the raise of the context's plan's `executions` from a to b is the program's submission of function
   executions [a, b) of the plan to its node's bridge ("A program submits each of its function executions to its
   node's bridge, and the bridge forwards it to the sequencer": the fixed sequencer of X. Défago, A. Schiper and P.
   Urbán, ACM Comput. Surv. 36(4), 2004, section 7.1, "a sender sends m to the sequencer"; P. Barham et al., Pathways,
   MLSys 2022, section 4.4); hdr `plans` is raised and woken after it (mesh.h mesh_notify), where hdr `plan` was woken,
   so the bridge sees the submission; the plan is the context's (`state`), and its serial the state section's, where
   they were hdr `state` and the entries' at hdr `hop_off`.  What must stay: the raise before the fence, the serial's
   read after it.
   2026-11-04, owner L2 (design/aot-compilation-change-request.md, appendix item 6; design/compilation-coordination.md,
   "The appendix's revisions", N2, N4, N5): the plan the caller's `plan`, its state section's region offset (mesh_plan's
   `*plan`, N2: the handle, as an instantiated graph's cudaGraphExec_t), where it was the context's `state`, gone; EINVAL where `plan` is 0 or its section lies outside the region, its entries tested by
   mesh_ports_named as before.  Each execution raised is one execution of the plan's compiled function (N4: "It may be
   launched any number of times without repeating the instantiation", §4.2.2).  Why: a program holds any number of
   compiled functions at once, each its own plan (item 6).  What must stay: the raise before the fence, the serial's
   read after it.
   2026-11-04, owner L2, the coordinator's ruling on review W1's finding 4: EINVAL unless `plan` passes the tests
   mesh-metal.m mesh_metal_transport_create applies (not 0, page-aligned, at or past hdr `data_off`, its struct inside
   the region) and names the state section of a plan standing at a claim head of this client (mesh_plan_head: the
   claim's record, mesh-dataflow.h mesh_claim_take; its owner word the client, `closed` 0, its publication `state`
   `plan`; N2), where the first form tested the offset's bounds alone.  Why: a retired handle's pages are released by
   the bridge and may be another claim's, so a raise through it would write a word that is not the plan's.  What must
   stay: EINVAL before anything is raised.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", SW): the raise with no wait before it,
   where it came after mesh_ports_held's (every port named holding the plan, its ESTALE where the plan's serial was not
   the map's, ETIMEDOUT at its bound, a port's refusal code), and the entries no longer tested by mesh_ports_named
   (the bridge's load tests them, mesh-flow.c plan_inside).  Why: design/aot-compilation-change-request.md, appendix
   item 2: "remove the start's wait for every port to load the plan, and the late-loading port's refusal of it
   (`mesh_ports_held`).  Nothing requires them"; the ESTALE stays, read after the raise as above.  What must stay: the
   raise before the fence, the serial's read after it; EINVAL before anything is raised
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS): a raise from a to b is one
   submission of the plan, appended to its submissions' queue (mesh.h struct mesh_state `submits`, `outstanding`,
   `submissions`): entry k mod `outstanding` its end b and the program's clock after one increment (L. Lamport, Time,
   clocks, and the ordering of events in a distributed system, CACM 21(7), 1978, IR1: "Each process Pi increments Ci
   between any two successive events"), then `submissions` k + 1 (release), before hdr `plans`' raise, with no wait
   and no check of the room (design/aot-compilation-change-request.md, appendix item 2: "Nothing waits to write, and
   nothing checks for a full queue"; the room is A10's bound on the calls outstanding at the program, mesh_plan's
   `outstanding`); EINVAL where the queue does not lie inside the region.  The program the queue's one writer; the
   bridge parent at the program's sender reads it (mesh-records.c orders_submit) and merges the program's plans' queues
   by the clock, so it TO-broadcasts the program's submissions in the order the program made them.  Why:
   design/aot-compilation-change-request.md A7: "It TO-broadcasts the program's submissions in the order the program
   makes them"; "With one sender, whose messages reach the sequencer first in, first out, Fig. 5's sequencer numbers
   them in the order they were sent".  What must stay: the raise before the fence, the serial's read after it; EINVAL
   before anything is raised.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 3f106747", BS): the queue's room `submit_room`,
   its self-timed buffer bound (mesh.h struct mesh_state's note), where it was `outstanding`; and, where the plan has a
   synchronization edge into this node (`waits`), the start that submits waits first, as the edge's sink, until the
   count its source raises reaches k - `wait_delay` (k this start's ordinal, the plan's `submissions` + 1), reading
   `wait_count` (acquire) and waiting on it (os_sync_wait_on_address_with_timeout, region-shared, woken where the bridge
   parent stores it: mesh-records.c count_take), for at most MESH_PLATFORM_NS (ETIMEDOUT), and ESTALE where the
   plan's serial is not the map's (its view or map has moved: the cancellation raises the count to UINT64_MAX and wakes
   it, mesh-flow.c plan_cancel), nothing raised in either case.  Why: design/aot-compilation-change-request.md, appendix
   item 2: "An added edge (vj, vi) holds "start(vi, k) ≥ end(vj, k − delay((vj, vi)))" (BSL (7)).  Its source raises a
   count that only rises ... and its sink reads that count before its k-th start"; "The only waits are at the sinks of
   the added synchronization edges, on a count that only rises.  Each ends when its source moves on, or when the view
   ends and the plan with it (`ESTALE`)"; the program's start is the sink's start, the sender's submission (A7) or a
   node's part of the call.  The bound is the rules' ("Every remaining wait ends in bounded time").  What must stay:
   the wait before the raise; nothing raised where it fails.
   2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"), reversing the notes above where they test `plan` and wait on one edge: no test of `plan`
   or of its submissions' queue (mesh_plan_head's walk and the EINVALs go: `plan` is a constant of the plan, mesh_plan's
   struct mesh_function `plan`, read as it is given); and the start that submits waits, before the raise, on each of
   the plan's synchronization edges into this node (mesh.h struct mesh_sync, `waits` of them at `syncs`) that holds
   at this start, k its ordinal (the plan's `submissions` before it) and K = k / `calls`: an entry whose `at` is k mod
   `calls` and whose `delay` is at most K, until its `count` (acquire) reaches (K - `delay`) x `every` + `position` + 1,
   the source's start of the call it waits for; each on os_sync_wait_on_address_with_timeout (region-shared, woken
   where the bridge parent raises it: mesh-records.c count_take), one deadline MESH_PLATFORM_NS from the start's first
   wait (ETIMEDOUT past it), ESTALE where the plan's serial is not the map's (the cancellation raises every entry's
   count to UINT64_MAX and wakes it: mesh-flow.c plan_cancel), nothing raised in either case; then the raise, the
   submission's entry, `plans`, the fence and the serial's read as before.  Why: point 5: "`mesh_plan_head`'s walk of
   every claim head in `mesh_transfers_start` and `mesh_plan_retire`, a run-time check" goes;
   design/aot-compilation-change-request.md A9, "What the bridge runs: the self-timed schedule": "Every address in it
   is fixed at compile time"; appendix item 2: "An added edge (vj, vi) holds "start(vi, k) ≥ end(vj, k −
   delay((vj, vi)))" (BSL (7)).  Its source raises a count that only rises ..., and its sink reads that count before
   its k-th start"; a plan has an actor for each call of its function (mesh.h struct mesh_state `calls`), each block's
   execution K calling it `calls` times, and an edge from a source of `every` calls an execution waits for its call
   at `position` in execution K - `delay` (BSL's iterations; point 4: the program's whole IPC graph).  What must stay:
   the waits before the raise, nothing raised where one fails; the raise before the fence, the serial's read after it.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", B1 and K1; "The
   revisions of 55b9cbad"), reversing the note above where it scans every entry, skips one by its place and its delay,
   and takes a deadline of its own:
   - The waits are the plan's row k mod `calls` alone (mesh.h struct mesh_state `rows`: entries rows[a] .. rows[a + 1]
     - 1 at `syncs`, mesh_wait_row), each until its `count` (acquire) >= K x `every` + `base` (struct mesh_sync), at
     once where that bound is 0 or below.  Why: K1, "the start's scan of every wait entry (mesh-dataflow.c 562): the
     waits are stored in rows keyed by call place, and each edge's delay is folded into its threshold", as MSCCL-IR's
     steps name their dependences by index (`depid`, `deps`).  The bound is compared with the count over the integers,
     in clang's 128-bit integer (`__int128`), so a bound of 0 or below holds at once and a count of UINT64_MAX (a
     cancellation, mesh-flow.c plan_cancel) holds every bound, as it did, with no branch per K against a delay: the
     implementer's (sub-agent D's) minimal form, the comparison the interface states, where a 64-bit compare would
     read a negative bound as a large one.
   - Every wait is bounded by the caller's `deadline` (clock_gettime_nsec_np(CLOCK_MONOTONIC) nanoseconds, absolute),
     ETIMEDOUT at it; the start takes no deadline of its own.  Why: B1, "An execution's waits each take a fresh
     MESH_PLATFORM_NS deadline: the start's edges, then each `_arrived`.  So one execution is bounded by 5 s × (1 +
     arrivals).  The code must give one execution one absolute deadline, shared by all its waits" (mesh-call.h).
   What must stay: the waits before the raise, nothing raised where one fails; ESTALE where the plan's serial is not the
   map's; the raise before the fence, the serial's read after it
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-7:
   "Every wait ends at MESH_PLATFORM_NS without progress, and the first ETIMEDOUT ends the run ... The operator's bound
   is judged by time without progress"; W10: "B1: a deadline per execution is not the rule.  The rule is the
   operator's, on stalls: "nothing you're running should take more than around 8000ms", judged by time without
   progress"; mesh-call.h's note; "The revisions of 14b9a757", R-7), reversing `deadline` above:
   - The start takes no deadline.  Each wait on an edge's count ends MESH_PLATFORM_NS after the last rise of that count
     it read (its progress), the wait's start standing for the first read: ETIMEDOUT past it, nothing raised.  The
     count only rises (appendix item 2, "a count that only rises"), so a value read that differs from the one before is
     a rise.  A bounded wait: each one of os_sync_wait_on_address_with_timeout's ends by its bound, and the loop ends at
     ETIMEDOUT where the count stops rising short of its bound.
   - ESTALE where the plan's serial is not the installed view's identifier (mesh.h hdr `view`, acquire), where it was hdr
     `map_serial`: the plan's `serial` is the view it was made in (mesh.h struct mesh_state; aot A7, corrected
     2026-11-06: "A plan then stands while its view does, and its view while the sequencer's map does"), and a view's
     installation cancels every plan of another (mesh-records.c view_install, mesh-flow.c plan_stale), raising each
     edge's count to UINT64_MAX and waking it, so a waiter of the old view reads that or ESTALE.
   - After the raise, the fence, then hdr `view`'s read, ESTALE where it is not the plan's serial: the pairing above,
     with the view's installation (view_install stores `view`, then plan_stale fences and reads the plans' `executions`).
   What must stay: the waits before the raise, nothing raised where one fails; the raise before the fence, the serial's
   read after it */
/* the plan's row of waits into its call at place `place` (mesh.h struct mesh_state `rows`): row[0] the index at `syncs`
   of its first, row[1] one past its last */
static inline const uint32_t *mesh_wait_row(struct hdr *m,const struct mesh_state *state,uint64_t place){
  return (const uint32_t *)(const void *)((const char *)m+state->rows)+place;
}
int mesh_transfers_start(struct mesh_ctx *context,uint64_t plan,uint64_t executions){
  struct hdr *m=context->M;
  struct mesh_state *state=mesh_state_at(m,plan);
  uint64_t served=atomic_load_explicit(&state->executions,memory_order_relaxed);
  const uint64_t k=atomic_load_explicit(&state->submissions,memory_order_relaxed),K=k/state->calls;
  if(served<executions){
    struct mesh_sync *edge=mesh_syncs(m,state);
    const uint32_t *row=mesh_wait_row(m,state,k%state->calls);
    for(uint32_t e=row[0];e<row[1];e++){
      const __int128 need=(__int128)K*edge[e].every+edge[e].base;
      uint64_t seen=atomic_load_explicit(&edge[e].count,memory_order_acquire),
               until=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;
      while(seen<need){
        const uint64_t now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
        if(atomic_load_explicit(&m->view,memory_order_acquire)!=state->serial)return ESTALE;
        if(now>=until)return ETIMEDOUT;
        os_sync_wait_on_address_with_timeout(&edge[e].count,seen,sizeof edge[e].count,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                             OS_CLOCK_MACH_ABSOLUTE_TIME,until-now);
        const uint64_t risen=atomic_load_explicit(&edge[e].count,memory_order_acquire);
        if(risen!=seen){seen=risen;until=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;}
      }
    }
  }
  while(served<executions && !atomic_compare_exchange_strong_explicit(&state->executions,&served,executions,
                                                                      memory_order_release,memory_order_relaxed));
  if(served<executions){
    struct mesh_submission *queue=(struct mesh_submission *)(void *)((char *)m+state->submits);
    queue[k%state->submit_room]=(struct mesh_submission){.end=executions,.clock=++context->clock};
    atomic_store_explicit(&state->submissions,k+1,memory_order_release);
  }
  mesh_notify(&m->plans);
  atomic_thread_fence(memory_order_seq_cst);
  return atomic_load_explicit(&m->view,memory_order_relaxed)!=state->serial?ESTALE:0;
}

/* 2026-11-04, sub-assignment 5: mesh_plan_erased is removed (sub-assignment 7's 2026-11-03 wait, bounded by
   MESH_PLATFORM_NS, until no port held the previous plan whose supplied words and landed records the next plan zeroed
   and rewrote in place, and its seq_cst fence, Dekker's pair with the ports' data path).  Why: a new plan's state is
   in new pages, never the previous plan's, so nothing erases a plan or waits for one to be erased
   (design/disjoint-pages-change-request.md, Appendix C, C3.4, C3.6 and C4).  What must stay: a port's announcement of
   the generation it loads, which the release reads (mesh_retired_release; design/disjoint-pages-revision-
   coordination.md, D10) */
