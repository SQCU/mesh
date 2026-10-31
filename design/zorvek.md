# zorvek

The structure of a group's encode and of the functions it runs, in pseudowords. Code that implements it uses these
names for its functions and its data, and C's own keywords as C's. The key below is the only translation; code cites
this file rather than restating it, so no idiom from another design is carried into the implementation.

```
zorvek(zimble):
  zuth zalashank zel zimble.zibbets ⊕ zimble.zonders ⊕ zimble.zennets:
    zalashank.zeeb = zappo(zalashank.zirn)
  zwoggle = zoblimp(zimble.zorks, zeeb)
  zinth(zwoggle)
  zabbadoo(zimble.zellos)
  zappoff(zeeb) zumble zabbadoo

zirrup(zub):
  zoop zub.zook: zwazz(zub)

zirrupel(zibbet, zub):
  zoop zibbet.zannet(zub): zwazz(zub)

zelloth(zonder, zennet):
  zarr zeeb zeppa zonder zuggles:
    zimp(zennet[zeeb.zarn], zeeb)
    zappoff(zeeb)
  zoop zuuz zimped: zennet zoft zoobit

zelloq(zibbet):
  zoop zibbet.zuuz: zozz

zantipoff: zorbles zandrels zumquats zibbins zethers zolvies zarpers zilbos zokkens zumpets
```

`(zennet[zeeb.zarn], zeeb)` is one tuple: a range of the sum and the landed pages folded into it.

## Key

| pseudoword | what it is |
|---|---|
| zorvek | encode one group |
| zimble | the group |
| zibbets | its landings (pages a peer's units land in) |
| zonders | its partials (contributions a reduction folds) |
| zennets | its sums (a reduction's result pages) |
| zalashank | one of those values |
| zirn | its bytes |
| zeeb | its pages |
| zappo | take pages from the page allocator (the bridge's region) |
| zappoff | give pages back to the allocator: their use released |
| zoblimp | lay span tables over the pages for the group's entries |
| zorks | the group's entries |
| zwoggle | the laid tables |
| zinth | present them to the bridge |
| zabbadoo | run the group's functions on the GPU |
| zellos | those functions |
| zumble zabbadoo | as the stream passes that work (stream-ordered) |
| zirrup | send a unit of the rank's own pages |
| zub | a unit |
| zook | its readiness (the rank's own pages: ready at encode) |
| zwazz | put a unit on the wire (a SEND) |
| zirrupel | forward a landed unit (the bridge alone, no GPU) |
| zannet | that unit's landing |
| zelloth | a reduction |
| zuggles | lands (whatever unit lands, in any order) |
| zimp | fold a contribution into its range exactly once, with no update lost |
| zarn | a unit's range in the sum |
| zuuz zimped | every contribution folded (the sum's one count) |
| zoft zoobit | goes to the next function |
| zelloq | a copy's consumption |
| zuuz | all of it landed |
| zozz | done |
| zantipoff | deleted |

Structural pseudowords in the block (zuth, zel, zoop, zarr, zeppa) are the host language's own loop and conditional
forms; code writes them as C.

## zantipoff, in order

zorbles: the generation slots (struct slot, slot_take, bind_slot, slot_laid's communicator storage). zandrels: the
generations (GENERATIONS, x->uses % GENERATIONS, the per-generation prepared-issue table). zumquats: the guards
(guard). zibbins: the gates (the let-go gate, gate_of's landing gate). zethers: the slot tickets. zolvies: the credit
SENDs. zarpers: the bridge's descriptor rings (MESH_STREAM_ENTRIES, taken, walk_take's copy). zilbos: the encoder
barriers between groups. zokkens: the combine chains (metal_after, combine-after-combine priors). zumpets: imposed
orders (the in-order cursors' serialization of independent tasks).

## What remains to delete, in the order the code allows

Done: zorbles, zandrels, zumquats, zethers and the let-go zibbin (f5276dbc: each issue on pages of its own, zappo,
zappoff, zumbled, zorvek, zoblimp). zelloth and zelloq for every group of copies and sums whose SENDs its presents
publish (a77e5b74: zabbadoo, metal_zelloth, present_publish); the pool runs only what zabbadoo does not take (fused
crossings, WRITEs, partials merges, a combine with a `need`, a SEND waiting on a consumption it cannot forward). Seven machines stand between the code and the block above; each is listed with
what of the block takes its place. The order is forced: each later one guards a hazard only an earlier one makes.

1. The pool's combines (zokkens, zumpets): tracks, in-order cursors, claims, the chunk protocol, `need`, priors,
   `metal_after` chains, and the accumulator made in place ("send op piece", later combines into the same bytes).
   In its place zelloth: for each page of a sum, one writer computes it once from the rank's own page and every
   contribution page for it, as soon as all of them have landed, in whatever order they landed; no atomics, since
   each page has one writer. Its count of pages done is the readiness word of a SEND of the sum (zirrupel's) and its
   completion. The sum's pages are the call's output or pages of its own (zennets), never bytes a SEND of the same
   call reads. zelloq needs no kernel: a copy's landing is the copy; the call's completion waits on `landed`.
2. The witness and retirement proofs (zolvies' remnant: mesh-routed.c retirement_proofs, byte_witness, witnessed;
   nccl-mesh.c bind_retirements, retired_first): they prove in-place overwrites safe; after 1 there are none.
3. The publication tasks and the pool's `ready` cursor: zook. The rank's own pages are ready when the group is
   encoded (one store with the presents); a sum's pages by zelloth's count.
4. The host's mirror of GPU progress (struct inflight, complete, retire, published/consumed): the bridge's counts and
   zumbled.
5. The encoder barriers between groups and the metal_boundary frontier (zilbos): with no transport on the GPU and no
   page shared between calls, a call's functions order after what they read in stream order alone (zumble).
6. Done: the GPU presents and the descriptor rings (zarpers: MESH_STREAM_ENTRIES, `taken` as ring room, the
   bridge's walk_take copy): the bridge walks each call's laid tables, linked at the call's place (zinth, below), its
   walk the executor of zirrup and zirrupel, which its entry word already is.
7. The landing gate and the operand staging copies (zibbins): transport by view; a premultiplier folded into zelloth.

## zinth: a call's tables linked to the bridge (item 6)

Measured 2026-10-31 (test-ov3, not landed): letting a "declared repeat"'s presents skip their barrier gave no rate and
deadlocked the chain service, a burst of small steps filling both ends' 4,096-entry rings with entries of different
calls in orders neither end could drain. Retracted with it: the repeat index and the overlap flag were a positional
rule standing in for a fact about pages, and so was the first form of this section's interface (73b5df63: a stream's
compiled program, a declared call order with per-call unit prefixes and bases relative to the call's place in it). A
call's pages are the allocator's (zappo), not a position in an order, so no order compiles them.

What runs (mesh.h struct mesh_zwoggle, nccl-mesh.c present_group and zinth, mesh-link.c walk_chain):

- zoblimp lays, in the issue's span-table lease (linear backing: the bridge reads it as one region interval), one
  table a channel: the call's send and landing entries and a `next` word.
- zinth is one store a channel at the call's place in the program, after the operand copies and the gate: the
  table's region offset into the word the holding's last call left for it (the stream's `zwoggle`, then that call's
  `next`). Links are stored in whatever order command buffers run; the chain is the holding's call order regardless.
  Nothing waits for ring room or for a predecessor's count; the GPU present kernel and metal_present are deleted.
- The bridge walks the chain (mesh-graph.h mesh_graph_chain, its operands the stream's tables), keeps no copy of an
  entry, and counts in each list's `taken` the calls it has left. A table is read until its walk leaves it, which
  happens only once the next call is linked, so its lifetime is that count, never a unit's completion (a list may
  carry no unit after it). An issue is taken again once both walks of each channel have left its tables (zumbled),
  and its table lease is returned only then (mesh-lease-retire.h MESH_LEASE_LEFT_SEND, MESH_LEASE_LEFT_LAND).
- Measured on all five, 2026-10-31: chain service exits 0, bandwidth 45 of 45 correct in each of two runs, every
  64 MiB rate within main's band (the A/A pair decides each outlier).

- An issue's entries are laid at its first call, each base relative to one of the call's bases (mesh.h struct
  mesh_zwoggle `bases`: each channel's units before the call on each list, the communicator's gates before it); a
  later call writes its bases and each table's `next` alone. A WRITE's source readiness word is its invocation's,
  which the issue's binding instance does not fix, so it alone is written each call.

- The rings are deleted (MESH_LAYOUT 156): MESH_STREAM_ENTRIES, the stream's `lists`, `sends` and `lands`, the
  bridge's list pages, walk_take and its ring of copied entries, mesh_stream_present. The quiet probe links one
  call's tables like any holder (mesh-diagnostics.c lay). A group still declares one word a list to its caller's
  schedule, the chain's head (`zwoggle`) in place of `presented`, so its crossing stays after the channel's earlier
  groups (their `ready` publications keep their order). chain.py's input frames are no longer bounded by the rings.
  Measured on all five: chain service exits 0, bandwidth 45 of 45 in each of three runs, every rate within main's
  band measured the same hour (two runs' scattered outliers, 2-3x on one case each, moved between cases and did not
  repeat).

## A landing's `after`: pages two calls are given

Where two calls are given the same pages, a later call's receive could be posted into them while an earlier call's
SEND of them was still on the wire: no call's completion waits on `sent`, and consecutive calls of one definition run
on different issues over the same caller's buffers (zorvek lays another where the last is not zumbled, which it never
is at the next call: its walk leaves its tables only once that call links). A relay forwarding from its recvbuff and an
in-place operand's SEND are the cases; the next call's landing there needs that SEND completed
(design/pages-and-functions.md Freeing 6-7: a page is written again only once its last outgoing DMA read ended).

- landing_readers (nccl-mesh.c), at an issue's first call: each landing's `after`, the channels on which a SEND of any
  issue the program lays reads the caller's bytes it lands into, and for each such definition, binding instance and
  channel the last of those SENDs. An issue's own pages are no other's, and its previous call's SENDs all completed
  before it is taken again (zumbled), so they need nothing.
- landing_afters, each call: on each channel, the end of that SEND in the latest call of every other issue that has had
  one (their laid ends), into the call's struct mesh_zwoggle_after; the bridge posts no unit of a landing whose bit is
  set until that channel's `sent` reaches it (mesh-link.c after_holds). MESH_LAYOUT 157.
- The pages' local reads (zelloth, the caller's kernels) are before the call's link in stream order: the barrier before
  zinth's first store orders them, and item 5 keeps it.
- The landing gate is already inert: its store is encoded before that barrier, so the bridge never reads a landing
  entry before the gate holds. Deleting it (item 7 above) removes no order.
- Measured first in a coarser form (test-after, not landed): every SEND before the call on each such channel, the
  issue's own included. It deadlocked the chain service on all five: m5ult4's and m5ult1's landing walks each held for
  SENDs the other's held receives kept from completing, SENDs past the real reader in the channel's order. The exact
  form (the last real reader, other issues' only), on all five, 2026-10-31: chain service exits 0, bandwidth 45 of 45
  in each of five runs. With the readers found at an issue's first call (test-after2), the issues the stream lays
  mid-loop scanned every SEND of the program inside the timed calls, and 64 MiB reduce-scatter and all-reduce ran up
  to 1.5x slower in two of three runs; found at preparation (mesh-graph.h mesh_graph_meets) and copied by an issue the
  stream lays from its instance's prepared one, every 64 MiB rate is within main's band (reduce-scatter 2741 and 2756
  us, all-reduce 4960 and 4884, against 2722 to 2760 and 4892 to 4897).

## The pool's last users, as the code stands at f4086e35

zabbadoo (nccl-mesh.c) takes a group whose every landing is a copy or a sum and whose every SEND its presents
published; complete() runs the rest, on the host's mirror (struct inflight, `published`, `consumed`) and, but for a
fused crossing, the tracks pool (metal_tracks, metal_record, metal_depend, metal_after). What it leaves there, and the
form each takes in the block above:

- A fused crossing (`fused`): records for its caller's kernels (metal_send_deferred, metal_land_deferred), no tracks,
  but on the mirror and `fusing`. Its form: the records written where the group is laid, its own pages' SENDs zook's.
- A declared frame train (`repeats` past 1: only WRITE plans declare frames; the chain service's input frames):
  zabbadoo's sums are keyed on a step's buffer and elements, not its frame, so two frames' contributions would fold
  into one sum. Its form: the frame in the key; zelloq unchanged (each channel's landings awaited past its last).
- A WRITE (`write_binding`): its SENDs name their own readiness words or are the rank's own pages (zook); a plain
  landing is zelloq's (a copy's landing is the copy); a landing with consumer records (`land_records`) is a wait its
  consumer's kernel makes (metal_land_deferred), encoded at the same place.
- A partials merge (METAL_MERGE, its row's floats): zelloth has no merge operator. Its form: zelloth with it.
- A SEND waiting on a consumption it cannot forward (present_publish stops at it: forwards() finds no receive that
  holds its piece, or send_dependencies drops a sum whose units do not line up with the landing's): the pool
  publishes it after that receive's whole consumption. Its form: the bridge's whole requirement on the landing's own
  word (`landed` or its summed word, every unit at the landing's end), which mesh_entry's per-unit form cannot state:
  an entry kind.
- A combine with a `need` (a sum into bytes an earlier SEND of its call on that channel reads: landing_needs): the
  pool runs it after that SEND's publication, the witness proofs (item 4) standing for the SEND's completion. Its
  form: the sum on pages of its own (zennets), never bytes a SEND of the same call reads, which is the compiler's
  placement of the sum (mesh-collective.c mesh_compile_sets' scratch), not a wait.

Then the pool and the mirror go: metal_tracks and its kernels, complete, retire, consume, publish, inflight and the
channels' `published`, `consumed`, `waited`.

### Moved, and found, from 2ac8f638

- A WRITE that is not a fused crossing: it publishes with its presents and zabbadoo takes it, frame trains
  included (2ac8f638). Its landings are declared to the caller's schedule (metal_boundary), where the pool's
  barrier used to stand.
- A partials merge: zelloth's fold is a kernel parameter, merge_row the merge's. Its forwards wait on summed
  words (84ac0814).
- A combine with a `need`: zabbadoo takes it (cb70ddb1). zabbadoo publishes every SEND with the presents, so the
  need is met before zelloth runs; the SEND's read finishing is the witness proofs' until item 4.
- Found, not in the inventory: a group that makes its operands in place (operand_copies: a premultiply, a copy
  within the region, a scratch's copy) never publishes with its presents (`direct` 0), so it always ran in the
  pool. Its form is a publication once the copies are made (launch_metal). Moving it exposed a second gap.
- Found: two steps of one call can combine into overlapping ranges that are not one sum, a later receive's
  combine over an earlier one's result. The pool orders the two (consume's metal_depend). zelloth does not: both
  sums fold the rank's own operand, so one loses the other's contributions. Bandwidth's 16 MiB reduce-scatter
  max came out 4.0 against 5.0 on rank 1 once groups that make copies reached zabbadoo. A group whose sums'
  result ranges are not disjoint stays the pool's (mesh_graph_disjoint). Its form: the later sum's own operand is
  the earlier sum's result, its blocks after that sum's done words, an explicit edge between the two sums.
- Found: the chain service stands, on main too, when the placement puts m5ult4-m5ult1 on one of its two
  cables (routing:["tp",0] wire 1555147594227890318 for node [0, 2]; the other wire passes). Each node's GPU
  then waits for a landing the other's SEND would make, and that SEND queues behind an earlier one on the same
  channel that is not ready yet. The cable abstraction in that sentence is the code's, not the model's: a
  channel is one cable (struct channel `cable`, `wire`), routing binds each piece to one wire, and the walk
  takes each wire's entries in host call order. In the model there are only edges, a node pair with all its
  cables and relay paths, and nothing is ordered but by a page. The one real constraint is a lane's: the
  transport is two-sided (TN3205, design/deliverables.md I1), so on a queue pair the k-th SEND lands in the
  k-th posted receive, an order both ends compute. The defect is that this order comes from host call
  order instead of the dependencies. It is not item 3's.
- fabric-dnn fails at 4f34df7d (before zinth), at 2b5a2fac and since: m5ult1's GPU wait ends with nothing it
  sends sent.
- Found: zelloth's threadgroups take blocks in a fixed order (block q to threadgroup q mod G) and spin on each
  one's inputs. A sum over an earlier sum's result, ordered by a wait on that sum's done word (each sum laid in a
  topological order of its steps' priors), stood bandwidth on all five twice, identically (mini: a zelloth
  input's word at 139008, 11080 calls linked; test-ov, test-ov2, 2026-10-31). The same stand came with or
  without waits for COPY landings. A wait inside a fixed order holds the threadgroups that the sums laid after
  it need, while some peer's input for the awaited sum is queued behind this rank's forward of those later sums.
  The pool never stands this way because it claims whichever task is ready. The form zelloth lacks: each
  threadgroup claims any block whose inputs and earlier sums hold (a scan, not a spin on one block), and each
  sum's summed words are published as the prefix of its computed blocks grows (the threadgroup that extends the
  prefix publishes it), so no threadgroup waits on another's publication. With it a sum can also wait on a
  COPY's landing. Until then, a group whose sums overlap stays the pool's (a01e4cd4).

### The pool and the mirror deleted (item 3's end)

- cf32b3f1: zelloth's threadgroups claim whichever block is ready (mesh-graph.h MESH_GRAPH_METAL, the GPU
  worklist), so a sum may wait on another without starving the sums laid after it. Overlapping sums are
  ordered by their steps' priors.
- 8cd0c480: sum steps fold together only at one depth (mesh_graph_depth), so a group's sums hold no cycle.
  That was bandwidth's last group in the pool.
- f82f0b15: zelloq's per-channel waits dispatch together (metal_spins).
- a9bd4e7a, 53c1fe72: the tracks pool deleted, from nccl-mesh.c, then the Metal layer and the kernels. A
  group zabbadoo cannot run fails when its plan is prepared. A fused crossing's records are fused_records.
- Then the host's mirror: the channels' `published`, `consumed`, `waited`, the list of groups in flight,
  retire and the tickets. Every group is published and awaited at its own call.
- Still open: the chain service stands intermittently, on main since before these (3 of its last 5 runs).
  At one stand m5ult2 had published nothing (`ready` 0 on every channel). m5ult3 waited on m5ult2's first
  landing. m5ult2 reaches m5ult4 only through m5ult1, which had relayed it 8 units of m5ult4's input frames,
  and those frames queue on m5ult4's channel behind its later input frames. Those wait for m5ult4's chain
  kernel, which waits on m5ult1's state, which waits on m5ult2's. That is a cycle through the chain
  program's input publication and a lane order taken from host call order, not through the library's
  waits. Its form: the compiler orders each lane's units by a topological order of the actual page
  dependencies, the cross-rank ones included (the chain's input publication order is declared in its
  production records). Units are spread over the whole edge's capacity, every cable between the two nodes
  and every relay path, so no unit waits behind one that depends on it. Host call order and cables are no
  inputs.
