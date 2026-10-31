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
