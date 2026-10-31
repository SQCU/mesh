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
zappoff, zumbled, zorvek, zoblimp). Seven machines stand between the code and the block above; each is listed with
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
6. The GPU presents and the descriptor rings (zarpers: MESH_STREAM_ENTRIES, `taken`, the bridge's walk_take copy):
   the bridge runs compiled page lists per call (zinth hands it the call), its walk the executor of zirrup and
   zirrupel, which its entry word already is.
7. The landing gate and the operand staging copies (zibbins): transport by view; a premultiplier folded into zelloth.
