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
