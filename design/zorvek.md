# zorvek

Current source changes and remaining limits are recorded in [the compiler integration](compiler-integration-2026-10-31.md).

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
| zoblimp | lay tables over the pages for the group's entries |
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

Structural pseudowords in the block (zuth, zel, zoop, zarr, zeppa) are the host language's own loop and conditional
forms; code writes them as C.

zelloth: for each page of a sum, one writer computes it once from the rank's own page and every
contribution page for it, as soon as all of them have landed, in whatever order they landed; no atomics, since
each page has one writer. Its count of pages done is the readiness word of a SEND of the sum (zirrupel's) and its
completion. The sum's pages are the call's output or pages of its own (zennets), never bytes a SEND of the same
call reads. zelloq needs no kernel: a copy's landing is the copy; the call's completion waits on `landed`.

## zinth: a call's tables linked to the bridge

What runs (mesh.h struct mesh_zwoggle, nccl-mesh.c ncclMeshCommLowerOccurrence, mesh-link.c walk_chain):

- zinth is one store a channel: the
  table's region offset into the word the holding's last call left for it (the stream's `zwoggle`, then that call's
  `next`).
- The bridge walks the chain, keeps no copy of an
  entry, and counts in each list's `taken` the calls it has left. A table is read until its walk leaves it, which
  happens only once the next call is linked, so its lifetime is that count, never a unit's completion (a list may
  carry no unit after it). Each prepared occurrence owns its tables through execution. Lease retirement also
  retains descriptor reads (mesh-lease-retire.h MESH_LEASE_LEFT_SEND, MESH_LEASE_LEFT_LAND).
- The quiet probe links one call's tables like any holder (mesh-diagnostics.c lay).

## Found, from 2ac8f638

- fabric-dnn fails at 4f34df7d (before zinth), at 2b5a2fac and since: m5ult1's GPU wait ends with nothing it
  sends sent.
