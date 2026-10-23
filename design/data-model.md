# The data model

What main's code holds, who writes and reads each part, and how a call's pages move. Paths are under `rdma/`; line
numbers are at main (9b213e74). The bridge is mesh-flow (mesh-link.c, mesh-records.c). The collective is libnccl-mesh
(nccl-mesh.c, nccl-mesh-metal.m) and the libmesh calls it makes in its process (mesh-call.c, mesh-dataflow.c). Data
code is torch-mesh and the caller's own kernels.

## What exists

| Thing | Code name | Defined | What it holds |
|---|---|---|---|
| Region | `struct hdr`, shared memory `/mesh0` | mesh.h:169, layout mesh.h:236 | One per node: header, row planes, arena bits, page entries, buffers, the bridge's pages, 32 streams, 64 client slots, the arena. |
| Arena page | `pgsz` bytes from `data_off` | mesh.h:207, 215 | The registered memory. |
| Page | a row's block: `block` arena pages; counted in `struct mesh_pages` | mesh.h:68-82; mesh-link.c:152 | The unit of every count, entry, SEND and receive. |
| Row | `struct mesh_row`; `struct mesh_page_entry` and `struct mesh_buffer` at its index | mesh.h:55-65, 82 | `mapping` (lease slot, arena page) and `address` (region offset); `owner`, `closed`, `pages`. |
| Planes | `MESH_ROW_OWN`, `MESH_ROW_HOT`, `MESH_FREE`; the arena bits | mesh.h:52, 208-211 | A bit a row: claimed; leased and not reclaimed; released, to be reclaimed. A bit an arena page: in use. |
| Client slot | `struct mesh_client` | mesh.h:163 | `owner` (generation, slot, pid: mesh.h:196) and `start`. |
| Lease | `struct mesh_section`; the collective's `struct lease` | mesh-call.h:7; nccl-mesh.c:648 | `count` slots of rows from `first`, `stride` rows a slot, each `pages` arena pages, held in the slot's head row. |
| Stream | `struct mesh_stream` | mesh.h:136 | Four per port (mesh-observe.h:11), toward node `peer`: `live`, `wire`, `bandwidth`, `owner`, list counts, three page counts, lane counts, ring offsets. |
| Holding | a stream whose `owner` names a client; the bridge's `hold[t]` | mesh-call.h:17-25; mesh-bridge.h:142 | One client's use of one stream, from take to give. |
| Entry | `struct mesh_entry {row, pages, word, base}`; the bridge's copy, `struct taken` | mesh.h:76; mesh-bridge.h:131 | A range of the holder's rows and the word that gates it. |
| Lists | the send list and the landing list at `lists` | mesh.h:41, 128-130 | 4096 entry slots each, a ring. |
| Lane | `struct lane_state`; queue pair k of a link | mesh-bridge.h:42-57, 121 | Stream s uses lanes 2 + 2s and 3 + 2s. Page p of the stream's sequence goes on its lane p mod 2. |
| Unit | one posted SEND or receive of one page | mesh.h:96-110 | Counted in `posts`; its row kept in the stream's unit ring. |
| Link plan | `link_plan(budget)` | mesh-bridge.h:72 | The lanes a link makes, a bit a lane. |
| Program | `struct mesh_program`, `struct mesh_move`, `struct mesh_step` | mesh-plan.h:147-161 | A compiled collective: every rank's moves, by round. |
| Placement | `struct mesh_place`; `mesh_lay`, `mesh_group_lay` | mesh-plan.h:220-251 | Each step's entry index on its channel's list. |
| Channel | `struct channel` over a `struct hold` | nccl-mesh.c:247, 652 | A communicator's holding toward one peer rank's node. |
| Schedule | `struct schedule` (held in `plans`), `struct message` | nccl-mesh.c:1880-1885 | One group's entries on one channel, each with its rows and its end in pages. |
| Gate word | `gate_word`, `gate_at`, `gates` | nccl-mesh.c:317-318, 1207-1209 | A one-page lease holding a count of the communicator's groups. |
| Failure words | `violation`, `words` | nccl-mesh.c:319-320, 1204; nccl-mesh-metal.h:62-80 | A Metal shared buffer: cause, claim, entry, channel, bound, stop. |
| Time rings | `posted_at`, `landed_at` | mesh.h:42, 132-135 | 16384 slots: an entry's ticks at its first SEND posted and its last page landed. |
| Records | `struct mesh_record` at `map_off` | mesh.h:157 | Each node's region bytes, memory, model, health and the nodes its bridge reaches. |

## Who writes and who reads

| Word | Writer | Readers |
|---|---|---|
| header geometry, `node`, `boot` | bridge: region_open, mesh-link.c:1990-2023, 2147 | everyone |
| `version` | bridge (mesh-link.c:2001, 2016); heartbeat sets 0 (mesh-heartbeat.c:114) | every reader: mesh-dataflow.c:28; collective: nccl-mesh.c:670 |
| `map`, `map_off`, `map_nodes`, `health`, records | bridge: mesh-records.c | mesh.py and mesh-stat: mesh_map, mesh_nodes_health, mesh-dataflow.c:70-213 |
| `busy`, `idle_since` | bridge: busy_tell, mesh-records.c:6-13 | mesh_busy, mesh-dataflow.c:52-59 |
| slot `owner`, `start` | collective takes it (mesh-dataflow.c:256-267); collective or bridge retires it (mesh-dataflow.c:241-254; mesh-records.c:195-212) | bridge; heartbeat (mesh-heartbeat.c:146-148) |
| row `mapping`, `address` | collective at lease: mesh-dataflow.c:341-353, 377-383 | bridge: row_at, mesh-link.c:157-162; collective: mesh-call.c:124 |
| row `owner`, `pages` | collective at lease: mesh-call.c:102-106; cleared at reclaim, mesh-dataflow.c:386-398 | collective, bridge |
| row `closed` | collective or bridge at retire (mesh-dataflow.c:247); collective at return (mesh-dataflow.c:467) | bridge: mesh_retired_release, mesh-dataflow.c:437-452 |
| OWN bits | collective claims (mesh-dataflow.c:299-311) and clears (mesh-dataflow.c:466, 474); retire clears (mesh-dataflow.c:248) | every allocator |
| HOT bits | collective: set at mesh-call.c:107, cleared at mesh-dataflow.c:397 | every allocator |
| FREE bits | bridge: mesh-dataflow.c:446 | any allocator reclaims them: mesh-dataflow.c:335, 401-409 |
| stream `peer`, `live`, `bandwidth`, `wire`, `child`, `child_start` | bridge's port child: streams_open, mesh-link.c:801-823; `live` 0 at mesh-link.c:281, 924, 2034 | collective: mesh-call.c:14-58 |
| stream `lists`, `posted_at`, `landed_at`, `rows` | bridge: stream_lists, mesh-link.c:888-906 | collective through mesh_stream_view, mesh-call.c:66-77 |
| stream `owner` | collective: take mesh-call.c:42, give mesh-call.c:52; bridge: give for a gone client mesh-link.c:933-939, free mesh-link.c:228-232 | both |
| list slots, `sends`, `lands` | collective: mesh-call.c:78-90 | bridge: walk_take, mesh-link.c:184-199 |
| `taken[2]` | bridge: mesh-link.c:198 | collective: mesh-call.c:86; nccl-mesh.c:2090-2093 |
| `ready` | collective on the host (nccl-mesh.c:2250) or the GPU (nccl-mesh-metal.m:384-386; mesh.metal:92-95); data kernels on a fused crossing (nccl-mesh-kernels.h:101-103) | bridge: mesh-link.c:348, and as a landing gate, mesh-link.c:406 |
| `sent` | bridge: link_send, mesh-link.c:477; ~0 at loss, mesh-link.c:925 | collective host (nccl-mesh.c:2233, 2262) and GPU waits (nccl-mesh.c:2676); fused data kernels |
| `landed` | bridge: link_receive, mesh-link.c:512; ~0 at loss, mesh-link.c:926 | as `sent` |
| `posts`, unit ring | bridge: unit_told, mesh-link.c:244-250; mesh-link.c:468, 509 | mesh_pinned (mesh-dataflow.c:420-433); collective at take (nccl-mesh.c:1167-1169) |
| time rings | bridge: stream_time, mesh-link.c:145-150, 475, 512 | collective: ncclMeshTimes, nccl-mesh.c:3379-3413 |
| gate word | collective's GPU: metal_open_gate, nccl-mesh-metal.m:388-390 | bridge, as a landing entry's word: mesh-link.c:406 |
| failure words | GPU waits: nccl-mesh-kernels.h:76-89 | collective host: comm_failure, nccl-mesh.c:1638-1649 |
| `struct link`, `lane_state`, `send_gate`, `walk` | bridge alone: mesh-bridge.h:99-208 | bridge alone |

Data code reaches the region through ncclMemAlloc (torch-mesh/torch_mesh/_stream.mm:762) and, on a fused crossing,
through the six bindings of ncclMeshFused, which include a holding's counts (nccl.h:249, 1026-1035).

## How a call's pages move

| Stage | What happens | Carried by | Code |
|---|---|---|---|
| Leased | ncclMemAlloc claims rows and arena pages and binds each row to its page. | `mesh_buffer`, `mesh_page_entry`, OWN and HOT bits | nccl-mesh.c:1449-1474; mesh-call.c:93-121 |
| Laid | A group's steps become entries on each channel's two lists, in the order both ends make. Each entry is a range of a lease's rows. | `struct schedule`, `struct message` | nccl-mesh.c:1913-1968, 2023-2056; mesh-plan.h:250 |
| Presented | Each entry is written to a list slot, then the list's count moves. The bridge copies it and moves `taken`. | `mesh_entry`, `sends` or `lands`, `taken` | nccl-mesh.c:2125-2155; mesh-call.c:78-90; mesh-link.c:184-199 |
| Published | The holder stores the entry's end in `ready`. A send entry's word is `ready` (word 0); its base is the holding's pages before it. | `ready` | nccl-mesh.c:2246-2252, 2623-2648; mesh.metal:92-95 |
| Posted | Send thread: page i of an entry once `base + i < ready`, on the stream's lane p mod 2 (p its place in the stream's sequence), from the row's page. Receive thread: once the landing entry's word reaches `base`, into the row's page, within the lane's window. Each unit's row and count are stored before its post. | `posts[way].posted`, unit ring | mesh-link.c:244-250, 332-364, 393-420 |
| Sent | A SEND completion moves the lane's `done`, then `sent` over the pages completed in order. | `posts[0].done`, `sent`, `posted_at` | mesh-link.c:448-479 |
| Landed | A receive completion moves the lane's `got`, then `landed`. | `posts[1].done`, `landed`, `landed_at` | mesh-link.c:486-514 |
| Stamped | No word of its own. An entry is in once `landed` reaches its end. The host spins on it; a GPU wait loads it up to `polls` times. | `landed` | nccl-mesh.c:2228-2245, 2660-2698; mesh.metal:46 |
| Released | Rows for a next landing: its gate word reaches `base`. A lease: ncclMemFree returns it, or closes its head while a unit still names it; the bridge releases closed rows once none does. A stream: its owner word is given; the bridge frees it, or ends it for the pairing where units are outstanding. | gate word or `ready`; `closed`, FREE bits; stream `owner` | mesh-link.c:406; nccl-mesh.c:1487-1506; mesh-dataflow.c:420-470; mesh-records.c:209; mesh-call.c:48-53; mesh-link.c:271-323 |

A lost link stores ~0 in `sent` and `landed` (`MESH_PAGES_LOST`, mesh.h:95; mesh-link.c:919-929). An entry whose rows
leave the page table is sent as all-ones pages and lands in the link's spare page (mesh-link.c:345-351, 407-412).

## Departures from design/pages-and-functions.md

1. No row carries a stamp. A holding carries three page counts, `ready`, `sent` and `landed`, each over its list's
   order (mesh.h:143-145).
2. No row carries a use count. A row has an owner, a `closed` flag and three plane bits (mesh.h:52-59). Rows are not
   released per call; a lease is returned whole by ncclMemFree (nccl-mesh.c:1487-1506).
3. Nothing is zeroed when rows are reclaimed (mesh-dataflow.c:386-398).
4. A received page lands in a row the receiver leased and presented beforehand (mesh-link.c:393-420). No free page is
   chosen at landing, and no page goes back to the bridge.
5. The page table is not the only state. The region also holds client slots, streams, lists, unit rings, time rings
   and records. The bridge holds links, lanes, SEND gates and walks (mesh-bridge.h:99-208).
6. Main exposes streams and holdings to the collective. It lists free streams, takes one by `wire`, presents entries
   and reads the holding's counts (mesh-call.h:13-44; nccl-mesh.c:1076-1173).
7. Main exposes lane counts to the collective: `posts` in mesh_stream_view (mesh-call.h:34-38), summed at take
   (nccl-mesh.c:1167-1169).
8. Entries are laid and presented per group, at the call (nccl-mesh.c:2175-2189). Lists are rings of 4096 slots; a
   group waits for room (nccl-mesh.c:2101-2116).
9. On the Metal path a landing is gated by a word: the holding's `ready`, or the communicator's gate word, a count of
   groups (nccl-mesh.c:2064-2076).
10. GPU work waits. The library's kernels spin on `landed` and `sent` up to `polls` loads (nccl-mesh-metal.h:62-77;
    mesh.metal:46). Fused data kernels await them too (nccl.h:1030-1033).
11. The host path spins on the counts (nccl-mesh.c:2214-2274).
12. Errors come back as return codes. A count of ~0, or a holding no longer held, fails the call with ncclRemoteError
    (nccl-mesh.c:2223-2237, 2263-2265). A communicator keeps its failure and fails later groups
    (nccl-mesh.c:1616-1658). GPU failures go to failure words in a Metal buffer. No metadata pages sit beside `out`.
13. There is no accumulator page and no index page. A REDUCE lands in a lease of the communicator's pool and is
    combined into its step's buffer (nccl-mesh.c:2040, 2238-2242, 2677-2685).
14. Ranks exchange reports over bootstrap sockets at communicator init: node, nodes reached, free streams, pages
    posted (nccl-mesh.c:1076-1095, 1142-1173).
